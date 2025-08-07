import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb

from mlp_model import MLP
from noise_schedule import PredefinedNoiseScheduleDiscrete, \
    MarginalUniformTransition, DiscreteUniformTransition
import diffusion_utils
from train_metrics import TrainLossDiscrete
from abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
import utils


class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Fdim = input_dims['Feat']
        self.Fdim_output = output_dims['Feat']

        self.dataset_info = dataset_infos
        self.train_loss = TrainLossDiscrete()

        self.val_nll = NLL()
        self.val_F_kl = SumExceptBatchKL()
        self.val_F_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_F_kl = SumExceptBatchKL()
        self.test_F_logp = SumExceptBatchMetric()

        self.model = MLP(n_layers=cfg.model.n_layers,
                         input_dims=input_dims,
                         hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                         hidden_dims=cfg.model.hidden_dims,
                         output_dims=output_dims,
                         act_fn_in=nn.ReLU(),
                         act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(f_classes=self.Fdim_output)
            f_limit = torch.ones(self.Fdim_output) / self.Fdim_output

            self.limit_dist = utils.PlaceHolder(Feat=f_limit)
        elif cfg.model.transition == 'marginal':
            features = self.dataset_info.feature_types.float()
            f_marginals = features / torch.sum(features)

            print(
                f"Marginal distribution of the classes: {f_marginals} for features")
            self.transition_model = MarginalUniformTransition(f_marginals=f_marginals)
            self.limit_dist = utils.PlaceHolder(Feat=f_marginals)

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps

        self.best_val_nll = 1e8
        self.val_counter = 0

    def training_step(self, data, i):
        noisy_data = self.apply_noise(data.feature)

        pred = self.forward(noisy_data)
        loss = self.train_loss(masked_pred_Feat=pred.Feat,
                               true_Feat=data.feature,
                               log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", str(self.cfg.dataset.node_feature_size))
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        #self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_F_kl.reset()
        self.val_F_logp.reset()

    # self.sampling_metrics.reset()

    def validation_step(self, data, i):
        noisy_data = self.apply_noise(data.feature)

        pred = self.forward(noisy_data)
        nll = self.compute_val_loss(pred, noisy_data, data.feature, test=False)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(),
                   self.val_F_kl.compute() * self.T,
                   self.val_F_logp.compute()]
        if wandb.run:
            wandb.log({"val/F_logp": metrics[2]},
                      commit=False)

        self.print(f"Epoch {self.current_epoch}:",
                   f"Val F_logp: {metrics[2] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_F_kl.reset()
        self.test_F_logp.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        noisy_data = self.apply_noise(data.feature)
        pred = self.forward(noisy_data)
        nll = self.compute_val_loss(pred, noisy_data,data.feature, test=True)
        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(),
                   self.test_F_kl.compute() * self.T,
                   self.test_F_logp.compute()]
        # if wandb.run:
        #     wandb.log({
        #         "test/epoch_NLL": metrics[0],
        #                "test/F_kl": metrics[1],
        #                "test/F_logp": metrics[2]},
        #               commit=False)

        # self.print(f"Epoch {self.current_epoch}:",
        #            f"Test F_logp: {metrics[2] :.2f}")

        test_nll = metrics[0]
        # if wandb.run:
        #     wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        #self.print(f'Test loss: {test_nll :.4f}')

        samples_to_generate = self.cfg.general.final_model_samples_to_generate

        samples = self.sample_batch(samples_to_generate)

        if self.cfg.dataset.node_type != '':
            torch.save(torch.tensor(samples.Feat),self.cfg.dataset.node_type+str(self.cfg.dataset.node_class)+
                       '_'+str(self.cfg.dataset.node_feature_size)+'feat.pt')
        else:
            torch.save(torch.tensor(samples.Feat),self.cfg.dataset.node_type  + '_' + str(self.cfg.dataset.node_feature_size) + 'feat.pt')
        self.print("Done testing.")

    def kl_prior(self, Feat):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((Feat.size(0), 1), device=Feat.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probFeat = Feat @ Qtb.Feat
        bs, n, _ = probFeat.shape

        limit_Feat = self.limit_dist.Feat[None, None, :].expand(bs, n, -1).type_as(probFeat)

        kl_distance_Feat = F.kl_div(input=probFeat.log(), target=limit_Feat, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_Feat)

    def compute_Lt(self, Feat, pred, noisy_data, test):

        pred_probs_Feat = F.softmax(pred.Feat, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        prob_true = diffusion_utils.posterior_distributions(Feat=Feat,
                                                            Feat_t=noisy_data['Feat_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)

        prob_pred = diffusion_utils.posterior_distributions(Feat=pred_probs_Feat,

                                                            Feat_t=noisy_data['Feat_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)

        kl_f = (self.test_F_kl if test else self.val_F_kl)(prob_true.Feat, torch.log(prob_pred.Feat))
        return self.T * kl_f

    def reconstruction_logp(self, t, Feat):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probFeat0 = Feat @ Q0.Feat  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probF=probFeat0)

        Feat0 = F.one_hot(sampled0.Feat, num_classes=self.Fdim_output).float()

        assert (Feat.shape == Feat0.shape)

        sampled_0 = utils.PlaceHolder(Feat=Feat0)

        # Predictions
        noisy_data = {'Feat_t': sampled_0.Feat,
                      't': torch.zeros(Feat0.shape[0], 1)}
        pred0 = self.forward(noisy_data)

        # Normalize predictions
        probFeat0 = F.softmax(pred0.Feat, dim=-1)

        return utils.PlaceHolder(Feat=probFeat0)

    def apply_noise(self, Feat):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(Feat.size(0), 1), device=Feat.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar,
                                               device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.Feat.sum(dim=2) - 1.) < 1e-4).all(), Qtb.Feat.sum(dim=2) - 1

        # Compute transition probabilities
        probF = Feat @ Qtb.Feat

        sampled_t = diffusion_utils.sample_discrete_features(probF=probF)

        Feat_t = F.one_hot(sampled_t.Feat, num_classes=self.Fdim_output).float()

        assert (Feat.shape == Feat_t.shape)

        z_t = utils.PlaceHolder(Feat=Feat_t)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'Feat_t': z_t.Feat}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, Feat, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        #kl_prior = self.kl_prior(Feat)

        # 3. Diffusion loss
        #loss_all_t = self.compute_Lt(Feat, pred, noisy_data,test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (Feat | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, Feat)

        #Feature loss added
        feature_loss = self.val_F_logp(Feat * prob0.Feat.log())
        print('feature_loss', feature_loss)

        #Combine terms
        # nlls = kl_prior + loss_all_t - feature_loss
        # assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        #nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        if wandb.run:
            wandb.log({
                "feature_loss": feature_loss})
        return feature_loss

    def forward(self, noisy_data):

        return self.model(noisy_data['Feat_t'])

    @torch.no_grad()
    def sample_batch(self, batch_size: int):

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        num_features = self.cfg.dataset.node_feature_size
        num_feature_types = self.cfg.dataset.node_feature_types
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist,num_nodes=batch_size,
                                                            num_features=num_features, num_feature_types=num_feature_types)
        Feat = z_T.Feat

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1))
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, Feat.to(self.device))


        # Sample
        Feat = sampled_s.Feat

        return Feat

    def sample_p_zs_given_zt(self, s, t, Feat_t):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""

        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'Feat_t': Feat_t, 't': t}
        pred = self.forward(noisy_data)

        # Normalize predictions
        pred_Feat = F.softmax(pred.Feat, dim=-1)  # bs, n, d0


        p_s_and_t_given_0_Feat = diffusion_utils.compute_batched_over0_posterior_distribution(Feat_t=Feat_t,
                                                                                           Qt=Qt.Feat,
                                                                                           Qsb=Qsb.Feat,
                                                                                           Qtb=Qtb.Feat)


        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_Feat = pred_Feat.unsqueeze(-1) * p_s_and_t_given_0_Feat  # bs, n, d0, d_t-1
        unnormalized_prob_Feat = weighted_Feat.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_Feat[torch.sum(unnormalized_prob_Feat, dim=-1) == 0] = 1e-5
        prob_Feat = unnormalized_prob_Feat / torch.sum(unnormalized_prob_Feat, dim=-1, keepdim=True)  # bs, n, d_t-1



        assert ((prob_Feat.sum(dim=-1) - 1).abs() < 1e-4).all()


        sampled_s = diffusion_utils.sample_discrete_features(prob_Feat)

        out_discrete = utils.PlaceHolder(Feat=sampled_s)

        return out_discrete
