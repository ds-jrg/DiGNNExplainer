import torch
import utils
import diffusion_utils


class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            betas = diffusion_utils.cosine_beta_schedule_discrete(timesteps)

        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]


class MarginalUniformTransition:
    def __init__(self, f_marginals):

        self.F_classes = len(f_marginals)
        self.f_marginals = f_marginals

        self.u_f = f_marginals.unsqueeze(0).expand(self.F_classes, -1).unsqueeze(0)



    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy). """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)

        self.u_f = self.u_f.to(device)

        q_f = beta_t * self.u_f + (1 - beta_t) * torch.eye(self.F_classes, device=device).unsqueeze(0)

        return utils.PlaceHolder( Feat=q_f)

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)

        self.u_f = self.u_f.to(device)


        q_f = alpha_bar_t * torch.eye(self.F_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_f

        return utils.PlaceHolder(Feat=q_f)

class DiscreteUniformTransition:
    def __init__(self, f_classes: int):

        self.F_classes = f_classes

        self.u_f = torch.ones(1, self.F_classes, self.F_classes)
        if self.F_classes > 0:
            self.u_f = self.u_f/ self.F_classes



    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_f = self.u_f.to(device)

        q_f = beta_t * self.u_f + (1 - beta_t) * torch.eye(self.F_classes, device=device).unsqueeze(0)

        return utils.PlaceHolder(Feat = q_f)

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)

        self.u_f = self.u_f.to(device)

        q_f = alpha_bar_t * torch.eye(self.F_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_f

        return utils.PlaceHolder(Feat = q_f)

