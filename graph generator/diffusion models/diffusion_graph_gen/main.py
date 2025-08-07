import graph_tool as gt
import os
import pathlib
import warnings
import shutil
import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from diffusion_model_discrete import DiscreteDenoisingDiffusion
from extra_features import DummyExtraFeatures, ExtraFeatures


warnings.filterwarnings("ignore", category=PossibleUserWarning)


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    MIN_NODES = MAX_NODES = 0
    dataset_config = cfg["dataset"]
    ds_name = cfg.general.dataset_name
    if ds_name in ['dblp','BA_shapes','Tree_Cycle','Tree_Grids']:
        MIN_NODES = 10
        MAX_NODES = 15

    elif ds_name == 'imdb':
        MIN_NODES = 5
        MAX_NODES = 10

    elif ds_name == 'mutag':
        MIN_NODES = 6
        MAX_NODES = 6

    if ds_name == 'ba3':
        MIN_NODES = 15
        MAX_NODES = 15

    for node_size in range(MIN_NODES, MAX_NODES+1):

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[0]
        path = os.path.join(base_path, cfg.dataset.datadir)

        if os.path.exists(path):
            shutil.rmtree(path)
        #node_size = cfg.general.node_size

        if dataset_config["name"] == 'planar':
            from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
            from spectre_utils import PlanarSamplingMetrics
            from visualization import NonMolecularVisualization

            datamodule = SpectreGraphDataModule(cfg,node_size)
            sampling_metrics = PlanarSamplingMetrics(datamodule)

            dataset_infos = SpectreDatasetInfos(cfg,datamodule)
            train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
            visualization_tools = NonMolecularVisualization()

            if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
                extra_features = ExtraFeatures(cfg.model.extra_features, dataset_config["name"], dataset_info=dataset_infos)
            else:
                extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

            dataset_infos.compute_input_output_dims(dataset_config["name"], datamodule=datamodule, extra_features=extra_features,
                                                    domain_features=domain_features)

            model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                            'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                            'extra_features': extra_features, 'domain_features': domain_features}

        elif dataset_config["name"] == 'mutag':
            from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
            from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
            from extra_features_molecular import ExtraMolecularFeatures
            from visualization import MolecularVisualization

            from datasets import mutag_dataset
            datamodule = mutag_dataset.MUTAGDataModule(cfg,node_size)
            dataset_infos = mutag_dataset.MUTAGinfos(datamodule, cfg)
            train_smiles = None

            if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
                extra_features = ExtraFeatures(cfg.model.extra_features, dataset_config["name"], dataset_info=dataset_infos)
                domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
            else:
                extra_features = DummyExtraFeatures()
                domain_features = DummyExtraFeatures()

            dataset_infos.compute_input_output_dims(dataset_config["name"], datamodule=datamodule, extra_features=extra_features,
                                                    domain_features=domain_features)

            if cfg.model.type == 'discrete':
                train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
            else:
                train_metrics = TrainMolecularMetrics(dataset_infos)

            # We do not evaluate novelty during training
            sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
            visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

            model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                            'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                            'extra_features': extra_features, 'domain_features': domain_features}
        else:
            raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

        utils.create_folders(cfg)

        if cfg.model.type == 'discrete':
            model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
        else:
            model = ''

        callbacks = []
        if cfg.train.save_model:
            checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                                  filename='{epoch}',
                                                  monitor='val/epoch_NLL',
                                                  save_top_k=5,
                                                  mode='min',
                                                  every_n_epochs=1)
            last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
            callbacks.append(last_ckpt_save)
            callbacks.append(checkpoint_callback)



        name = cfg.general.name


        use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
        trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                          strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                          accelerator='gpu' if use_gpu else 'cpu',
                          devices=cfg.general.gpus if use_gpu else 1,
                          max_epochs=cfg.train.n_epochs,
                          check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                          fast_dev_run=cfg.general.name == 'debug',
                          enable_progress_bar=False,
                          callbacks=callbacks,
                          log_every_n_steps=50 if name != 'debug' else 1,
                          logger = [])

        if not cfg.general.test_only:
            trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
            if cfg.general.name not in ['debug', 'test']:
                trainer.test(model, datamodule=datamodule)

        else:
            # Start by evaluating test_only_path
            trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
            if cfg.general.evaluate_all_checkpoints:
                directory = pathlib.Path(cfg.general.test_only).parents[0]
                print("Directory:", directory)
                files_list = os.listdir(directory)
                for file in files_list:
                    if '.ckpt' in file:
                        ckpt_path = os.path.join(directory, file)
                        if ckpt_path == cfg.general.test_only:
                            continue
                        print("Loading checkpoint", ckpt_path)
                        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()

