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

from diffusion_model_discrete import DiscreteDenoisingDiffusion

warnings.filterwarnings("ignore", category=PossibleUserWarning)


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    if dataset_config['name'] == 'dblp':
        datadir = 'data/dblp'
    elif dataset_config['name'] == 'imdb':
        datadir = 'data/imdb'

    base_path = pathlib.Path(os.path.realpath(__file__)).parents[0]
    root_path = os.path.join(base_path, datadir)
    if os.path.exists(root_path):
        shutil.rmtree(root_path)

    from dataset import FeatureDataModule, FeatureDatasetInfos
    datamodule = FeatureDataModule(cfg)
    dataset_infos = FeatureDatasetInfos(datamodule)

    dataset_infos.compute_input_output_dims(datamodule=datamodule)
    model_kwargs = {'dataset_infos': dataset_infos}

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
    #use_gpu = False
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      num_sanity_val_steps=-1,
                      strategy="ddp_find_unused_parameters_false",
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger=[])

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)

    else:
        pass
    #add code for loading checkpoints


if __name__ == '__main__':
    main()
