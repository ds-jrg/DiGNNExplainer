from torch_geometric.utils import to_dense_batch
import omegaconf
import wandb

def to_dense(feature):

    Feat = to_dense_batch(x=feature)
    return PlaceHolder(Feat=Feat[0])

class PlaceHolder:
    def __init__(self, Feat):

        self.Feat = Feat

def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')