import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

import utils

class MLPBlock(nn.Module):
    """
        dim_feedforward: the dimension of the feedforward network model
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """

    def __init__(self, dfeat: int, dim_ffFeat: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()


        self.linFeat1 = Linear(dfeat, dim_ffFeat, **kw)
        self.linFeat2 = Linear(dim_ffFeat, dfeat, **kw)
        self.normFeat1 = LayerNorm(dfeat, eps=layer_norm_eps, **kw)
        self.normFeat2 = LayerNorm(dfeat, eps=layer_norm_eps, **kw)
        self.dropoutFeat1 = Dropout(dropout)
        self.dropoutFeat2 = Dropout(dropout)
        self.dropoutFeat3 = Dropout(dropout)

        self.activation = F.silu # using silu instead of relu improves results for 8 features

    def forward(self, Feat: Tensor):

        newFeat_d = self.dropoutFeat1(Feat)
        Feat = self.normFeat1(newFeat_d)

        ff_outputFeat = self.linFeat2(self.dropoutFeat2(self.activation(self.linFeat1(Feat))))
        ff_outputFeat = self.dropoutFeat3(ff_outputFeat)
        Feat = self.normFeat2(Feat + ff_outputFeat)

        return Feat


class MLP(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """

    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers

        self.out_dim_Feat = output_dims['Feat']

        self.mlp_in_Feat = nn.Sequential(nn.Linear(input_dims['Feat'], hidden_mlp_dims['Feat']),
                                         #act_fn_in,
                                         nn.SiLU(),
                                         nn.Linear(hidden_mlp_dims['Feat'], hidden_dims['dfeat'])
                                         #,act_fn_in
                                        ,nn.SiLU()
                                         )

        self.tf_layers = nn.ModuleList([MLPBlock(
                                                dfeat=hidden_dims['dfeat'],
                                                dim_ffFeat=hidden_dims['dim_ffFeat'])
                                        for i in range(n_layers)])


        self.mlp_out_Feat = nn.Sequential(nn.Linear(hidden_dims['dfeat'], hidden_mlp_dims['Feat']),
                                          act_fn_out,
                                          nn.SiLU(),
                                          nn.Linear(hidden_mlp_dims['Feat'], output_dims['Feat']))

    def forward(self, Feat):

        Feat_to_out = Feat[..., :self.out_dim_Feat]

        after_in = utils.PlaceHolder(Feat=self.mlp_in_Feat(Feat))
        Feat = after_in.Feat

        for layer in self.tf_layers:
            Feat = layer(Feat)

        Feat = self.mlp_out_Feat(Feat)

        Feat = (Feat + Feat_to_out)

        return utils.PlaceHolder(Feat=Feat)