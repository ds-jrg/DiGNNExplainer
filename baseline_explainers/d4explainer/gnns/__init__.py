import torch_geometric
ver = torch_geometric.__version__
if ver == '2.4.0':
    from .real_nc_gnn import Real_NC_GCN
    __all__ = [
        "Real_NC_GCN"
    ]
elif ver == '2.0.4':
    from .ba3motif_gnn import BA3MotifNet
    from .mutag_gnn import Mutag_GCN
    from .syn_nc_gnn import Syn_NC_GCN
    from .tree_grids_gnn import Syn_GCN_TG

    __all__ = [
        "BA3MotifNet",
        "Mutag_GCN",
        "Syn_NC_GCN",
        "Syn_GCN_TG"
    ]


