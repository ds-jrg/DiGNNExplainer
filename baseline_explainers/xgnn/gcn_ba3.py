import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
from torch.nn import  Linear, ModuleList, ReLU, Softmax
from torch_geometric.nn import GCNConv, global_mean_pool

EPS = 1

class GCN(torch.nn.Module):
    def __init__(self, num_unit):
        super().__init__()

        self.num_unit = num_unit

        self.node_emb = Linear(4, 64)

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        for i in range(num_unit):
            conv = GCNConv(in_channels=64, out_channels=64)
            self.convs.append(conv)
            self.relus.append(ReLU())

        self.lin1 = Linear(64, 64)
        self.relu = ReLU()
        self.lin2 = Linear(64, 3)
        self.softmax = Softmax(dim=1)

    def forward(self, x, edge_index, batch):
        edge_attr = torch.ones((edge_index.size(1),), device=edge_index.device)
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        pred = self.relu(self.lin1(graph_x))
        pred = self.lin2(pred)
        return pred

    def get_node_reps(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        x = F.dropout(x, p=0.4)
        for conv, relu in zip(self.convs, self.relus):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = relu(x)
        x = F.dropout(x, p=0.4)
        node_x = x
        return node_x

