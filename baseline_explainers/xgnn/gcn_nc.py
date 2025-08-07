#https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py
import argparse
import torch
import torch.nn.functional as F

from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')

args = parser.parse_args()


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        return x


