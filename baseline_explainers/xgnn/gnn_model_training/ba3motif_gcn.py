
from ba3motif_dataset import BA3Motif
import argparse
import torch
from torch_geometric.loader import DataLoader
import easydict

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn import CrossEntropyLoss, Linear, ModuleList, ReLU, Softmax



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=300, help="Number of epoch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--hidden_channels", type=int, default=64, help="hidden size.")
    parser.add_argument("--dropout", type=float, default=0.4, help="dropout.")
    parser.add_argument("--num_unit", type=int, default=3, help="number of Convolution layers(units)")

    return parser.parse_args()

class GCN(torch.nn.Module):
    def __init__(self, num_unit):
        super().__init__()

        self.num_unit = num_unit

        self.node_emb = Linear(4, 64)

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        for i in range(num_unit):
            conv = GCNConv(in_channels=args.hidden_channels, out_channels=args.hidden_channels)
            self.convs.append(conv)
            self.relus.append(ReLU())

        self.lin1 = Linear(args.hidden_channels, args.hidden_channels)
        self.relu = ReLU()
        self.lin2 = Linear(args.hidden_channels, 3)
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
        x = F.dropout(x, p=args.dropout)
        for conv, relu in zip(self.convs, self.relus):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = relu(x)
        x = F.dropout(x, p=args.dropout)
        node_x = x
        return node_x

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        #print(data.batch)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=-1)
        #softmax = out.softmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

args = parse_args()
test_dataset = BA3Motif('../data/BA3', mode="testing")
val_dataset = BA3Motif('../data/BA3', mode="evaluation")
train_dataset = BA3Motif('../data/BA3', mode="training")


test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = GCN(args.num_unit).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

best_test_acc = 0
start_patience = patience = 100
for epoch in range(1, args.epoch + 1):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if epoch%100==0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

    if best_test_acc <= test_acc:
        print('saving....')
        patience = start_patience
        best_test_acc = test_acc
        print('best acc is', best_test_acc)
        torch.save(model.state_dict(), '../checkpoint/ba3_gcn.pth')

    else:
        patience -= 1

    if patience <= 0:
        print('Stopping training as validation accuracy did not improve '
              f'for {start_patience} epochs')
        break

