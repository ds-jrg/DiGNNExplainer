from torch_geometric.loader import DataLoader
import torch
from torch import nn
import torch_geometric as pyg
import torch.nn.functional as F
from torch_geometric.logging import log
import time
import torch_geometric.utils
import random
import numpy as np
from torch_geometric.utils.convert import from_networkx
import networkx as nx
from ba3motif_dataset import BA3Motif
import glob
import argparse

import torch
import os
from torch_geometric.loader import DataLoader
from torch_geometric.logging import log
import easydict
import torch_geometric.transforms as T
import matplotlib.pyplot as plt

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

parser = argparse.ArgumentParser()
args = easydict.EasyDict({
    "dataset": 'BA3',
    "batch_size": 128,
    "hidden_channels": 64,
    "lr": 0.0005,
    #"lr":1e-3,
    "epochs": 3000,
})
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(4, args.hidden_channels)
        self.conv2 = GCNConv(args.hidden_channels, args.hidden_channels)
        self.conv3 = GCNConv(args.hidden_channels, args.hidden_channels)
        self.lin = Linear(args.hidden_channels, 3)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)

        return x
def evaluate_gnn(graph):
    with torch.no_grad():
        model.eval()
        graph = graph.to(device)
        # The pred is not used for accuracy, the softmax is used
        out = model(graph.x.float(), graph.edge_index, graph.batch)

        # Getting class prediction probabilities from the softmax layer
        softmax = out.softmax(dim=-1)


        return softmax.tolist()
def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        
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
        
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

test_dataset = BA3Motif('data/BA3', mode="testing")
val_dataset = BA3Motif('data/BA3', mode="evaluation")
train_dataset = BA3Motif('data/BA3', mode="training")


test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)




graphs_path = '../graph generator/diffusion models/sampled_graphs_diffusion/'
print('Original no of graphs:',len([name for name in glob.iglob(graphs_path + 'ba3_15/*.gexf')]))

avg_max_pred_list = []
max_pred_list = []
graph_dict_list = []
softmax_dict_list = []
#all_edges_list = []
class_graphid_list = []


def get_max_pred(softmax_dict, i):
    # Getting the list of predictions for each class
    prob_class0_dict = {}
    prob_class1_dict = {}
    prob_class2_dict = {}

    for nodeid in softmax_dict:
        list0 = []
        list1 = []
        list2 = []

        if len(softmax_dict[nodeid]) > 0:
            list0 = []
            list1 = []
            list2 = []

            for prob in softmax_dict[nodeid]:
                list0.append(prob[0])
                list1.append(prob[1])
                list2.append(prob[2])

        # Taking max probability of all nodes of each class in a graph
        if len(list0) != 0:
            prob_class0_dict[nodeid] = max(list0)
        if len(list1) != 0:
            prob_class1_dict[nodeid] = max(list1)
        if len(list2) != 0:
            prob_class2_dict[nodeid] = max(list2)


    max_pred0 = max(prob_class0_dict.values())
    max_pred1 = max(prob_class1_dict.values())
    max_pred2 = max(prob_class2_dict.values())


    print('Run' + str(i), max_pred0, max_pred1, max_pred2)
    max_pred_list.append([max_pred0, max_pred1, max_pred2])

    graph_dict_list.append(graph_dict)
    #all_edges_list.append(all_edges)
    softmax_dict_list.append(softmax_dict)

    avg_max_pred = (max_pred0 + max_pred1 + max_pred2) / 3
    avg_max_pred_list.append(avg_max_pred)

    class0_graphid = max(prob_class0_dict, key=prob_class0_dict.get)
    class1_graphid = max(prob_class1_dict, key=prob_class1_dict.get)
    class2_graphid = max(prob_class2_dict, key=prob_class2_dict.get)


    class_graphid_list.append([class0_graphid, class1_graphid, class2_graphid])


eval_time = []
for i in range(0,10):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = GCN(hidden_channels=args.hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_test_acc = 0
    start_patience = patience = 100
    for epoch in range(1, 300 + 1):
        loss = train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        if epoch%100==0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

        if best_test_acc <= test_acc:
            #print('saving....')
            patience = start_patience
            best_test_acc = test_acc
            #print('best acc is', best_test_acc)

        else:
            patience -= 1

        if patience <= 0:
            print('Stopping training as validation accuracy did not improve '
                  f'for {start_patience} epochs')
            break

    all_edges = {}
    softmax_dict = {}
    nodefeature_dict = {}
    graph_dict = {}

    for filepath in glob.iglob(graphs_path + 'ba3_15/*.gexf'):
        graph = nx.read_gexf(filepath)
        
        filename = os.path.basename(filepath)
        graph_id = filename.split('.')[0]

        small_graph = from_networkx(graph)      
        small_graph.x = torch.ones(small_graph.num_nodes, 4)


        small_graph.pop('mode')
        small_graph.pop('weight')
        
        small_graph.pop('id')
        small_graph.pop('num_nodes')
        

        evaluation_start = time.time()
        softmax = evaluate_gnn(small_graph)
        softmax_dict[graph_id] = softmax
        graph_dict[graph_id]=graph
        eval_time.append(time.time() - evaluation_start)

    print("No. of graphs evaluated: ", len(softmax_dict))
    get_max_pred(softmax_dict, i)

print('Avg prob',np.mean(avg_max_pred_list))
path= 'ba3_plots/'
m = max(avg_max_pred_list)
index = avg_max_pred_list.index(m)


graph_sel_start = time.time()
class0_graphid = class_graphid_list[index][0]
class1_graphid = class_graphid_list[index][1]
class2_graphid = class_graphid_list[index][2]

max_pred0 = max_pred_list[index][0]
max_pred1 = max_pred_list[index][1]
max_pred2 = max_pred_list[index][2]

def plot_graph(graphid, max_pred, nodetype):
    graph = nx.read_gexf(graphs_path + 'ba3/' + graphid + '.gexf')

    print('Number of nodes: ', graph.number_of_nodes())
    print('graph id: ', graphid)
    print(f'Max pred probability for class {nodetype} is {max_pred}')

    nx.draw(graph,
            with_labels=False,
            node_size=100)
    plt.savefig(path + graphid + '.pdf')
    plt.show()

explanation_graph0 = plot_graph(class0_graphid, max_pred0, 0)
explanation_graph1 = plot_graph(class1_graphid, max_pred1, 1)
explanation_graph2 = plot_graph(class2_graphid, max_pred2, 2)

graph_sel_time = time.time() - graph_sel_start

print('Graph selection time', np.sum(eval_time) + graph_sel_time)


def get_faithfulness(graph_list):
    class_faithfulness = []
    for i, expln_graph in enumerate(graph_list):
        expln_graph = graph_dict.get(expln_graph)
        faith_score_list = []
        if i == 0:

            motifs_path = '../motifs/ba3/class0/'
        elif i == 1:

            motifs_path = '../motifs/ba3/class1/'
        elif i == 2:

            motifs_path = '../motifs/ba3/class2/'

        files_motif = os.listdir(motifs_path)

        for index_m, file_m in enumerate(files_motif):
            filepath_m = os.path.join(motifs_path, file_m)

            motif_graph = nx.read_gexf(filepath_m)

            GM = nx.algorithms.isomorphism.GraphMatcher(expln_graph, motif_graph)
            x = 1 if GM.subgraph_is_isomorphic() else 0
            faith_score_list.append(x)

        class_faithfulness.append(np.mean(faith_score_list))

    return np.mean(class_faithfulness)


faithfulness_list = []

for i in range(0,10):
    faithfulness = get_faithfulness(class_graphid_list[i])

    print('Run'+str(i),faithfulness)
    faithfulness_list.append(faithfulness)


print('Avg faithfulness', np.mean(faithfulness_list))
