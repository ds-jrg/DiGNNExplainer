import argparse
import torch

import matplotlib.pyplot as plt

from constants import feature_dict, task_type, dataset_choices
from explainers import *
from gnns import *
from utils.dataset import get_datasets
import numpy as np
torch.cuda.empty_cache()
import networkx as nx
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train explainers")
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument("--root", type=str, default="results/", help="Result directory.")
    parser.add_argument("--dataset", type=str, default="dblp", choices=dataset_choices)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--gnn_type", type=str, default="gcn")
    parser.add_argument("--task", type=str, default="nc")

    parser.add_argument("--train_batchsize", type=int, default=8)
    parser.add_argument("--test_batchsize", type=int, default=8)
    parser.add_argument("--sigma_length", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--feature_in", type=int)
    parser.add_argument("--data_size", type=int, default=-1)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha_cf", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.001)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--prob_low", type=float, default=0.0)
    parser.add_argument("--prob_high", type=float, default=0.4)
    parser.add_argument("--sparsity_level", type=float, default=2.5)

    parser.add_argument("--normalization", type=str, default="instance")
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--layers_per_conv", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--cat_output", type=bool, default=True)
    parser.add_argument("--residual", type=bool, default=False)
    parser.add_argument("--noise_mlp", type=bool, default=True)
    parser.add_argument("--simplified", type=bool, default=False)

    parser.add_argument("--nodes", type=int, default=15)

    return parser.parse_args()


args = parse_args()
args.noise_list = None
args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
args.feature_in = feature_dict[args.dataset]
args.task = task_type[args.dataset]
train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset)

train_dataset = train_dataset[: args.data_size]
gnn_path = f"param/gnns/{args.dataset}_{args.gnn_type}.pt"

explainer = DiffExplainer(args.device, gnn_path)
# start time
start = time.time()
# Train D4Explainer over train_dataset and evaluate
explainer.explain_graph_task(args, train_dataset, val_dataset)

########################### Model-Level ###############################################

if args.dataset == 'dblp':
    motifs_path = '../../evaluation/motifs/dblp/'
    CLASSES = [0, 1, 2, 3]
    MIN_NODES = 10
    MAX_NODES = 15

if args.dataset == 'imdb':
    motifs_path = '../../evaluation/motifs/imdb/'
    CLASSES = [0, 1, 2]
    MIN_NODES = 5
    MAX_NODES = 10

if args.dataset == 'mutag':
    motifs_path = '../../evaluation/motifs/mutag/'
    CLASSES = [0, 1]
    MIN_NODES = 6
    MAX_NODES = 6

if args.dataset == 'BA_shapes':
    motifs_path = '../../evaluation/motifs/BAshapes/'
    CLASSES = [0, 1, 2, 3]
    MIN_NODES = 10
    MAX_NODES = 15

if args.dataset == 'Tree_Cycle':
    motifs_path = '../../evaluation/motifs/TreeCycle/'
    CLASSES = [0, 1]
    MIN_NODES = 10
    MAX_NODES = 15

if args.dataset == 'Tree_Grids':
    motifs_path = '../../evaluation/motifs/TreeGrids/'
    CLASSES = [0, 1]
    MIN_NODES = 10
    MAX_NODES = 15

if args.dataset == 'ba3':
    motifs_path = '../../evaluation/motifs/ba3/'
    CLASSES = [0, 1, 2]
    MIN_NODES = 15
    MAX_NODES = 15


#Ground-truth faithfulness
def get_faithfulness(graph_list):
    class_faithfulness = []
    for i, A in enumerate(graph_list):

        expln_graph = nx.from_numpy_array(A)
        faith_score_list = []

        path = motifs_path + 'class' + str(i) + '/'

        files_motif = os.listdir(path)

        for index_m, file_m in enumerate(files_motif):
            filepath_m = os.path.join(path, file_m)

            motif_graph = nx.read_gexf(filepath_m)

            GM = nx.algorithms.isomorphism.GraphMatcher(expln_graph, motif_graph)
            x = 1 if GM.subgraph_is_isomorphic() else 0

            faith_score_list.append(x)

        class_faithfulness.append(np.mean(faith_score_list))

    return np.mean(class_faithfulness)

mean_faithfulness_list = []
avg_prob_list = []
for nodesize in range(MIN_NODES, MAX_NODES+1):
    MAX_NODES = nodesize
    print('NODE_SIZE', nodesize)

    expln_graphs_list = []
    for i in range(0, 10):
        print('Run'+str(i))
        expln_graphs = []
        avg_prob = []
        for target_class in CLASSES:

            adj, prob = explainer.multi_step_model_level(args, target_class, CLASSES, nodesize)
            avg_prob.append(prob)
            expln_graphs.append(adj.detach().cpu().numpy())
            print(target_class, adj, prob)
            G = nx.from_numpy_array(adj.detach().cpu().numpy())
            #nx.draw(G,node_size=100)
            #plt.savefig(args.dataset+str(target_class)+'.pdf')
            #plt.show()
            #plt.clf()
        avg_prob_list.append(np.mean(avg_prob))
        expln_graphs_list.append(expln_graphs)

    faithfulness_list = []

for i in range(0,10):

    faithfulness = get_faithfulness(expln_graphs_list[i])

    print('Run'+str(i),faithfulness)
    faithfulness_list.append(faithfulness)
    # print(np.mean(faithfulness_list))
    mean_faithfulness_list.append(np.mean(faithfulness_list))
 # end time
duration = time.time() - start
print('Total time(in secs)', duration)
print('Average faithfulness for all node sizes', np.mean(mean_faithfulness_list))
print('Average probability for all node sizes', np.mean(avg_prob_list))



