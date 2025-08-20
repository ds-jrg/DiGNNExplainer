import os
import numpy as np
import networkx as nx
import argparse
import matplotlib.pyplot as plt
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Train explainers")
    parser.add_argument("--dataset", type=str, default="dblp")
    #parser.add_argument("--nodes", type=int, default=10)
    return parser.parse_args()

args = parse_args()

if args.dataset == 'dblp':
    from dblp_gnn_explain import gnn_explain
    motifs_path = '../../evaluation/motifs/dblp/'
    CLASSES = [0, 1, 2, 3]
    MIN_NODES = 10
    MAX_NODES = 15

if args.dataset == 'imdb':
    from imdb_gnn_explain import gnn_explain
    motifs_path = '../../evaluation/motifs/imdb/'
    CLASSES = [0, 1, 2]
    MIN_NODES = 5
    MAX_NODES = 10


if args.dataset == 'mutag':
    from mutag_gnn_explain import gnn_explain
    motifs_path = '../../evaluation/motifs/mutag/'
    CLASSES = [0, 1]
    MIN_NODES = 6
    MAX_NODES = 6

if args.dataset == 'BA_shapes':
    from BA_shapes_gnn_explain import gnn_explain
    motifs_path = '../../evaluation/motifs/BAshapes/'
    CLASSES = [0, 1, 2, 3]
    MIN_NODES = 10
    MAX_NODES = 15

if args.dataset == 'Tree_Cycle':
    from Tree_Cycle_gnn_explain import gnn_explain
    motifs_path = '../../evaluation/motifs/TreeCycle/'
    CLASSES = [0, 1]
    MIN_NODES = 10
    MAX_NODES = 15

if args.dataset == 'Tree_Grids':
    from Tree_Grids_gnn_explain import gnn_explain
    motifs_path = '../../evaluation/motifs/TreeGrids/'
    CLASSES = [0, 1]
    MIN_NODES = 10
    MAX_NODES = 15

if args.dataset == 'ba3':
    from ba3_gnn_explain import gnn_explain
    motifs_path = '../../evaluation/motifs/ba3/'
    CLASSES = [0, 1, 2]
    MIN_NODES = 15
    MAX_NODES = 15

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
    for i in range(0, 1):
        print('Run'+str(i))
        expln_graphs = []
        avg_prob = []
        # start time
        start = time.time()
        for target_class in CLASSES:

            explainer = gnn_explain(nodesize, 30,  target_class, 50)  ####arguments: (max_node, max_step, target_class, max_iters)
            adj, prob = explainer.train()
            avg_prob.append(prob)
            expln_graphs.append(adj.detach().cpu().numpy())
            print(target_class, adj, prob)

            # G = nx.from_numpy_array(adj.detach().cpu().numpy())
            # nx.draw(G,node_size=100)
            # plt.savefig(args.dataset+str(target_class)+'.pdf')
            # plt.show()
            # plt.clf()

        # end time
        duration = time.time() - start
        print('Total Time for all classes(in secs)', duration)

        avg_prob_list.append(np.mean(avg_prob))
        expln_graphs_list.append(expln_graphs)

    faithfulness_list = []

    for i in range(0,1):
        faithfulness = get_faithfulness(expln_graphs_list[i])

        print('Run'+str(i),faithfulness)
        faithfulness_list.append(faithfulness)
    # print(np.mean(faithfulness_list))
    mean_faithfulness_list.append(np.mean(faithfulness_list))

print('Average faithfulness for all node sizes', np.mean(mean_faithfulness_list))
print('Average probability for all node sizes', np.mean(avg_prob_list))


