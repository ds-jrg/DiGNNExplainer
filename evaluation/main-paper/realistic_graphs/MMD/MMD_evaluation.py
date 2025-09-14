import sys
import os
sys.path.append("..")
import networkx as nx
import pandas as pd
from in_distribution.ood_stat import eval_graph_list
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MMD")
    parser.add_argument("--dataset", type=str, default="ba3")

    return parser.parse_args()

args = parse_args()

if args.dataset == 'dblp':
    CLASSES = [0, 1, 2, 3]

if args.dataset == 'imdb':
    CLASSES = [0, 1, 2]

if args.dataset == 'mutag':
    CLASSES = [0, 1]

if args.dataset == 'BA_shapes':
    CLASSES = [0, 1, 2, 3]

if args.dataset == 'Tree_Cycle':
    CLASSES = [0, 1]

if args.dataset == 'Tree_Grids':
    CLASSES = [0, 1]

if args.dataset == 'ba3':
    CLASSES = [0, 1, 2]


def get_graphs(files,path):
    graph_list = []
    for index, file in enumerate(files):

        if file.endswith('.gexf'):
            filepath = os.path.join(path, file)

            G_syn = nx.read_gexf(filepath)
            graph_list.append(G_syn)
    return graph_list

def mmd(expln_graph_path):

    class_files = os.listdir(expln_graph_path)
    graph_list = get_graphs(class_files,expln_graph_path)
    # Compute MMD
    MMD = eval_graph_list(orig_graph_list, graph_list, args.dataset)
    return MMD

def avg_mmd(expln_graph_path, explainer):
    MMD_list = []
    for target_class in CLASSES:
        expln_path = expln_graph_path + args.dataset + '/class' + str(target_class)
        MMD_list.append(mmd(expln_path))
    df = pd.DataFrame(MMD_list)
    mean = dict(df.mean())
    print(explainer, mean)


#Real graphs
path = '../real_graphs/' + args.dataset+'/'
files = os.listdir(path)
orig_graph_list = get_graphs(files,path)


# XGNN MMD
xgnn_mmd = avg_mmd( '../explanation_graphs/xgnn/', 'xgnn')
# GNNInterpreter MMD
gnnint_mmd = avg_mmd( '../explanation_graphs/gnnint/', 'gnninterpreter')
# D4Explainer MMD
gnnint_mmd = avg_mmd( '../explanation_graphs/d4/', 'd4explainer')
# VAE MMD
vae_mmd = avg_mmd( '../explanation_graphs/vae/', 'vae')
# DiGNNExplainer MMD
dignn_mmd = avg_mmd( '../explanation_graphs/dignn/', 'dignnexplainer')





