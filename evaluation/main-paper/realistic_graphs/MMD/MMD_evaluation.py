import sys
import os
sys.path.append("..")
import networkx as nx
import pandas as pd
from in_distribution.ood_stat import eval_graph_list
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MMD")
    parser.add_argument("--dataset", type=str, default="dblp")

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


def get_graphs(files):
    graph_list = []
    for index, file in enumerate(files):

        if file.endswith('.gexf'):
            filepath = os.path.join(path, file)

            G_syn = nx.read_gexf(filepath)
            graph_list.append(G_syn)
    return graph_list

#Real graphs
path = '../real_graphs/' + args.dataset+'/'
files = os.listdir(path)
orig_graph_list = get_graphs(files)


# #XGNN MMD
MMD_list = []
for target_class in CLASSES:
    path = '../explanation_graphs/xgnn/' + args.dataset+'/class' + str(target_class)
    class_files = os.listdir(path)
    graph_list = get_graphs(class_files)
    # Compute MMD
    MMD = eval_graph_list(orig_graph_list, graph_list, args.dataset)
    MMD_list.append(MMD)

df = pd.DataFrame(MMD_list)
mean = dict(df.mean())
print('xgnn',mean)

# GNNInterpreter MMD
MMD_list = []
for target_class in CLASSES:
    path = '../explanation_graphs/gnnint/' + args.dataset+'/class' + str(target_class)
    class_files = os.listdir(path)
    graph_list = get_graphs(class_files)
    # Compute MMD
    MMD = eval_graph_list(orig_graph_list, graph_list, args.dataset)
    MMD_list.append(MMD)

df = pd.DataFrame(MMD_list)
mean = dict(df.mean())
print('gnninterpreter',mean)

# D4Explainer MMD
MMD_list = []
for target_class in CLASSES:
    path = '../explanation_graphs/d4/' + args.dataset+'/class' + str(target_class)
    class_files = os.listdir(path)
    graph_list = get_graphs(class_files)
    # Compute MMD
    MMD = eval_graph_list(orig_graph_list, graph_list, args.dataset)
    MMD_list.append(MMD)

df = pd.DataFrame(MMD_list)
mean = dict(df.mean())
print('d4explainer',mean)

#DiGNNExplainer MMD
MMD_list = []
for target_class in CLASSES:
    path = '../explanation_graphs/dignn/' + args.dataset+'/class' + str(target_class)
    class_files = os.listdir(path)
    graph_list = get_graphs(class_files)
    # Compute MMD
    MMD = eval_graph_list(orig_graph_list, graph_list, args.dataset)
    MMD_list.append(MMD)

df = pd.DataFrame(MMD_list)
mean = dict(df.mean())
print('dignnexplainer',mean)

#VAE MMD
MMD_list = []
for target_class in CLASSES:
    path = '../explanation_graphs/vae/' + args.dataset+'/class' + str(target_class)
    class_files = os.listdir(path)
    graph_list = get_graphs(class_files)
    # Compute MMD
    MMD = eval_graph_list(orig_graph_list, graph_list, args.dataset)
    MMD_list.append(MMD)

df = pd.DataFrame(MMD_list)
mean = dict(df.mean())
print('vae',mean)




