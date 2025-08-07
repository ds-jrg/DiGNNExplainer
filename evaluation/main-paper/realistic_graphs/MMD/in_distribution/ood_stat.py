import concurrent.futures
import copy
import os
import subprocess as sp
from datetime import datetime

import networkx as nx
import numpy as np
from collections import Counter

from scipy.linalg import eigvalsh
from utils import compute_mmd, gaussian, gaussian_emd

PRINT_TIME = False
ORCA_DIR = "orca"  # the relative path to the orca dir


def degree_worker(G):
    """
    Compute the degree distribution of a graph.
    :param G: a networkx graph
    :return: a numpy array of the degree distribution
    """
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    """
    Compute the distance between the degree distributions of two unordered sets of graphs.
    :param graph_ref_list: a list of networkx graphs
    :param graph_pred_list: a list of networkx graphs
    :param is_parallel: whether to use parallel computing
    :return: the distance between the two degree distributions
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


###################### Added for node type #########################################################
def get_node_dist(node_types):
    dist_dict = Counter(node_types)
    dist_dict = dict(sorted(dist_dict.items()))
    return list(dist_dict.values())

def node_type_worker(G):
    """
    Compute the marginal distribution of node types
    :param G: a list of node types
    :return: a numpy array of the node type distribution
    """

    node_types = list(nx.get_node_attributes(G, "node_type").values())
    node_dist = get_node_dist(node_types)
    marginal_prob = np.array(node_dist) / np.array(node_dist).sum()
    return marginal_prob

def node_type_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    """
    Compute the distance between the node type distributions of two unordered sets of graphs.
    :param graph_ref_list: a list of networkx graphs
    :param graph_pred_list: a list of networkx graphs
    :param is_parallel: whether to use parallel computing
    :return: the distance between the two node type distributions
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(node_type_worker, graph_ref_list):
                sample_ref.append(deg_hist)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(node_type_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist

###############################################################################
def spectral_worker(G):
    """
    Compute the spectral pmf of a graph.
    :param G: a networkx graph
    :return: a numpy array of the spectral pmf
    """
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    """
    Compute the distance between the degree distributions of two unordered sets of graphs.
    :param graph_ref_list: a list of networkx graphs
    :param graph_pred_list: a list of networkx graphs
    :param is_parallel: whether to use parallel computing
    :return: the distance between the two degree distributions
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist

def clustering_worker(param):
    """
    Compute the clustering coefficient distribution of a graph.
    :param param: a tuple of (graph, number of bins)
    :return: a numpy array of the clustering coefficient distribution
    """
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True):
    """
    Compute the distance between the clustering coefficient distributions of two unordered sets of graphs.
    :param graph_ref_list: a list of networkx graphs
    :param graph_pred_list: a list of networkx graphs
    :param bins: number of bins for the histogram
    :param is_parallel: whether to use parallel computing
    :return: the distance between the two clustering coefficient distributions
    """
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing clustering mmd: ", elapsed)
    return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {"3path": [1, 2], "4cycle": [8]}
COUNT_START_STR = "orbit counts: \n"


def edge_list_reindexed(G):
    """
    Convert a graph to a list of edges, where the nodes are reindexed to be integers from 0 to n-1.
    :param G: a networkx graph
    :return: a list of edges, where each edge is a tuple of integers
    """
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for u, v in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    """
    Compute the orbit counts of a graph.
    :param graph: a networkx graph
    :return: a numpy array of shape (n, 2), where n is the number of nodes in the graph. The first column is the node index, and the second column is the orbit count.
    """
    tmp_file_path = os.path.join(ORCA_DIR, "tmp.txt")
    f = open(tmp_file_path, "w")
    f.write(str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n")
    for u, v in edge_list_reindexed(graph):
        f.write(str(u) + " " + str(v) + "\n")
    f.close()

    output = sp.check_output([os.path.join(ORCA_DIR, "orca"), "node", "4", tmp_file_path, "std"])
    output = output.decode("utf8").strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array(
        [list(map(int, node_cnts.strip().split(" "))) for node_cnts in output.strip("\n").split("\n")]
    )

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list):
    """
    Compute the distance between the orbit counts of two unordered sets of graphs.
    :param graph_ref_list: a list of networkx graphs
    :param graph_pred_list: a list of networkx graphs
    :return: the distance between the two orbit counts
    """
    total_counts_ref = []
    total_counts_pred = []

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
        except Exception:
            continue

        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G)
        except Exception:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False, sigma=30.0)

    return mmd_dist


METHOD_NAME_TO_FUNC = {
    "degree": degree_stats,
    "node_type": node_type_stats,
    "cluster": clustering_stats,
    "orbit": orbit_stats_all,
    "spectral": spectral_stats

}

def eval_graph_list(graph_ref_list, grad_pred_list, dataset, methods=None):
    """
    Compute the evaluation metrics for a list of graphs.
    :param graph_ref_list: a list of networkx graphs
    :param grad_pred_list: a list of networkx graphs
    :param methods: a list of evaluation methods to be used
    :return: a dictionary of evaluation results
    """
    if methods is None:
        if dataset in ['dblp', 'imdb']:
            methods = ["node_type", "degree", "cluster", "spectral", "orbit"]
        else:
            methods = ["degree", "cluster", "spectral", "orbit"]


    results = {}
    for method in methods:
        results[method] = METHOD_NAME_TO_FUNC[method](graph_ref_list, grad_pred_list)
    if "orbit" not in methods:
        results["orbit"] = 0.0
    print(results)
    return results

