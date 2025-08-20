import numpy as np
import os
import pandas as pd
import networkx as nx
from sklearn.feature_selection import VarianceThreshold


def get_faithfulness(graphid_list,edges_list,motifs_path):
    class_faithfulness = []
    for i, graphid in enumerate(graphid_list):
        
        edges = edges_list[graphid]
        expln_graph = nx.Graph(edges)

        faith_score_list = []

        path = motifs_path + 'class' + str(i) + '/'
        
        files_motif = os.listdir(path)
        
        for index_m, file_m in enumerate(files_motif):
            filepath_m = os.path.join(path, file_m)
            
            motif_graph = nx.read_gexf(filepath_m)

            GM = nx.algorithms.isomorphism.GraphMatcher(expln_graph,motif_graph)
            x = 1 if GM.subgraph_is_isomorphic() else 0    
            faith_score_list.append(x)

        class_faithfulness.append(np.mean(faith_score_list))

    return np.mean(class_faithfulness)
    
def get_faithfulness_common(graphid_list,edges_list,motifs_path):
    class_faithfulness = []
    for graphid in graphid_list:
        
        edges = edges_list[graphid]
        expln_graph = nx.Graph(edges)

        faith_score_list = []
         
        files_motif = os.listdir(motifs_path)  
           
        for index_m, file_m in enumerate(files_motif):
            filepath_m = os.path.join(motifs_path, file_m)

            motif_graph = nx.read_gexf(filepath_m)

            GM = nx.algorithms.isomorphism.GraphMatcher(expln_graph,motif_graph)
            x = 1 if GM.subgraph_is_isomorphic() else 0    
            faith_score_list.append(x)

        class_faithfulness.append(np.mean(faith_score_list))

    return np.mean(class_faithfulness)
    

def selected_features_freq(X,feature_size):
    col_sum = X.sum(axis=0)
    sorted_colsum = sorted(col_sum, reverse=True)[:feature_size]

    colsum_df = pd.DataFrame(col_sum)
    index_list = list(np.ravel(colsum_df[colsum_df[0].isin(sorted_colsum)].index))

    imp_feat = X[index_list]
   
    return imp_feat
  
  
def selected_features_var(X,threshold):
    sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    fitted_X = sel.fit_transform(X)
    imp_feat = pd.DataFrame(fitted_X)

    return imp_feat
    
def index_2d(prob_list, v):
    for i, x in enumerate(prob_list):
        if v in x:
            return (i, x.index(v))
            

def print_stat_cont_features(df):
    print('mean',df.stack().mean())
    print('std dev',df.stack().std())
    
    
def apply_threshold(df):
    return df.applymap(lambda x: 0.0 if x<0.5 else 1.0)
    
def convert_to_discrete(df):
    return df.applymap(lambda x: 0.0 if x<0.5 else 1.0 if 0.5>=x<0.6 
                       else 2.0 if 0.6>=x<0.7 else 3.0 if 0.7>=x<0.8 else 4.0 if 0.8>=x<0.9 
                       else 5.0 if 0.9>=x<1.0 else 1.0)
