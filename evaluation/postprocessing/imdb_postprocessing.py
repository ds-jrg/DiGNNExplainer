import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import numpy as np
import pandas as pd
import random


def remap_indices(node_list):
    val_list = [*range(0, len(node_list), 1)]
    return dict(zip(node_list,val_list)) 


def preprocess_edges(edgelist,node_list):
    res = [[node_list[i] for i, j in edgelist],[node_list[j] for i, j in edgelist]] 
    node_from = torch.tensor(res[0])
    node_to = torch.tensor(res[1])
    edges = torch.concat((node_from,node_to)).reshape(-1,len(node_from))
    return edges
    
    
def get_node_type(node_type):
    if node_type == 1:
        return 'actor'
    elif node_type == 0:
        return 'director'
    elif node_type == 2:
        return 'movie'


def create_dataset(nodes,edges,movie_df,director_df,actor_df,node_id,df_graph):
    all_edges = {}
    movie = np.asarray(movie_df.drop(columns=['class']))
    director = np.asarray(director_df)
    actor = np.asarray(actor_df)
    
    x_movie = torch.tensor(movie).to(dtype=torch.float32)
    y_movie = torch.tensor(np.array(movie_df["class"]), dtype=torch.long)
    x_director = torch.tensor(director).to(dtype=torch.float32)
    x_actor = torch.tensor(actor).to(dtype=torch.float32)
    
    #Edges
    source,dest =list(map(list, zip(*edges)))

    movie_to_director = []
    director_to_movie = []
    movie_to_actor = []
    actor_to_movie = []
    remaining_edges = []

    class_dict = {'Director':0, 'Actor':1, 'Movie':2}

    for i in range(len(edges)):
        if (df_graph.iloc[int(source[i])]['class'] == class_dict['Movie']) and \
        (df_graph.iloc[int(dest[i])]['class'] == class_dict['Director']):
                movie_to_director.append((int(source[i]),int(dest[i])))

        elif (df_graph.iloc[int(source[i])]['class'] == class_dict['Movie']) and \
            (df_graph.iloc[int(dest[i])]['class'] == class_dict['Actor']):
                movie_to_actor.append((int(source[i]),int(dest[i])))

        elif (df_graph.iloc[int(source[i])]['class'] == class_dict['Director']) and \
            (df_graph.iloc[int(dest[i])]['class'] == class_dict['Movie']):
                director_to_movie.append((int(source[i]),int(dest[i])))

        elif (df_graph.iloc[int(source[i])]['class'] == class_dict['Actor']) and \
            (df_graph.iloc[int(dest[i])]['class'] == class_dict['Movie']):
                actor_to_movie.append((int(source[i]),int(dest[i])))

        else:
            #The edges not present in the metagraph  
            source_node_type = get_node_type(df_graph.iloc[int(source[i])]['class'])
            dest_node_type = get_node_type(df_graph.iloc[int(dest[i])]['class'])
            remaining_edges.append((source_node_type,dest_node_type))
            
            
    all_edges[node_id] = [*movie_to_director,*movie_to_actor,*director_to_movie,*actor_to_movie]
            
    actor = list(df_graph[df_graph['class'] == 1]['nodeId'])
    actor = [int(i) for i in actor]
    actor_nodes_mapping = remap_indices(actor)
    
    director = list(df_graph[df_graph['class'] == 0]['nodeId'])
    director = [int(i) for i in director]
    director_nodes_mapping = remap_indices(director)
    
    movie = list(df_graph[df_graph['class'] == 2]['nodeId'])
    movie = [int(i) for i in movie]
    movie_nodes_mapping = remap_indices(movie)

    node_list = {}
    for d in [movie_nodes_mapping, director_nodes_mapping, actor_nodes_mapping]:
        node_list.update(d)            

    #Create Hetero Data      
    small_graph = HeteroData({'movie':{'x': x_movie, 'y':y_movie}, 
                              'director':{'x': x_director},'actor':{'x': x_actor}})

    if movie_to_director:
        edge_index_movie_director = preprocess_edges(movie_to_director,node_list)
        small_graph['movie','to','director'].edge_index = edge_index_movie_director
        
    if director_to_movie:
        edge_index_director_movie = preprocess_edges(director_to_movie,node_list)
        small_graph['director','to','movie'].edge_index = edge_index_director_movie
    
    if actor_to_movie:
        edge_index_actor_movie = preprocess_edges(actor_to_movie,node_list)
        small_graph['actor','to','movie'].edge_index = edge_index_actor_movie
        
    if movie_to_actor:
        edge_index_movie_actor = preprocess_edges(movie_to_actor,node_list)
        small_graph['movie','to','actor'].edge_index = edge_index_movie_actor
    
    #Removing isolated nodes
    transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])
    small_graph = transform(small_graph)

    #Adding test mask for prediction
    transform = T.RandomNodeSplit(split='train_rest', num_val=0.0, num_test=1.0)
    small_graph = transform(small_graph)
            
    return small_graph, remaining_edges, all_edges
    
    
def get_node_features(G,df_class0,df_class1,df_class2,director,actor):
    nodes = []
    director_node_features = []
    movie_node_features = []
    movie_class = []
    actor_node_features = []

    for key, value in G.nodes(data=True):

        nodes.append(key)
        edges = [e for e in G.edges]
        node_id = G.nodes[key]["label"]
        node_type = G.nodes[key]["node_type"]
        if node_type == 0:

            node_id = G.nodes[key]["label"]

            director_node = director.loc[int(node_id), :].values.flatten().tolist()
            director_node_features.append(director_node)
        elif node_type == 2:

            node_class = random.choice([0, 1, 2])

            if node_class == 0:
                
                movie_node = df_class0.loc[int(node_id), :].values.flatten().tolist()
                movie_class.append(0)
                movie_node_features.append(movie_node)
                
            elif node_class == 1:
                
                movie_node = df_class1.loc[int(node_id), :].values.flatten().tolist()
                movie_class.append(1)
                movie_node_features.append(movie_node)
                
            elif node_class == 2:
                
                movie_node = df_class2.loc[int(node_id), :].values.flatten().tolist()
                movie_class.append(2)
                movie_node_features.append(movie_node)

        elif node_type == 1:
   
            actor_node = actor.loc[int(node_id), :].values.flatten().tolist()
            actor_node_features.append(actor_node)


        movie_node_features_df = pd.DataFrame(movie_node_features)
        movie_node_features_df['class'] = movie_class
        director_node_features_df = pd.DataFrame(director_node_features)
        actor_node_features_df = pd.DataFrame(actor_node_features)

    return nodes, edges, movie_node_features_df, director_node_features_df, actor_node_features_df
