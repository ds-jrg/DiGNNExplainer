import networkx as nx
import torch
import torch.nn as nn
import numpy as np
import copy
from policy_nn import PolicyNN

import gcn_nc
import torch.optim as optim

import matplotlib.pyplot as plt

from torch_geometric.data import Data



class gnn_explain():
    def __init__(self, max_node, max_step, target_class, max_iters): 
        print('Start training pipeline')
        self.graph= nx.Graph()
        self.max_node = max_node
        self.max_step = max_step
        self.max_iters = max_iters
        self.num_class = 4
        self.node_type = 4
        self.learning_rate = 0.01
        self.roll_out_alpha = 2
        self.roll_out_penalty = -0.1
        self.policyNets= PolicyNN(self.node_type, self.node_type)
        self.gnnNets = gcn_nc.GCN(4,32,4)

        self.reward_stepwise= 0.1
        self.target_class = target_class
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.policyNets.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)

        self.max_poss_degree = {0: 55, 1: 3, 2: 2, 3: 3}
        

    def train(self):
        ####given the well-trained model
        ### Load the model
        checkpoint = torch.load('checkpoint/BA_shapes_gcn.pth')
        self.gnnNets.load_state_dict(checkpoint)
        
        for i in range(self.max_iters):
            self.graph_reset()
            for j in range(self.max_step):
                self.optimizer.zero_grad()
                reward_pred = 0
                reward_step = 0
                n = self.graph.number_of_nodes()
                if(n>self.max_node):
                    break
                self.graph_old = copy.deepcopy(self.graph)
                ###get the embeddings
                X, A = self.read_from_graph(self.graph)
           #
                X = torch.from_numpy(X)
                A = torch.from_numpy(A)
                ### Feed to the policy nets for actions
                start_action, start_logits_ori, tail_action, tail_logits_ori  = self.policyNets(X.float(), A.float(), n+self.node_type)

                #flag is used to track whether adding operation is success/valid.
                if(tail_action>=n): ####we need add node, then add edge
                    if(n==self.max_node):
                        flag = False
                    else:
                        self.add_node(self.graph, n, tail_action.item()-n)
                        flag = self.add_edge(self.graph, start_action.item(), n)
                else:
                    flag= self.add_edge(self.graph, start_action.item(), tail_action.item())
                
                if flag == True:
                    validity = self.check_validity(self.graph)
                
                
                if  flag == True: #### add edge  successfully
                    if validity == True:                        
                        reward_step = self.reward_stepwise
                        X_new, A_new = self.read_from_graph_raw(self.graph)
                        X_new = torch.from_numpy(X_new).to(dtype=torch.float32)
                        A_new = torch.from_numpy(A_new)

                        edge_index = A_new.nonzero().t().contiguous()
                        labels = torch.tensor([0] * X_new.shape[0])
                        data = Data(x=X_new, edge_index=edge_index, y=labels)
                        data.train_mask = torch.tensor(labels==self.target_class, dtype=torch.bool)

                        logits = self.gnnNets(data.x, data.edge_index)
                        #### based on logits, define the reward
                        prediction = logits.argmax(dim=-1)
                        correct = (prediction[data.train_mask] == data.y[data.train_mask])
                        correct_indices = [i for i, x in enumerate(correct.tolist()) if x]

                        if correct_indices:
                            probs = logits.softmax(dim=-1)[correct_indices]
                            reward_pred = probs.ravel()[self.target_class]- 0.01 # positive reward
                        else:
                            probs = logits.softmax(dim=-1)[0]
                            reward_pred = probs[self.target_class] - 0.01 #negative reward


                        ### Then we need to roll out.
                        reward_rollout= []
                        for roll in range(10):
                            reward_cur = self.roll_out(self.graph, j)
                            reward_rollout.append(reward_cur)
                        reward_avg = torch.mean(torch.stack(reward_rollout))
                            ###desgin loss
                        total_reward = reward_step+reward_pred+reward_avg*self.roll_out_alpha  ## need to tune the hyper-parameters here. 
                        
                        if total_reward < 0:
                            self.graph = copy.deepcopy(self.graph_old) ### rollback

                        loss = total_reward*(self.criterion(start_logits_ori[None,:], start_action.expand(1)) 
                                + self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))
                    else:
                        total_reward = -1  # graph is not valid 
                        self.graph = copy.deepcopy(self.graph_old)
                        loss = total_reward*(self.criterion(start_logits_ori[None,:], start_action.expand(1)) 
                                + self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))                        
                else:
                    # ### case adding edge not successful
                    reward_step = -1
                    total_reward= reward_step+reward_pred

                    loss = total_reward*(self.criterion(start_logits_ori[None,:], start_action.expand(1)) + 
                            self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))               

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policyNets.parameters(), 100)
                self.optimizer.step()
        print(self.graph.nodes(data=True))
        #self.graph_draw(self.graph)
        #plt.show()
        X_new, A_new = self.read_from_graph_raw(self.graph)
        X_new = torch.from_numpy(X_new).to(dtype=torch.float32)
        A_new = torch.from_numpy(A_new)

        edge_index = A_new.nonzero().t().contiguous()
        labels = torch.argmax(X_new, dim=1)
        data = Data(x=X_new, edge_index=edge_index, y=labels)
        data.train_mask = torch.tensor(torch.ones(len(labels)), dtype=torch.bool)

        logits = self.gnnNets(data.x, data.edge_index)
        probs =  logits.softmax(dim=-1)[0]
        prob = probs[0].item()

        print(prob)
        return A_new, prob


    def graph_draw(self, graph):
        attr = nx.get_node_attributes(graph, "label")
        labels = {}
        color = ''
        for n in attr:
            labels[n]= self.dict[attr[n]]
            color = color+ self.color[attr[n]]

        nx.draw(graph,labels=labels, node_color=color)
        
        
    def check_validity(self, graph):
        node_types = nx.get_node_attributes(graph,'label')
        for i in range(graph.number_of_nodes()):
            degree = graph.degree(i)
            max_allow = self.max_poss_degree[node_types[i]]
            if(degree> max_allow):
                return False
        return True
    
    def roll_out(self, graph, j):
        cur_graph = copy.deepcopy(graph)
        step = 0
        while(cur_graph.number_of_nodes()<=self.max_node and step<self.max_step-j):
            graph_old = copy.deepcopy(cur_graph)
            step = step + 1
            X, A = self.read_from_graph(cur_graph)
            n = cur_graph.number_of_nodes()
            X = torch.from_numpy(X)
            A = torch.from_numpy(A)
            start_action, start_logits_ori, tail_action, tail_logits_ori  = self.policyNets(X.float(), A.float(), n+self.node_type)
            if(tail_action>=n): ####we need add node, then add edge
                if(n==self.max_node):
                    flag = False
                else:
                    self.add_node(cur_graph, n, tail_action.item()-n)
                    flag = self.add_edge(cur_graph, start_action.item(), n)
            else:
                flag= self.add_edge(cur_graph, start_action.item(), tail_action.item())
                    
            ## if the graph is not valid in rollout, two possible solutions
            ## 1. return a negative reward as overall reward for this rollout  --- what we do here. 
            ## 2. compute the loss but do not update model parameters here--- update with the step loss togehter. 
            if flag == True:
                validity = self.check_validity(cur_graph)
                if validity == False:
                    return torch.tensor(self.roll_out_penalty)

            else:  ### case 1: add edges but already exists, case2: keep add node when reach max_node
                return torch.tensor(self.roll_out_penalty)
                
        ###Then we evaluate the final graph
        X_new, A_new = self.read_from_graph_raw(cur_graph)

        X_new = torch.from_numpy(X_new).to(dtype=torch.float32)
        A_new = torch.from_numpy(A_new)

        edge_index = A_new.nonzero().t().contiguous()
        labels = torch.argmax(X_new, dim=1)
        data = Data(x=X_new, edge_index=edge_index, y=labels)
        data.train_mask = torch.tensor(torch.ones(len(labels)), dtype=torch.bool)

        logits = self.gnnNets(data.x, data.edge_index)
        ### Todo
        probs = logits.softmax(dim=-1)[0]
        reward = probs[self.target_class] - 0.01
        return reward
        

    def add_node(self, graph, idx, node_type):
        graph.add_node(idx, label=node_type)
        return 
    
    def add_edge(self, graph, start_id, tail_id):
        if graph.has_edge(start_id, tail_id) or start_id==tail_id:
            return False
        else:
            graph.add_edge(start_id, tail_id)
            return True
    
    def read_from_graph(self, graph): ## read graph with added  candidates nodes
        n = graph.number_of_nodes()

        F = np.zeros((self.max_node+self.node_type, self.node_type))
        attr = nx.get_node_attributes(graph, "label")
        attr = list(attr.values())
        nb_clss  = self.node_type
        targets=np.array(attr).reshape(-1)
        one_hot_feature = np.eye(nb_clss)[targets]
        F[:n,:]= one_hot_feature
    ### then get the onehot features for the candidates nodes
        F[n:n+self.node_type,:]= np.eye(self.node_type)      
        
        E = np.zeros([self.max_node+self.node_type, self.max_node+self.node_type])
        E[:n,:n] = np.asarray(nx.to_numpy_array(graph))
        E[:self.max_node+self.node_type,:self.max_node+self.node_type] += np.eye(self.max_node+self.node_type)
        return F, E


    def read_from_graph_raw(self, graph): ### do not add more nodes
        n = graph.number_of_nodes()

        attr = nx.get_node_attributes(graph, "label")
        attr = list(attr.values())
        nb_clss  = self.node_type
        targets=np.array(attr).reshape(-1)
        one_hot_feature = np.eye(nb_clss)[targets]

        E = np.zeros([n, n])
        E[:n,:n] = np.asarray(nx.to_numpy_array(graph))

        return one_hot_feature, E

    def graph_reset(self):
        self.graph.clear()
        self.graph.add_node(0, label=self.target_class)
        self.step = 0
        return 
    
                       
              
       

                

                

                

