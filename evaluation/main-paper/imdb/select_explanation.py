import torch


def evaluate_gnn(small_graph, model):
    with torch.no_grad():
        model.eval()
        
        #Getting class prediction probabilities from the softmax layer
        softmax = model(small_graph.x_dict, small_graph.edge_index_dict).softmax(dim=-1)

        return softmax.tolist()
        
        
        
def get_max_pred(softmax_dict,i):
    #Getting the list of predictions for each class
    prob_class0_dict = {}
    prob_class1_dict = {}
    prob_class2_dict = {}
    
    for nodeid in softmax_dict:
        list0= []
        list1= []
        list2= []
        
        if len(softmax_dict[nodeid]) > 0:
            list0= []
            list1= []
            list2= []
            
    
            for prob in softmax_dict[nodeid]:        
                list0.append(prob[0])        
                list1.append(prob[1]) 
                list2.append(prob[2]) 
                
    
         #Taking max probability of all nodes of each class in a graph
        if len(list0) != 0:
            prob_class0_dict[nodeid]=max(list0)
        if len(list1) != 0:    
            prob_class1_dict[nodeid]=max(list1)
        if len(list2) != 0:    
            prob_class2_dict[nodeid]=max(list2)
        

    max_pred0 = max(prob_class0_dict.values())
    max_pred1 = max(prob_class1_dict.values())
    max_pred2 = max(prob_class2_dict.values())
    

    print('Run'+str(i), max_pred0, max_pred1, max_pred2)
 
    max_pred = [max_pred0, max_pred1, max_pred2]
  
    avg_max_pred = (max_pred0+max_pred1+max_pred2)/3
    
    class0_graphid = max(prob_class0_dict, key=prob_class0_dict.get)
    class1_graphid = max(prob_class1_dict, key=prob_class1_dict.get)
    class2_graphid = max(prob_class2_dict, key=prob_class2_dict.get)
    

    class_graphid = [class0_graphid,class1_graphid,class2_graphid]

    return avg_max_pred, max_pred, class_graphid
