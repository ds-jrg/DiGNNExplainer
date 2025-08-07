import os

import numpy as np
import torch
from torch.cuda import device
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import networkx as nx

from explainers.base import Explainer
from explainers.diffusion.graph_utils import (
    gen_full,
    gen_list_of_data_single,
    generate_mask,
    graph2tensor,
    tensor2graph,
)
from explainers.diffusion.pgnn import Powerful
import random




def model_save(args, model, mean_train_loss, best_sparsity):
    """
    Save the model to disk
    :param args: arguments
    :param model: model
    :param mean_train_loss: mean training loss
    :param best_sparsity: best sparsity
    :param mean_test_acc: mean test accuracy
    """
    to_save = {
        "model": model.state_dict(),
        "train_loss": mean_train_loss,
        "eval sparsity": best_sparsity,
        # "eval acc": mean_test_acc,
    }
    exp_dir = f"{args.root}/{args.dataset}/"
    os.makedirs(exp_dir, exist_ok=True)
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))
    print(f"save model to {exp_dir}/best_model.pth")


def loss_func_bce(score_list, groundtruth, sigma_list, mask, device, sparsity_level):
    """
    Loss function for binary cross entropy
    param score_list: [len(sigma_list)*bsz, N, N]
    param groundtruth: [len(sigma_list)*bsz, N, N]
    param sigma_list: list of sigma values
    param mask: [len(sigma_list)*bsz, N, N]
    param device: device
    param sparsity_level: sparsity level
    return: BCE loss
    """
    bsz = int(score_list.size(0) / len(sigma_list))
    num_node = score_list.size(-1)
    score_list = score_list * mask
    groundtruth = groundtruth * mask
    pos_weight = torch.full([num_node * num_node], sparsity_level).to(device)
    BCE = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    score_list_ = torch.flatten(score_list, start_dim=1, end_dim=-1)
    groundtruth_ = torch.flatten(groundtruth, start_dim=1, end_dim=-1)
    loss_matrix = BCE(score_list_, groundtruth_)
    loss_matrix = loss_matrix.view(groundtruth.size(0), num_node, num_node)
    loss_matrix = loss_matrix * (
        1
        - 2
        * torch.tensor(sigma_list)
        .repeat(bsz)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(groundtruth.size(0), num_node, num_node)
        .to(device)
        + 1.0 / len(sigma_list)
    )
    loss_matrix = loss_matrix * mask
    loss_matrix = (loss_matrix + torch.transpose(loss_matrix, -2, -1)) / 2
    loss = torch.mean(loss_matrix)
    return loss


def sparsity(score, groundtruth, mask, threshold=0.5):
    """
    Calculate the sparsity of the predicted adjacency matrix
    :param score: [bsz, N, N, 1]
    :param groundtruth: [bsz, N, N]
    :param mask: [bsz, N, N]
    :param threshold: threshold for the predicted adjacency matrix
    :return: sparsity
    """
    score_tensor = torch.stack(score, dim=0).squeeze(-1)  # [len_sigma_list, bsz, N, N]
    score_tensor = torch.mean(score_tensor, dim=0)  # [bsz, N, N]
    pred_adj = torch.where(torch.sigmoid(score_tensor) > threshold, 1, 0).to(groundtruth.device)
    pred_adj = pred_adj * mask
    groundtruth_ = groundtruth * mask
    adj_diff = torch.abs(groundtruth_ - pred_adj)  # [bsz, N, N]
    num_edge_b = groundtruth_.sum(dim=(1, 2))
    adj_diff_ratio = adj_diff.sum(dim=(1, 2)) / num_edge_b
    ratio_average = torch.mean(adj_diff_ratio)
    return ratio_average


def gnn_pred(graph_batch, graph_batch_sub, gnn_model, ds, task):
    """
    Predict the labels of the graph
    :param graph_batch: graph batch
    :param graph_batch_sub: subgraph batch
    :param gnn_model: GNN model
    :param ds: dataset
    :param task: task
    :return: predicted labels (full graph and subgraph)
    """
    gnn_model.eval()
    if task == "nc":
        output_prob, _ = gnn_model.get_node_pred_subgraph(
            x=graph_batch.x,
            edge_index=graph_batch.edge_index,
            mapping=graph_batch.mapping,
        )
        output_prob_sub, _ = gnn_model.get_node_pred_subgraph(
            x=graph_batch_sub.x,
            edge_index=graph_batch_sub.edge_index,
            mapping=graph_batch_sub.mapping,
        )
    else:
        output_prob, _ = gnn_model.get_pred(
                x=graph_batch.x,
                edge_index=graph_batch.edge_index,
                batch=graph_batch.batch,
            )
        output_prob_sub, _ = gnn_model.get_pred(
                x=graph_batch_sub.x,
                edge_index=graph_batch_sub.edge_index,
                batch=graph_batch_sub.batch,
            )

    y_pred = output_prob.argmax(dim=-1)
    y_exp = output_prob_sub.argmax(dim=-1)
    return y_pred, y_exp


def loss_cf_exp(gnn_model, graph_batch, score, y_pred, y_exp, full_edge, mask, ds, task="nc"):
    """
    Loss function for counterfactual explanation
    :param gnn_model: GNN model
    :param graph_batch: graph batch
    :param score: list of scores
    :param y_pred: predicted labels
    :param y_exp: predicted labels for subgraph
    :param full_edge: full edge index
    :param mask: mask
    :param ds: dataset
    :param task: task
    :return: loss
    """
    score_tensor = torch.stack(score, dim=0).squeeze(-1)
    score_tensor = torch.mean(score_tensor, dim=0).view(-1, 1)
    mask_bool = mask.bool().view(-1, 1)
    edge_mask_full = score_tensor[mask_bool]
    assert edge_mask_full.size(0) == full_edge.size(1)
    criterion = torch.nn.NLLLoss()
    if task == "nc":
        output_prob_cont, output_repr_cont = gnn_model.get_pred_explain(
            x=graph_batch.x,
            edge_index=full_edge,
            edge_mask=edge_mask_full,
            mapping=graph_batch.mapping,
        )
    else:
        output_prob_cont, output_repr_cont = gnn_model.get_pred_explain(
            x=graph_batch.x,
            edge_index=full_edge,
            edge_mask=edge_mask_full,
            batch=graph_batch.batch,
        )
    n = output_repr_cont.size(-1)
    bsz = output_repr_cont.size(0)
    y_exp = output_prob_cont.argmax(dim=-1)
    inf_diag = torch.diag(-torch.ones((n)) / 0).unsqueeze(0).repeat(bsz, 1, 1).to(y_pred.device)
    neg_prop = (output_repr_cont.unsqueeze(1).expand(bsz, n, n) + inf_diag).logsumexp(-1)
    neg_prop = neg_prop - output_repr_cont.logsumexp(-1).unsqueeze(1).repeat(1, n)
    loss_cf = criterion(neg_prop, y_pred)
    labels = torch.LongTensor([[i] for i in y_pred]).to(y_pred.device)
    fid_drop = (1 - output_prob_cont.gather(1, labels).view(-1)).detach().cpu().numpy()
    fid_drop = np.mean(fid_drop)
    acc_cf = float(y_exp.eq(y_pred).sum().item() / y_pred.size(0))  # less, better
    return loss_cf, fid_drop, acc_cf


class DiffExplainer(Explainer):
    def __init__(self, device, gnn_model_path):
        super(DiffExplainer, self).__init__(device, gnn_model_path)

    def explain_graph_task(self, args, train_dataset, test_dataset):
        """
        Explain the graph for a specific dataset and task
        :param args: arguments
        :param train_dataset: training dataset
        :param test_dataset: test dataset
        """
        gnn_model = self.model.to(args.device)
        model = Powerful(args).to(args.device)
        self.train(args, model, gnn_model, train_dataset, test_dataset)

    def train(self, args, model, gnn_model, train_dataset, test_dataset):
        """
        Train the model
        :param args: arguments
        :param model: Powerful (explanation) model
        :param gnn_model: GNN model
        :param train_dataset: training dataset
        :param test_dataset: test dataset
        """
        best_sparsity = np.inf
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
        noise_list = args.noise_list
        for epoch in range(args.epoch):
            print(f"start epoch {epoch}")
            train_losses = []

            train_sparsity = []
            model.train()
            train_loader = DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True)
            for i, graph in enumerate(train_loader):
                if graph.is_directed():
                    edge_index_temp = graph.edge_index
                    graph.edge_index = to_undirected(edge_index=edge_index_temp)
                graph.to(args.device)
                train_adj_b, train_x_b = graph2tensor(graph, device=args.device)
                # train_adj_b: [bsz, N, N]; train_x_b: [bsz, N, C]
                sigma_list = (
                    list(np.random.uniform(low=args.prob_low, high=args.prob_high, size=args.sigma_length))
                    if noise_list is None
                    else noise_list
                )
                train_node_flag_b = train_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)  # [bsz, N]
                # all nodes that are not connected with others
                if isinstance(sigma_list, float):
                    sigma_list = [sigma_list]
                (train_x_b, train_ori_adj_b, train_node_flag_sigma, train_noise_adj_b, _) = gen_list_of_data_single(
                    train_x_b, train_adj_b, train_node_flag_b, sigma_list, args
                )
                optimizer.zero_grad()
                train_noise_adj_b_chunked = train_noise_adj_b.chunk(len(sigma_list), dim=0)
                train_x_b_chunked = train_x_b.chunk(len(sigma_list), dim=0)
                train_node_flag_sigma = train_node_flag_sigma.chunk(len(sigma_list), dim=0)
                score = []
                masks = []
                for i, sigma in enumerate(sigma_list):
                    mask = generate_mask(train_node_flag_sigma[i])
                    score_batch = model(
                        A=train_noise_adj_b_chunked[i].to(args.device),
                        node_features=train_x_b_chunked[i].to(args.device),
                        mask=mask.to(args.device),
                        noiselevel=sigma,
                    )  # [bsz, N, N, 1]
                    score.append(score_batch)
                    masks.append(mask)

                score_b = torch.cat(score, dim=0).squeeze(-1).to(args.device)  # [len(sigma_list)*bsz, N, N]
                masktens = torch.cat(masks, dim=0).to(args.device)  # [len(sigma_list)*bsz, N, N]
                modif_r = sparsity(score, train_adj_b, mask)

                loss_dist = loss_func_bce(
                    score_b,
                    train_ori_adj_b,
                    sigma_list,
                    masktens,
                    device=args.device,
                    sparsity_level=args.sparsity_level,
                )

                loss = loss_dist
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                train_sparsity.append(modif_r.item())

            scheduler.step(epoch)
            mean_train_loss = np.mean(train_losses)
            mean_train_sparsity = np.mean(train_sparsity)
            print(
                (
                    f"Training Epoch: {epoch} | "
                    f"training loss: {mean_train_loss} | "
                    f"training average modification: {mean_train_sparsity} | "
                )
            )
            # evaluation
            if (epoch + 1) % args.verbose == 0:
                test_losses = []
                test_sparsity = []
                test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batchsize, shuffle=False)
                model.eval()
                for graph in test_loader:
                    if graph.is_directed():
                        edge_index_temp = graph.edge_index
                        graph.edge_index = to_undirected(edge_index=edge_index_temp)

                    graph.to(args.device)
                    test_adj_b, test_x_b = graph2tensor(graph, device=args.device)
                    test_x_b = test_x_b.to(args.device)
                    test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
                    sigma_list = (
                        list(np.random.uniform(low=args.prob_low, high=args.prob_high, size=args.sigma_length))
                        if noise_list is None
                        else noise_list
                    )
                    if isinstance(sigma_list, float):
                        sigma_list = [sigma_list]
                    (test_x_b, test_ori_adj_b, test_node_flag_sigma, test_noise_adj_b, _) = gen_list_of_data_single(
                        test_x_b, test_adj_b, test_node_flag_b, sigma_list, args
                    )
                    with torch.no_grad():
                        test_noise_adj_b_chunked = test_noise_adj_b.chunk(len(sigma_list), dim=0)
                        test_x_b_chunked = test_x_b.chunk(len(sigma_list), dim=0)
                        test_node_flag_sigma = test_node_flag_sigma.chunk(len(sigma_list), dim=0)
                        score = []
                        masks = []
                        for i, sigma in enumerate(sigma_list):
                            mask = generate_mask(test_node_flag_sigma[i])
                            score_batch = model(
                                A=test_noise_adj_b_chunked[i].to(args.device),
                                node_features=test_x_b_chunked[i].to(args.device),
                                mask=mask.to(args.device),
                                noiselevel=sigma,
                            ).to(args.device)
                            masks.append(mask)
                            score.append(score_batch)

                        score_b = torch.cat(score, dim=0).squeeze(-1).to(args.device)
                        masktens = torch.cat(masks, dim=0).to(args.device)
                        modif_r = sparsity(score, test_adj_b, mask)

                        loss_dist = loss_func_bce(
                            score_b,
                            test_ori_adj_b,
                            sigma_list,
                            masktens,
                            device=args.device,
                            sparsity_level=args.sparsity_level,
                        )

                        loss = loss_dist
                        test_losses.append(loss.item())

                        test_sparsity.append(modif_r.item())
                mean_test_loss = np.mean(test_losses)
                mean_test_sparsity = np.mean(test_sparsity)
                print(
                    (
                        f"Evaluation Epoch: {epoch} | "
                        f"test loss: {mean_test_loss} | "
                        f"test average modification: {mean_test_sparsity} | "
                    )
                )
                if mean_test_sparsity < best_sparsity:
                    best_sparsity = mean_test_sparsity
                    model_save(args, model, mean_train_loss, best_sparsity)

    def explain_evaluation(self, args, graph):

        model = Powerful(args).to(args.device)
        exp_dir = f"{args.root}/{args.dataset}/"
        model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pth"))["model"])

        model.eval()
        graph.to(args.device)
        test_adj_b, test_x_b = graph2tensor(graph, device=args.device)  # [bsz, N, N]
        test_x_b = test_x_b.to(args.device)
        test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
        sigma_list = (
            list(np.random.uniform(low=args.prob_low, high=args.prob_high, size=args.sigma_length))
            if args.noise_list is None
            else args.noise_list
        )
        if isinstance(sigma_list, float):
            sigma_list = [sigma_list]
        (test_x_b, _, test_node_flag_sigma, test_noise_adj_b, _) = gen_list_of_data_single(
            test_x_b, test_adj_b, test_node_flag_b, sigma_list, args
        )
        test_noise_adj_b_chunked = test_noise_adj_b.chunk(len(sigma_list), dim=0)
        test_x_b_chunked = test_x_b.chunk(len(sigma_list), dim=0)
        test_node_flag_sigma = test_node_flag_sigma.chunk(len(sigma_list), dim=0)
        score = []
        masks = []
        for i, sigma in enumerate(sigma_list):
            mask = generate_mask(test_node_flag_sigma[i])
            score_batch = model(
                A=test_noise_adj_b_chunked[i].to(args.device),
                node_features=test_x_b_chunked[i].to(args.device),
                mask=mask.to(args.device),
                noiselevel=sigma,
            ).to(args.device)
            masks.append(mask)
            score.append(score_batch)
        graph_batch_sub = tensor2graph(graph, score, mask)
        full_edge_index = gen_full(graph.batch, mask)
        modif_r = sparsity(score, test_adj_b, mask)
        score_tensor = torch.stack(score, dim=0).squeeze(-1)  # len_sigma_list, bsz, N, N]
        score_tensor = torch.mean(score_tensor, dim=0).view(-1, 1)  # [bsz*N*N,1]
        mask_bool = mask.bool().view(-1, 1)
        edge_mask_full = score_tensor[mask_bool]
        if args.task == "nc":
            output_prob_cont, _ = self.model.get_pred_explain(
                x=graph.x, edge_index=full_edge_index, edge_mask=edge_mask_full, mapping=graph.mapping
            )
        else:
            output_prob_cont, _ = self.model.get_pred_explain(
                x=graph.x, edge_index=full_edge_index, edge_mask=edge_mask_full, batch=graph.batch
            )
        y_ori = graph.y if args.task == "gc" else graph.self_y
        y_exp = output_prob_cont.argmax(dim=-1)
        return graph_batch_sub, y_ori, y_exp, modif_r


    #One-step Model level explanation using the trained model
    def one_step_model_level(self, args, adj, node_feature, target_class,node_size):

        adj = adj.unsqueeze(0).to(args.device)  # batchsize=1
        node_feature = node_feature.unsqueeze(0).to(args.device)  # batchsize=1
        mask = torch.ones_like(adj).to(args.device)

        model = Powerful(args).to(args.device)
        exp_dir = f"{args.root}/{args.dataset}/"
        model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pth"))["model"])
        model.eval()

        sigma_list = (
            list(np.random.uniform(low=args.prob_low, high=args.prob_high, size=args.sigma_length))
            if args.noise_list is None
            else args.noise_list
        )

        softmax_dict = {}
        best_softmax = torch.tensor(0)
        best_adj = torch.tensor([])
        #Sample candidates
        for i, sigma in enumerate(sigma_list):
            score = model(A=adj.float(), node_features=node_feature.float(), mask=mask.float(), noiselevel=sigma)
            score = score.squeeze(0).squeeze(-1)

            pred_adj = torch.where(torch.sigmoid(score) > 0.5, 1, 0).to(score.device)
            edge_index = pred_adj.nonzero().t().contiguous()
            if args.task == "nc":
                #pass
                output_prob, _ = self.model.get_node_pred_subgraph(
                    x=node_feature.float().squeeze(0).to(args.device),
                    edge_index=edge_index.to(args.device),
                    mapping=torch.ones(node_size, dtype=torch.bool), #6
                )
                y_pred = output_prob.argmax(dim=-1)
                correct = (y_pred == torch.full((1,node_size), target_class).to(args.device)) #6
                correct_indices = [i for i, x in enumerate(correct.tolist()) if x]

                softmax_pred_list = [output_prob.tolist()[i] for i in correct_indices]
                softmax_dict[pred_adj] = softmax_pred_list
            else:
                output_prob, _ = self.model.get_pred(
                    x=node_feature.float().squeeze(0).to(args.device) , edge_index=edge_index.to(args.device) , batch=torch.zeros(node_feature.shape[0]).long().to(args.device)
                )

                y_pred = output_prob.argmax(dim=-1)

                if y_pred == target_class:

                    temp = output_prob[0][target_class]

                    # select graph with highest probability
                    if best_softmax < temp:
                        best_softmax = temp
                        best_adj = pred_adj


        if args.task == "nc":
            prob_class_dict = {}

            for nodeid in softmax_dict:
                list_class = []

                if len(softmax_dict[nodeid]) > 0:
                    list_class = []

                    for prob in softmax_dict[nodeid]:
                        list_class.append(prob[target_class])

                # Taking max probability of all nodes of each class in a graph
                if len(list_class) != 0:
                    prob_class_dict[nodeid] = max(list_class)

            best_softmax = max(prob_class_dict.values())
            best_adj = max(prob_class_dict, key=prob_class_dict.get)

        else:
            if len(best_adj)==0:

                best_adj = adj.squeeze(0)

        #temporary explanation
        return best_softmax, best_adj  # [N, N]

    # https://networkx.org/documentation/stable/auto_examples/graph/plot_erdos_renyi.html
    def multi_step_model_level(self, args, target_class, classes, node_size):
        n = node_size   # nodes
        m = n / 2  # edges
        seed = 20160  # seed random number generators for reproducibility

        #Sample an Erdős–Rényi graph
        G = nx.gnm_random_graph(n, m, seed=seed)
        #Get node features
        if args.dataset in ['dblp', 'imdb', 'mutag']:
            node_types = [random.choice(classes) for _ in range(n)]
            targets = np.array(node_types).reshape(-1)
            node_feature = torch.tensor(np.eye(len(classes))[targets])

        elif args.dataset in ['BA_shapes','Tree_Cycle','Tree_Grids','ba3']:
            node_feature = torch.tensor(np.eye(len(classes))[0]).repeat(n,1)

        adj = torch.tensor(nx.adjacency_matrix(G).todense())

        steps = 50
        for i in range(steps, 1, -1):

            softmax, pred_adj = self.one_step_model_level(args,adj, node_feature, target_class, node_size)
            adj = pred_adj

        return adj, softmax


