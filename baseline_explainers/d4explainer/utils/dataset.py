import os

from datasets import  BA3Motif, Mutag, Syn_NC_Dataset
from datasets.real_nc_dataset import Real_NC_Dataset


def get_datasets(name, root="data/"):
    """
    Get preloaded datasets by name
    :param name: name of the dataset
    :param root: root path of the dataset
    :return: train_dataset, test_dataset, val_dataset
    """
    if name == "mutag":
        folder = os.path.join(root, "MUTAG")
        train_dataset = Mutag(folder, mode="training")
        test_dataset = Mutag(folder, mode="testing")
        val_dataset = Mutag(folder, mode="evaluation")
    elif name == "ba3":
        folder = os.path.join(root, "BA3")
        train_dataset = BA3Motif(folder, mode="training")
        test_dataset = BA3Motif(folder, mode="testing")
        val_dataset = BA3Motif(folder, mode="evaluation")
    elif name == "BA_shapes":
        folder = os.path.join(root)
        test_dataset = Syn_NC_Dataset(folder, mode="testing", name="BA_shapes")
        val_dataset = Syn_NC_Dataset(folder, mode="evaluating", name="BA_shapes")
        train_dataset = Syn_NC_Dataset(folder, mode="training", name="BA_shapes")
    elif name == "dblp":
        folder = os.path.join(root)
        test_dataset = Real_NC_Dataset(folder, mode="testing", name="dblp")
        val_dataset = Real_NC_Dataset(folder, mode="evaluating", name="dblp")
        train_dataset = Real_NC_Dataset(folder, mode="training", name="dblp")
    elif name == "imdb":
        folder = os.path.join(root)
        test_dataset = Real_NC_Dataset(folder, mode="testing", name="imdb")
        val_dataset = Real_NC_Dataset(folder, mode="evaluating", name="imdb")
        train_dataset = Real_NC_Dataset(folder, mode="training", name="imdb")
    elif name == "Tree_Cycle":
        folder = os.path.join(root)
        test_dataset = Syn_NC_Dataset(folder, mode="testing", name="Tree_Cycle")
        val_dataset = Syn_NC_Dataset(folder, mode="evaluating", name="Tree_Cycle")
        train_dataset = Syn_NC_Dataset(folder, mode="training", name="Tree_Cycle")
    elif name == "Tree_Grids":
        folder = os.path.join(root)
        test_dataset = Syn_NC_Dataset(folder, mode="testing", name="Tree_Grids")
        val_dataset = Syn_NC_Dataset(folder, mode="evaluating", name="Tree_Grids")
        train_dataset = Syn_NC_Dataset(folder, mode="training", name="Tree_Grids")


    else:
        raise ValueError
    return train_dataset, val_dataset, test_dataset
