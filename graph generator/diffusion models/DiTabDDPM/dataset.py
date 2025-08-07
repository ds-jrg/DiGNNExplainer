import os
import pathlib
import torch
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


import torch_geometric.transforms as T
from torch_geometric.datasets.dblp import DBLP
from torch_geometric.datasets import IMDB
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset
from sklearn.feature_selection import VarianceThreshold
from abstract_dataset import AbstractDataModule, AbstractDatasetInfos
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import degree


class FeatureDataset(InMemoryDataset):
    def __init__(self, dataset, split, root,transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset.name
        self.node_class = dataset.node_class
        self.node_type = dataset.node_type
        self.node_feature_size = dataset.node_feature_size
        self.node_feature_types = dataset.node_feature_types
        self.feature_selection_method = dataset.feature_selection_method
        self.threshold = dataset.threshold

        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def get_selected_features(self, X):
        col_sum = X.sum(axis=0)
        sorted_colsum = sorted(col_sum, reverse=True)[:self.node_feature_size]

        colsum_df = pd.DataFrame(col_sum)
        index_list = list(np.ravel(colsum_df[colsum_df[0].isin(sorted_colsum)].index))

        imp_feat = X[index_list]
        return imp_feat.iloc[:, : self.node_feature_size]

    def feature_selection_var(self, X, threshold=0.0):
        sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
        fitted_X = sel.fit_transform(X)
        imp_feat = pd.DataFrame(fitted_X)

        return imp_feat

    def get_one_hot_degree(self, x, edge_index):
        node_data = Data(x=x, edge_index=edge_index, num_nodes=x.shape[0])
        max_degree = max(degree(edge_index[0], x.shape[0]))
        transform = T.OneHotDegree(max_degree=max_degree.to(torch.int32).numpy(), cat=False)
        return transform(node_data)

    def set_feat_col_names(self,df):
        columnnames = {}
        count = 0
        for i in df.columns:
            count += 1
            columnnames[i] = f"feat{count}"
            df.rename(columns=columnnames, inplace=True)
        return df

    def get_node_classs_feat(self,df):
        return df[df['class'] == self.node_class].drop(['class'], axis=1)

    def concat_feat_degree(self,df_feat,df_degree):
        return pd.concat([df_feat, df_degree], axis=1)

    def download(self):
        # """
        # Download dataset
        # """

        train_data = []
        val_data = []
        test_data = []
        node_feat = pd.DataFrame()

        if self.dataset_name == 'dblp':
            print('Dataset name',self.dataset_name)
            dataset = DBLP(root='./dblp_data', transform=T.Constant(node_types='conference'))
            data = dataset[0]

            if self.node_type == 'author':
                # Author

                # Get node degree
                author_nodes_deg = self.get_one_hot_degree(data['author'].x,data['author','to','paper'].edge_index)

                # Get class for original features
                author = data['author'].x.tolist()
                author_df = pd.DataFrame(author)
                author_df['class'] = data['author'].y.tolist()
                author_class = self.get_node_classs_feat(author_df)

                # Get node class for one-hot encoded degree
                author_df_deg = pd.DataFrame(author_nodes_deg.x)
                author_df_deg['class'] = data['author'].y.tolist()
                author_class_deg_df = self.get_node_classs_feat(author_df_deg)

                # Feature selection for Author class
                if self.feature_selection_method == 'frequency':
                    sel_feat  = self.get_selected_features(author_class)
                    sel_feat_df = self.set_feat_col_names(sel_feat)
                    node_feat =self.concat_feat_degree(sel_feat_df.reset_index(drop=True),author_class_deg_df.reset_index(drop=True))

                elif self.feature_selection_method == 'variance':
                    sel_feat = self.feature_selection_var(author_class, threshold=self.threshold).iloc[:, : self.node_feature_size]
                    sel_feat_df = self.set_feat_col_names(sel_feat)
                    node_feat =self.concat_feat_degree(sel_feat_df.reset_index(drop=True),author_class_deg_df.reset_index(drop=True))

            elif self.node_type == 'paper':
                # Paper
                paper_to_term_edge_index = data['paper', 'to', 'term'].edge_index
                paper_to_conf_edge_index = data['paper', 'to', 'conference'].edge_index
                paper_edge_index = torch.cat([paper_to_term_edge_index, paper_to_conf_edge_index], dim=1)
                paper_nodes_deg = self.get_one_hot_degree(data['paper'].x, paper_edge_index)
                paper_df_deg = pd.DataFrame(paper_nodes_deg.x)

                paper = data['paper'].x.tolist()
                df_paper = pd.DataFrame(paper)
                if self.feature_selection_method == 'frequency':
                    sel_feat = self.get_selected_features(df_paper)
                    sel_feat_df = self.set_feat_col_names(sel_feat)
                    node_feat = self.concat_feat_degree(sel_feat_df.reset_index(drop=True), paper_df_deg.reset_index(drop=True))

                elif self.feature_selection_method == 'variance':
                    sel_feat = self.feature_selection_var(df_paper, threshold = self.threshold).iloc[:, : self.node_feature_size]
                    sel_feat_df = self.set_feat_col_names(sel_feat)
                    node_feat = self.concat_feat_degree(sel_feat_df.reset_index(drop=True), paper_df_deg.reset_index(drop=True))

        elif self.dataset_name == 'imdb':
            dataset = IMDB(root='./imdb_data')
            data = dataset[0]
            # Get node degree
            movie_to_director_edge_index = data['movie', 'to', 'director'].edge_index
            movie_to_actor_edge_index = data['movie', 'to', 'actor'].edge_index
            movie_edge_index = torch.cat([movie_to_director_edge_index, movie_to_actor_edge_index], dim=1)
            movie_nodes_deg = self.get_one_hot_degree(data['movie'].x, movie_edge_index)

            # Get class for original features
            movie = data['movie'].x.tolist()
            movie_df = pd.DataFrame(movie)
            movie_df['class'] = data['movie'].y.tolist()
            movie_class = self.get_node_classs_feat(movie_df)

            # Get node class for one-hot encoded degree
            movie_df_deg = pd.DataFrame(movie_nodes_deg.x)
            movie_df_deg['class'] = data['movie'].y.tolist()
            movie_class_deg_df = self.get_node_classs_feat(movie_df_deg)

            if self.feature_selection_method == '':
                node_feat = movie_class
            # Feature selection for Movie class
            elif self.feature_selection_method == 'frequency':
                sel_feat = self.get_selected_features(movie_class)
                sel_feat_df = self.set_feat_col_names(sel_feat)
                node_feat = self.concat_feat_degree(sel_feat_df.reset_index(drop=True), movie_class_deg_df.reset_index(drop=True))
            elif self.feature_selection_method == 'variance':
                sel_feat = self.feature_selection_var(movie_class, threshold=self.threshold).iloc[:,: self.node_feature_size]
                sel_feat_df = self.set_feat_col_names(sel_feat)
                node_feat = self.concat_feat_degree(sel_feat_df.reset_index(drop=True), movie_class_deg_df.reset_index(drop=True))


        # Using train/test/val -80/10/10
        # https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
        train, val, test = np.split(node_feat.sample(frac=1, random_state=42),
                                        [int(.8 * len(node_feat)), int(.9 * len(node_feat))])


        train_data.append(torch.tensor(train.values))
        val_data.append(torch.tensor(val.values))
        test_data.append(torch.tensor(test.values))

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])



    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])
        feature_types = torch.tensor(self.node_feature_types)
        data_list = []
        for d in raw_dataset:

            node_feature = torch.tensor(d)

            X = torch.tensor([F.one_hot(x.long(), num_classes=len(feature_types)).float().tolist() for i, x in enumerate(node_feature)])
            data = torch_geometric.data.Data(feature=X)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class FeatureDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[0]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': FeatureDataset(dataset=self.cfg.dataset,
                                                 split='train', root=root_path),
                    'val': FeatureDataset(dataset=self.cfg.dataset,
                                        split='val', root=root_path),
                    'test': FeatureDataset(dataset=self.cfg.dataset,
                                        split='test', root=root_path)}


        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class FeatureDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self.feature_types = torch.tensor(self.datamodule.cfg.dataset.node_feature_types)


