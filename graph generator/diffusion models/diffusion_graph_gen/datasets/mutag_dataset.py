import os
import pathlib
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset
from datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos

atom_decoder = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']

class MUTAGDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, node_size,transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.split = split
        self.node_size = node_size
        #self.num_graphs = 200
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def download(self):
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]

        file_path = os.path.join(base_path, 'original_graphs/'+self.dataset_name+'/'+str(self.node_size)+'.pt')

        edge_indices, edge_feats, node_feats = torch.load(file_path)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        num_graphs = len(edge_indices)
        test_len = int(round(num_graphs * 0.2))
        train_len = int(round((num_graphs - test_len) * 0.8))
        val_len = num_graphs - train_len - test_len
        indices = torch.randperm(num_graphs, generator=g_cpu)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]

        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(edge_indices):
            if i in train_indices:
                train_data.append([edge_indices[i],edge_feats[i],node_feats[i]])
            elif i in val_indices:
                val_data.append([edge_indices[i],edge_feats[i],node_feats[i]])
            elif i in test_indices:
                test_data.append([edge_indices[i],edge_feats[i],node_feats[i]])
            else:
                raise ValueError(f'Index {i} not in any split')

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []


        for data in raw_dataset:
            edge_indices, edge_feats, node_feats = data
            num_nodes = node_feats.shape[0]


            #X = F.one_hot(torch.tensor(node_types), num_classes=len(atom_decoder)).float()
            y = torch.zeros([1, 0]).float()

            #edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            #edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            #edge_attr[:, 1] = 1
            #num_nodes = n * torch.ones(1, dtype=torch.long)

            # data = torch_geometric.data.Data(x=torch.tensor(node_feats), edge_index=torch.tensor(edge_indices), edge_attr=torch.tensor(edge_feats),
            #                                  y=y, n_nodes=num_nodes)

            data = torch_geometric.data.Data(x=torch.tensor(node_feats), edge_index=torch.tensor(edge_indices), edge_attr=torch.tensor(edge_feats),
                                             y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class MUTAGDataModule(MolecularDataModule):
    def __init__(self, cfg,size):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': MUTAGDataset(dataset_name=self.cfg.general.dataset_name,
                                                 split='train', root=root_path, node_size = size),
                    'val': MUTAGDataset(dataset_name=self.cfg.general.dataset_name,
                                        split='val', root=root_path,node_size = size),
                    'test': MUTAGDataset(dataset_name=self.cfg.general.dataset_name,
                                        split='test', root=root_path,node_size = size)}


        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class MUTAGinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False, meta=None):
        self.name = 'MUTAG'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = True
        #self.data = cfg
        self.datamodule = datamodule

        self.atom_decoder = atom_decoder
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        #https://www.acs.org/content/dam/acsorg/education/whatischemistry/periodic-table-of-elements/acs-periodic-table-poster_download.pdf
        self.atom_weights = {0: 12, 1: 14, 2:16, 3:19, 4:126.9, 5: 35.4, 6: 79.9}
        #https://www.thoughtco.com/valences-of-the-elements-chemistry-table-606458
        self.valencies = [4, 3, 2, 1, 7, 1, 1]
        self.num_atom_types = len(self.atom_decoder)
        self.max_weight = 350

        meta_files = dict(n_nodes=f'{self.name}_n_counts.txt',
                          node_types=f'{self.name}_atom_types.txt',
                          edge_types=f'{self.name}_edge_types.txt',
                          valency_distribution=f'{self.name}_valencies.txt')

        # self.n_nodes = torch.tensor([0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.097634362347889692e-06,
        #                              1.858580617408733815e-05, 5.007842264603823423e-05, 5.678996240021660924e-05,
        #                              1.244216400664299726e-04, 4.486406978685408831e-04, 2.253012731671333313e-03,
        #                              3.231865121051669121e-03, 6.709992419928312302e-03, 2.289564721286296844e-02,
        #                              5.411050841212272644e-02, 1.099515631794929504e-01, 1.223291903734207153e-01,
        #                              1.280680745840072632e-01, 1.445975750684738159e-01, 1.505961418151855469e-01,
        #                              1.436946094036102295e-01, 9.265746921300888062e-02, 1.820066757500171661e-02,
        #                              2.065089574898593128e-06])
        self.n_nodes = self.datamodule.node_counts()
        self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None
        self.node_types = torch.tensor([0.722338, 0.13661, 0.163655, 0.103549, 0.005411, 0.00150, 0.0])
        self.edge_types = torch.tensor([0.0472947, 0.062670, 0.0003524, 0.0486])
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        self.valency_distribution[:7] = torch.tensor([0.0, 0.1055, 0.2728, 0.3613, 0.2499, 0.00544, 0.00485])

        if meta is None:
            meta = dict(n_nodes=None, node_types=None, edge_types=None, valency_distribution=None)
        assert set(meta.keys()) == set(meta_files.keys())
        for k, v in meta_files.items():
            if (k not in meta or meta[k] is None) and os.path.exists(v):
                meta[k] = np.loadtxt(v)
                setattr(self, k, meta[k])
        if recompute_statistics or self.n_nodes is None:
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(meta_files["n_nodes"], self.n_nodes.numpy())
            self.max_n_nodes = len(self.n_nodes) - 1
        if recompute_statistics or self.node_types is None:
            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt(meta_files["node_types"], self.node_types.numpy())

        if recompute_statistics or self.edge_types is None:
            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(meta_files["edge_types"], self.edge_types.numpy())
        if recompute_statistics or self.valency_distribution is None:
            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt(meta_files["valency_distribution"], valencies.numpy())
            self.valency_distribution = valencies

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)



