import torch
from torch_geometric.data.lightning import LightningDataset

class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, datasets):
        super().__init__(train_dataset=datasets['train'], val_dataset=datasets['val'], test_dataset=datasets['test'],
                         batch_size=cfg.train.batch_size if 'debug' not in cfg.general.name else 2,
                         num_workers=cfg.train.num_workers,
                         pin_memory=getattr(cfg.dataset, "pin_memory", False))
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    def feature_counts(self):
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.feature.shape[1]
            break
        d = torch.range(1, num_classes)

        return d

class AbstractDatasetInfos:

    def compute_input_output_dims(self, datamodule):
        example_batch = next(iter(datamodule.train_dataloader()))

        self.input_dims = {'Feat': example_batch['feature'].size(2)}     
        self.output_dims = {'Feat': example_batch['feature'].size(2)}
