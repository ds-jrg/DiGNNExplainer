import torch
import torch.nn as nn
import wandb
from abstract_metrics import CrossEntropyMetric

class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self):
        super().__init__()

        self.feat_loss = CrossEntropyMetric()


    def forward(self, masked_pred_Feat, true_Feat,log: bool):

        true_Feat = torch.reshape(true_Feat, (-1, true_Feat.size(-1)))  # (bs * n, dx)
        masked_pred_Feat = torch.reshape(masked_pred_Feat, (-1, masked_pred_Feat.size(-1)))  # (bs * n, dx)

        # Remove masked rows

        mask_Feat = (true_Feat != 0.).any(dim=-1)
        flat_true_Feat = true_Feat[mask_Feat, :]
        flat_pred_Feat = masked_pred_Feat[mask_Feat, :]

        loss_Feat = self.feat_loss(flat_pred_Feat, flat_true_Feat) if true_Feat.numel() > 0 else 0.0

        if log:
            to_log = {
                      "train_loss/Feat_CE": self.feat_loss.compute() if true_Feat.numel() > 0 else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss_Feat

    def reset(self):
        for metric in [self.feat_loss]:
            metric.reset()

    def log_epoch_metrics(self):

        epoch_feat_loss = self.feat_loss.compute() if self.feat_loss.total_samples > 0 else -1

        to_log = {
                  "train_epoch/Feat_CE": epoch_feat_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log





