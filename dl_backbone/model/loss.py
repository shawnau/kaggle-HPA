import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, target, pred):
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), pred.size()))
        max_val = (-pred).clamp(min=0)
        bce_loss = pred - pred * target + max_val + ((-max_val).exp() + (-pred - max_val).exp()).log()
        inv_prob = F.logsigmoid(-pred * (target * 2.0 - 1.0))
        _focal_loss = (inv_prob * self.gamma).exp() * bce_loss
        return _focal_loss.sum(dim=1).mean()

    def __repr__(self):
        return "focal loss"


def make_loss_module(cfg):
    loss_dict = {
        "BCE": nn.BCEWithLogitsLoss(),
        "weighted BCE": nn.BCEWithLogitsLoss(weight=torch.Tensor(cfg.MODEL.LOSS_WEIGHT)),
        "focal loss": FocalLoss()
    }
    return loss_dict[cfg.MODEL.LOSS]