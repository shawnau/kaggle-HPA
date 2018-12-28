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


class MacroF1LogitLoss(nn.Module):
    def __init__(self):
        super(MacroF1LogitLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_pred:  Tensor of shape (B, num_class)
        :param y_true: Tensor of shape (B, num_class)
        :return: - macro f1 score
        """
        y_pred = y_pred.sigmoid()
        ep = torch.tensor(1e-7, dtype=torch.double)
        tp = torch.sum(y_true * y_pred, dim=0)
        fp = torch.sum((1 - y_true) * y_pred, dim=0)
        fn = torch.sum(y_true * (1 - y_pred), dim=0)

        p = tp / (tp + fp + ep)
        r = tp / (tp + fn + ep)

        f1 = 2 * p * r / (p + r + ep)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        return 1 - f1.mean()


def make_loss_module(cfg):
    if cfg.MODEL.LOSS == "BCE":
        return nn.BCEWithLogitsLoss()
    elif cfg.MODEL.LOSS == "weighted BCE":
        return nn.BCEWithLogitsLoss(weight=torch.Tensor(cfg.MODEL.LOSS_WEIGHT))
    elif cfg.MODEL.LOSS == "focal loss":
        return FocalLoss()
    elif cfg.MODEL.LOSS == "macro f1":
        return MacroF1LogitLoss()