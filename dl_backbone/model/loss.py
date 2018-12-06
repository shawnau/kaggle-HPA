import torch.nn.functional as F


def focal_loss(pred, target, gamma=2):
    if not (target.size() == pred.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})"
                         .format(target.size(), pred.size()))

    max_val = (-pred).clamp(min=0)
    bce_loss = pred - pred * target + max_val + ((-max_val).exp() + (-pred - max_val).exp()).log()
    inv_prob = F.logsigmoid(-pred * (target * 2.0 - 1.0))
    _focal_loss = (inv_prob * gamma).exp() * bce_loss
    return _focal_loss.sum(dim=1).mean()


def bce_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)
