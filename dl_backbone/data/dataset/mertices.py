import torch
from sklearn.metrics import f1_score


def p_r(logits, targets, th=0.5):
    """
    :param logits: Tensor of shape (B, num_class)
    :param targets: Tensor of shape (B, num_class)
    :param th: Float threshold
    :return: (Float, Float)
    """
    assert len(logits) == len(targets)
    sample_num = len(targets)
    pred = (logits.sigmoid() > th).int()
    targets = targets.int()
    p, r = 0, 0
    for idx in range(sample_num):
        hit = (targets[idx] * pred[idx]).sum()
        if pred[idx].sum() > 0:
            p += hit / pred[idx].sum().float()
        r += hit / targets[idx].sum().float()
    return p / sample_num, r / sample_num


def _macro_f1(logits, targets, th=0.5):
    """
    sklearn implementation
    :param logits: Tensor of shape (B, num_class)
    :param targets: Tensor of shape (B, num_class)
    :param th: Float threshold
    :return:
    """
    pred = (logits.sigmoid() > th).int()
    targets = targets.numpy().astype(int)
    pred = pred.numpy().astype(int)
    return f1_score(targets, pred, average='macro')


def macro_f1(logits, targets, th=0.5):
    """
    pytorch implementation
    :param logits: Tensor of shape (B, num_class)
    :param targets: Tensor of shape (B, num_class)
    :param th: Float threshold
    :return: Float
    """
    preds = (logits.sigmoid() > th).float()
    targets = targets.float()
    score = 2.0 * torch.sum(preds * targets, dim=0) / torch.sum(preds + targets, dim=0)
    return score.mean().item()


if __name__ == "__main__":
    for _ in range(3):
        logits = torch.randn((16, 28))
        targets = torch.empty((16, 28)).random_(2)
        print(macro_f1(logits, targets), _macro_f1(logits, targets))