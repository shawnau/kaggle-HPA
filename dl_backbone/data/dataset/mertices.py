import torch
import pandas as pd
from sklearn.metrics import f1_score
from dl_backbone.config import cfg
import torch.nn.functional as F


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


def macro_f1(logits, target, th=0.5):
    """
    :param logits: Tensor of shape (B, num_class)
    :param target: Tensor of shape (B, num_class)
    :param th: Float threshold
    :return:
    """
    pred = (logits.sigmoid() > th).int()
    target = target.numpy().astype(int)
    pred = pred.numpy().astype(int)
    return f1_score(target, pred, average='macro')


def evaluation(label_file, eval_file, thresholds):
    df_eval = pd.read_csv(label_file)
    raw_labels = df_eval['Target'].tolist()
    labels = [list(map(int, item.split(' '))) for item in raw_labels]

    target_tensor = []
    for label in labels:
        label_vec = torch.zeros(cfg.MODEL.NUM_CLASS)
        label_vec[label] = 1
        target_tensor.append(label_vec)
    target_tensor = torch.stack(target_tensor, dim=0)

    result = torch.load(eval_file)
    logits = torch.stack(result, dim=0)
    assert len(logits) == len(target_tensor)

    for threshold in thresholds:
        _mf1 = macro_f1(logits, target_tensor, th=threshold)
        p, r = p_r(logits, target_tensor, th=threshold)
        print("@%.2f precision: %.4f | recall: %.4f | macro f1: %.4f " % (threshold, p, r, _mf1))
    print("bce loss: %.4f" % F.binary_cross_entropy_with_logits(logits, target_tensor).item())
