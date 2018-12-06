import torch
import pandas as pd
from sklearn.metrics import f1_score
from dl_backbone.config import cfg


def macro_f1(pred, target):
    """
    :param target: Tensor of shape (B, num_class)
    :param pred: Tensor of shape (B, num_class)
    :return:
    """
    target = target.numpy().astype(int)
    pred = pred.numpy().astype(int)
    return f1_score(target, pred, average='macro')


def evaluation(label_file, eval_file):
    df_eval = pd.read_csv(label_file)
    ids = df_eval["Id"].tolist()
    raw_labels = df_eval['Target'].tolist()
    labels = [list(map(int, item.split(' '))) for item in raw_labels]

    target_tensor = []
    for label in labels:
        label_vec = torch.zeros(cfg.MODEL.NUM_CLASS)
        label_vec[label] = 1
        target_tensor.append(label_vec)
    target_tensor = torch.cat(target_tensor)

    result = torch.load(eval_file)
    pred_tensor = []
    for img_id in ids:
        pred_vec = result[img_id] > 0.5
        pred_tensor.append(pred_vec)
    pred_tensor = torch.cat(pred_tensor)

    print("macro f1: ", macro_f1(pred_tensor, target_tensor))
