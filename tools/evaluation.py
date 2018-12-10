import sys
sys.path.append('../')

import os
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
from dl_backbone.config import cfg
from dl_backbone.data.dataset.mertices import p_r, macro_f1
from threshold_optimizer import optimize_th


def calc_statistics():
    from dl_backbone.data.build import make_data_loader
    loader = make_data_loader(cfg, is_train=False)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for iteration, (images, targets, indices) in enumerate(loader):
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_size
        running_mean = mean / nb_samples
        running_std = std / nb_samples
        print("iter %d running mean: " % iteration, running_mean)
        print("iter %d running std : " % iteration, running_std)


def submit(sample_csv, eval_file, thresholds):
    print("loading data...")
    df_test = pd.read_csv(sample_csv)
    id_list = df_test["Id"].tolist()
    result = torch.load(eval_file)
    logits = torch.stack(result, dim=0)
    preds = (logits.sigmoid() > thresholds).int()
    label_list = []
    print("converting...")
    for pred in preds:
        labels = pred.nonzero().squeeze().numpy().tolist()
        if isinstance(labels, int):
            labels = [labels]
        elif isinstance(labels, list) and (len(labels) == 0):
            labels = [0]
        assert isinstance(labels, list)
        labels = list(map(str, labels))
        label_list.append(" ".join(labels))
    submit_df = pd.DataFrame({'Id': id_list, 'Predicted': label_list})
    submit_df.to_csv("submit.csv", header=True, index=False)


def evaluation(label_file, eval_file, thresholds, optim_th=None):
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

    if optim_th is not None:
        _mf1 = macro_f1(logits, target_tensor, th=optim_th)
        p, r = p_r(logits, target_tensor, th=optim_th)
        print("optim precision: %.4f | recall: %.4f | macro f1: %.4f " % (p, r, _mf1))


if __name__ == "__main__":
    # train_name = cfg.DATASETS.TRAIN
    # valid_name = cfg.DATASETS.VALID
    # test_name = cfg.DATASETS.TEST

    #train_pth = os.path.join(cfg.OUTPUT_DIR, "inference", train_name, "predictions.pth")
    #valid_pth = os.path.join(cfg.OUTPUT_DIR, "inference", valid_name, "predictions.pth")
    #test_pth = os.path.join(cfg.OUTPUT_DIR, "inference", test_name, "predictions.pth")
    #optim_th = optimize_th(0.3, train_pth, valid_pth)

    #evaluation(cfg.DATASETS.VALID_LABEL, valid_pth,
    #           thresholds=[0.05 * x for x in range(11)],
    #           optim_th=None)

    #submit(cfg.DATASETS.TEST_LABEL, test_pth, optim_th)
    calc_statistics()