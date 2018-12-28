import sys
sys.path.append('../')

import os
import torch
import torch.nn.functional as F
import pandas as pd
from dl_backbone.config import cfg
from dl_backbone.data.dataset.mertices import macro_f1
from threshold_optimizer import optimize_th


def submit(cfg, thresholds):
    test_pth = os.path.join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST, "predictions.pth")
    print("loading %s"%test_pth)

    df_test = pd.read_csv(cfg.DATASETS.TEST_LABEL)
    id_list = df_test["Id"].tolist()
    result = torch.load(test_pth)
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
    if isinstance(thresholds, float):
        submit_df.to_csv(
            os.path.join(cfg.OUTPUT_DIR,
                         "inference",
                         cfg.DATASETS.TEST,
                         "%s_%.2f.csv"%(cfg.MODEL.NAME, thresholds)),
            header=True, index=False)
    else:
        submit_df.to_csv(
            os.path.join(cfg.OUTPUT_DIR,
                         "inference",
                         cfg.DATASETS.TEST,
                         "%s_optim.csv" % cfg.MODEL.NAME),
            header=True, index=False)


def evaluation(train_csv, train_pth, valid_csv, valid_pth, thresholds, optim_th=None):
    def load_tensor(csv, pth):
        df = pd.read_csv(csv)
        raw_labels = df['Target'].tolist()
        labels = [list(map(int, item.split(' '))) for item in raw_labels]

        target_tensor = []
        for label in labels:
            label_vec = torch.zeros(cfg.MODEL.NUM_CLASS)
            label_vec[label] = 1
            target_tensor.append(label_vec)
        target_tensor = torch.stack(target_tensor, dim=0)

        result = torch.load(pth)
        logits = torch.stack(result, dim=0)
        assert len(logits) == len(target_tensor)
        return logits, target_tensor

    train_pred, train_tar = load_tensor(train_csv, train_pth)
    valid_pred, valid_tar = load_tensor(valid_csv, valid_pth)
    for threshold in thresholds:
        t_f1 = macro_f1(train_pred, train_tar, th=threshold)
        v_f1 = macro_f1(valid_pred, valid_tar, th=threshold)
        print("@%.4f  train f1 %.4f | valid f1: %.4f " % (threshold, t_f1, v_f1))
    t_loss = F.binary_cross_entropy_with_logits(train_pred, train_tar).item()
    v_loss = F.binary_cross_entropy_with_logits(valid_pred, valid_tar).item()
    print("bce loss: train: %.4f | valid: %.4f" % (t_loss, v_loss))

    if optim_th is not None:
        t_f1 = macro_f1(train_pred, train_tar, th=optim_th)
        v_f1 = macro_f1(valid_pred, valid_tar, th=optim_th)
        print("optim f1: train: %.4f | valid: %.4f " % (t_f1, v_f1))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pytorch Inference")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    train_pth = os.path.join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TRAIN, "predictions.pth")
    valid_pth = os.path.join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.VALID, "predictions.pth")
    optim_th = optimize_th(train_pth, valid_pth, init_number=0.30, lr=0.05, max_iter=14)
    evaluation(
        cfg.DATASETS.TRAIN_LABEL, train_pth,
        cfg.DATASETS.VALID_LABEL, valid_pth,
        thresholds=[0.05 * x for x in list(range(15))],
        optim_th=optim_th)

    submit(cfg, 0.10)
    submit(cfg, optim_th)


if __name__ == "__main__":
    main()