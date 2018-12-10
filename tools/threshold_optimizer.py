import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dl_backbone.config import cfg


def load_data(csv, pth):
    df_eval = pd.read_csv(csv)
    raw_labels = df_eval['Target'].tolist()
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


class MacroF1(nn.Module):
    def __init__(self, init_num, num_class):
        super(MacroF1, self).__init__()
        self.thresholds = nn.Parameter(torch.empty(num_class).fill_(init_num))

    def forward(self, logits, targets):
        """
        :param logits:  Tensor of shape (B, num_class)
        :param targets: Tensor of shape (B, num_class)
        :return: - macro f1 score
        """
        amplifier = 50
        preds = torch.sigmoid((logits.sigmoid() - self.thresholds)*amplifier)
        score = 2.0 * torch.sum(preds * targets, dim=0) / torch.sum(preds + targets, dim=0)
        return -score.mean()


def optimize_th(init_number, train_pth, valid_pth):
    print("loading data...")
    train_x, train_y = load_data(cfg.DATASETS.TRAIN_LABEL, train_pth)
    valid_x, valid_y = load_data(cfg.DATASETS.VALID_LABEL, valid_pth)
    print("Done!")
    model = MacroF1(init_number, cfg.MODEL.NUM_CLASS)
    optimizer = optim.LBFGS(model.parameters(), lr=0.05, max_iter=10)

    def closure():
        optimizer.zero_grad()
        train_loss = model(train_x, train_y)
        with torch.no_grad():
            test_loss = model(valid_x, valid_y)
            print('train loss: %.6f test loss: %.6f' % (train_loss.item(), test_loss.item()))
        train_loss.backward()
        return train_loss

    optimizer.step(closure)

    return model.thresholds.data
