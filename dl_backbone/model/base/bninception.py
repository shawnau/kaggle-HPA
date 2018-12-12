from pretrainedmodels.models import bninception
from torch import nn


def bninception_protein(num_classes):
    model = bninception(pretrained="imagenet")
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.conv1_7x7_s2 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes),
            )
    return model


def test():
    import torch
    model = bninception_protein(28)
    i = torch.randn((1, 4, 128, 128))
    o = model(i)
    print(o.size())


if __name__ == '__main__':
    test()