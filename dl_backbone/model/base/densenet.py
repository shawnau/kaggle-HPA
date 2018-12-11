import torch.nn as nn
from torchvision.models.densenet import densenet121


def densenet121_protein(num_classes):
    model = densenet121(pretrained=True)
    model.features[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(64, num_classes)
    return model