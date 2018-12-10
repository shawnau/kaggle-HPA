from torch import nn
from .base import DenseNet, resnet50, Inception3, GapNetPL


class NetWrapper(nn.Module):
    def __init__(self, cfg):
        super(NetWrapper, self).__init__()
        self.backbone = Inception3(num_classes=cfg.MODEL.NUM_CLASS)

    def forward(self, images):
        """
        :param images: Tensor (B, C, H, W)
        :param targets: Tensor (B, num_class)
        :return:
            loss: scalar Tensor in train mode
            logits: Tensor (B, num_class) in test mode
        """
        logits = self.backbone(images)
        return logits
