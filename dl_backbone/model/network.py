from torch import nn
from .base import *


class NetWrapper(nn.Module):
    def __init__(self, cfg):
        super(NetWrapper, self).__init__()
        self.model_name = cfg.MODEL.NAME
        model_list = {
            "resnet50": ResNet50Protein,
            "resnet34_maxavg": ResNet34MaxAvgProtein,
            "resnet34_maxavg_no_dropout": ResNet34MaxAvgNoDropout,
            "resnet34_maxpool": ResNet34MaxProtein,
            "resnet18_maxavg": ResNet18MaxAvgProtein,
            "resnet18_maxavg_no_dropout": ResNet18MaxAvgNoDropout,
            "resnet18_maxpool": ResNet18MaxProtein,
            "resnet18_avgpool": ResNet18AvgProtein,
            "resnet18_3c": ResNet183CProtein,
            "resnet343c": ResNet34Protein3C,
            "densenet": densenet121_protein,
            "gapnet-pl": GapNetPL,
            "bninception_avg": bninception_avg_protein,
            "bninception_max": bninception_max_protein,
            "bninception_maxavg": BNInceptionProtein,
            "seresnext50": SENeXt50Protein
        }
        self.backbone = model_list[cfg.MODEL.NAME](cfg.MODEL.NUM_CLASS)

    def forward(self, images):
        """
        :param images: Tensor (B, C, H, W)
        :param targets: Tensor (B, num_class)
        :return:
            loss: scalar Tensor in train mode
            logits: Tensor (B, num_class) in test mode
        """

        if self.model_name == "inceptionv3" and self.training:
            logits, aux_logits = self.backbone(images)
            return logits, aux_logits
        else:
            logits = self.backbone(images)
            return logits
