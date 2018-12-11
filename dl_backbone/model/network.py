from torch import nn
from .base import densenet121_protein, ResNet50Protein, GapNetPL, bninception_protein


class NetWrapper(nn.Module):
    def __init__(self, cfg):
        super(NetWrapper, self).__init__()
        self.model_name = cfg.MODEL.NAME
        model_list = {
            "resnet50": ResNet50Protein,
            "densenet": densenet121_protein,
            "gapnet-pl": GapNetPL,
            "bninception": bninception_protein
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
