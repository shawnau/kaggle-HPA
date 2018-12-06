from torch import nn
from .base.resnet import resnet50
from .loss import focal_loss, bce_loss


class NetWrapper(nn.Module):
    def __init__(self, cfg):
        super(NetWrapper, self).__init__()
        self.backbone = resnet50(num_classes=cfg.MODEL.NUM_CLASS)

    def forward(self, images, targets=None):
        """
        :param images: Tensor (B, C, H, W)
        :param targets: Tensor (B, num_class)
        :return:
            loss: scalar Tensor in train mode
            logits: Tensor (B, num_class) in test mode
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        logits = self.backbone(images)
        if self.training:
            loss = bce_loss(logits, targets)
            losses = {"bce loss": loss}
            return losses

        return logits
