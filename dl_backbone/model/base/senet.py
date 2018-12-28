from collections import OrderedDict
import torch
from torch import nn
from pretrainedmodels.models.senet import SENet, SEResNeXtBottleneck, pretrained_settings, model_zoo


class SENeXt50Protein(SENet):
    def __init__(self, num_classes):
        super(SENeXt50Protein, self).__init__(
            SEResNeXtBottleneck,
            [3, 4, 6, 3], groups=32, reduction=16,
            dropout_p=None, inplanes=64, input_3x3=False,
            downsample_kernel_size=1, downsample_padding=0,
            num_classes=1000)

        settings = pretrained_settings['se_resnext50_32x4d']['imagenet']
        self.load_state_dict(model_zoo.load_url(settings['url']))

        self.layer0[0] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.last_linear = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(4096)),
            ('drop1', nn.Dropout(p=0.5)),
            ('linear1', nn.Linear(4096, 2048)),
            ('relu1', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(2048)),
            ('drop2', nn.Dropout(p=0.5)),
            ('linear2', nn.Linear(2048, num_classes))
        ]))

    def logits(self, x):
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = torch.cat((x_avg, x_max), dim=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


def test():
    model = SENeXt50Protein(28)
    i = torch.randn((2, 3, 128, 128))
    o = model(i)
    print(o.size())

    for key, value in model.named_parameters():
        print(key)


if __name__ == '__main__':
    test()