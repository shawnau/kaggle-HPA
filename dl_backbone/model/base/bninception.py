import torch
from torch import nn
from collections import OrderedDict
from pretrainedmodels.models.bninception import bninception, pretrained_settings, BNInception, model_zoo


class BNInceptionProtein(BNInception):
    def init_pretrained(self):
        settings = pretrained_settings['bninception']['imagenet']
        self.load_state_dict(model_zoo.load_url(settings['url']))
        self.input_space = settings['input_space']
        self.input_size = settings['input_size']
        self.input_range = settings['input_range']
        self.mean = settings['mean']
        self.std = settings['std']

    def __init__(self, num_classes):
        super(BNInceptionProtein, self).__init__(1000)
        self.init_pretrained()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_pool_max = nn.AdaptiveMaxPool2d(1)
        self.conv1_7x7_s2 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.last_linear = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(2048)),
            ('drop1', nn.Dropout(0.5)),
            ('linear1', nn.Linear(2048, 1024)),
            ('relu1', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(1024)),
            ('drop2', nn.Dropout(p=0.5)),
            ('linear2', nn.Linear(1024, num_classes))
        ]))

    def logits(self, features):
        x_avg = self.global_pool(features)
        x_max = self.global_pool_max(features)
        x = torch.cat((x_avg, x_max), dim=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


def bninception_avg_protein(num_classes):
    model = bninception(pretrained="imagenet")
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.conv1_7x7_s2 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.last_linear = nn.Sequential(OrderedDict([
        ('bn1', nn.BatchNorm1d(1024)),
        ('drop1', nn.Dropout(0.5)),
        ('linear1', nn.Linear(1024, num_classes))
        ]))
    return model


def bninception_max_protein(num_classes):
    model = bninception(pretrained="imagenet")
    model.global_pool = nn.AdaptiveMaxPool2d(1)
    model.conv1_7x7_s2 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.last_linear = nn.Sequential(OrderedDict([
        ('bn1', nn.BatchNorm1d(1024)),
        ('drop1', nn.Dropout(0.5)),
        ('linear1', nn.Linear(1024, num_classes))
        ]))
    return model


def test():
    import torch
    model = BNInceptionProtein(28)
    i = torch.randn((2, 4, 128, 128))  # batch size == 1 would raise Exception
    o = model(i)
    print(o.size())

    for key, value in model.named_parameters():
        print(key)


if __name__ == '__main__':
    test()
