import torch
from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck, model_zoo, model_urls


class _ResNet50Protein(ResNet):
    def __init__(self, num_classes):
        super(_ResNet50Protein, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        self.conv1_y = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def forward(self, x):
        rgb = x[:, [0, 1, 2], :, :]
        y = x[:, [3], :, :]
        x = self.conv1(rgb)
        x_y = self.conv1_y(y)
        x = x + x_y
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet50Protein(ResNet):
    def __init__(self, num_classes):
        super(ResNet50Protein, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        self.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)


def test():
    model = ResNet50Protein(28)
    i = torch.randn((16, 4, 128, 128))
    o = model(i)
    print(o.size())


if __name__ == '__main__':
    test()