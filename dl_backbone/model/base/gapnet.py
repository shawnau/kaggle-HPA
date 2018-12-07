import torch
import torch.nn as nn
import torch.nn.functional as F


class GapNetPL(nn.Module):
    def __init__(self, num_class):
        super(GapNetPL, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(224, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out   = F.selu(self.bn1(self.conv1(x)))
        out_a = F.max_pool2d(out, kernel_size=2)
        out   = F.selu(self.bn2(self.conv2(out_a)))
        out   = F.selu(self.bn3(self.conv3(out)))
        out   = F.selu(self.bn4(self.conv4(out)))
        out_b = F.max_pool2d(out, kernel_size=2)
        out   = F.selu(self.bn5(self.conv5(out_b)))
        out   = F.selu(self.bn6(self.conv6(out)))
        out_c = F.selu(self.bn7(self.conv7(out)))

        vec_a = F.adaptive_avg_pool2d(out_a, (1, 1))
        vec_b = F.adaptive_avg_pool2d(out_b, (1, 1))
        vec_c = F.adaptive_avg_pool2d(out_c, (1, 1))

        vac_all = torch.cat((vec_a, vec_b, vec_c), dim=1)
        vec_all = vac_all.view(vac_all.size(0), -1)

        out = F.relu(self.fc1(F.dropout(vec_all, p=0.3)))
        out = F.relu(self.fc1(F.dropout(out, p=0.3)))
        out = self.fc3(out)

        return out


if __name__ == "__main__":
    model = GapNetPL(28)

