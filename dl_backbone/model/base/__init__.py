from .densenet import densenet121_protein
from .resnet import ResNet50Protein, ResNet34Protein
from .gapnet import GapNetPL
from .bninception import bninception_protein


finetune_params = {
            "resnet50":
                ['fc.bn1.weight',
                 'fc.bn1.bias',
                 'fc.linear1.weight',
                 'fc.linear1.bias',
                 'fc.bn2.weight',
                 'fc.bn2.bias',
                 'fc.linear2.weight',
                 'fc.linear2.bias',
                 'conv1_y.weight',
                 'conv1_y.bias'],
            "resnet34":
                ['fc.bn1.weight',
                 'fc.bn1.bias',
                 'fc.linear1.weight',
                 'fc.linear1.bias',
                 'fc.bn2.weight',
                 'fc.bn2.bias',
                 'fc.linear2.weight',
                 'fc.linear2.bias',
                 'conv1_y.weight',
                 'conv1_y.bias'],
            "densenet":
                ['features.conv0.weight',
                 'classifier.weight',
                 'classifier.bias'],
            "bninception":
                ['conv1_7x7_s2.weight',
                 'conv1_7x7_s2.bias',
                 'last_linear.bn1.weight',
                 'last_linear.bn1.bias',
                 'last_linear.linear1.weight',
                 'last_linear.linear1.bias']
        }
