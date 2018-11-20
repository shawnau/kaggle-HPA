import os
from yacs.config import CfgNode as CN

_C = CN()

_C.INPUT = CN()

_C.INPUT.MIN_SIZE_TRAIN = 800
_C.INPUT.MAX_SIZE_TRAIN = 1333
_C.INPUT.MIN_SIZE_TEST = 800
_C.INPUT.MAX_SIZE_TEST = 1333

_C.INPUT.PIXEL_MEAN = [0., 0., 0., 0.]
_C.INPUT.PIXEL_STD = [1., 1., 1., 1.]