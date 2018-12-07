import os
from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Input config
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()

_C.INPUT.MIN_SIZE_TRAIN = 512
_C.INPUT.MAX_SIZE_TRAIN = 512
_C.INPUT.MIN_SIZE_TEST = 512
_C.INPUT.MAX_SIZE_TEST = 512

_C.INPUT.TRAIN_PIXEL_MEAN = [0.08069, 0.05258, 0.05487, 0.08282]
_C.INPUT.TRAIN_PIXEL_STD = [0.13704, 0.10145, 0.15313, 0.13814]

_C.INPUT.TEST_PIXEL_MEAN = _C.INPUT.TRAIN_PIXEL_MEAN
_C.INPUT.TEST_PIXEL_STD = _C.INPUT.TRAIN_PIXEL_STD

# for test
# _C.INPUT.TEST_PIXEL_MEAN = [0.05913, 0.0454 , 0.04066, 0.05928]
# _C.INPUT.TEST_PIXEL_STD = [0.11734, 0.09503, 0.129 , 0.11528]

# ---------------------------------------------------------------------------- #
# Dataset config
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.TRAIN = "train"
_C.DATASETS.TEST = "valid"

_C.DATASETS.TRAIN_ROOT = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/train"
_C.DATASETS.TRAIN_LABEL = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/train_split.csv"

_C.DATASETS.TEST_ROOT = _C.DATASETS.TRAIN_ROOT
_C.DATASETS.TEST_LABEL = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/valid_split.csv"

# for test
# _C.DATASETS.TEST = "test"
# _C.DATASETS.TEST_ROOT = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/test"
# _C.DATASETS.TEST_LABEL = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/sample_submission.csv"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0

# ---------------------------------------------------------------------------- #
# Model config
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NUM_CLASS = 28
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 5000

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 32

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 16

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."