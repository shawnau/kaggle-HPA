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

_C.INPUT.TRAIN_PIXEL_MEAN = [0.0874, 0.0518, 0.0514, 0.0793]
_C.INPUT.TRAIN_PIXEL_STD =  [0.1273, 0.0805, 0.1309, 0.1174]

_C.INPUT.VALID_PIXEL_MEAN = _C.INPUT.TRAIN_PIXEL_MEAN
_C.INPUT.VALID_PIXEL_STD = _C.INPUT.TRAIN_PIXEL_STD

_C.INPUT.TEST_PIXEL_MEAN = [0.0591, 0.0453, 0.0407, 0.0592]
_C.INPUT.TEST_PIXEL_STD =  [0.1037, 0.0798, 0.1066, 0.0988]

# ---------------------------------------------------------------------------- #
# Dataset config
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()

_C.DATASETS.TRAIN = "train"
_C.DATASETS.TRAIN_ROOT = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/train"
_C.DATASETS.TRAIN_LABEL = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/train_split.csv"

_C.DATASETS.VALID = "valid"
_C.DATASETS.VALID_ROOT = _C.DATASETS.TRAIN_ROOT
_C.DATASETS.VALID_LABEL = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/valid_split.csv"

_C.DATASETS.TEST = "test"
_C.DATASETS.TEST_ROOT = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/test"
_C.DATASETS.TEST_LABEL = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/sample_submission.csv"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.SAMPLER = "even"
_C.DATALOADER.SAMPLER_WEIGHTS = "/unsullied/sharefs/ouxiaoxuan/isilon/kaggle/sample_weights.pickle"
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.AUGMENT = "normal"  # normal, heavy, extreme
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0

# ---------------------------------------------------------------------------- #
# Model config
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = "resnet34"
_C.MODEL.NUM_CLASS = 28
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHT = ""
_C.MODEL.LOSS = "BCE"
_C.MODEL.LOSS_WEIGHT = [1.00, 3.28, 2.03, 3.19, 2.77, 2.61, 3.08, 2.17, 5.94, 6.01, 6.08, 3.64, 3.62, 4.03, 3.45, 7.15, 4.16, 5.23, 3.77, 3.12, 5.25, 1.80, 3.41, 2.08, 5.23, 1.00, 4.75, 6.44]

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "sgd"  # sgd, adam
_C.SOLVER.TRAIN_EPOCH = 100
_C.SOLVER.BASE_LR = 0.02
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.FINETUNE = "off"
_C.SOLVER.FINETUNE_EPOCH = 3
_C.SOLVER.FINETUNE_LR = 0.002

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

# lr will divide by gamma after each step
# options: ["ReduceLROnPlateau", "MultiSetpLR", "StepLR", "CosineAnnealingLR"]
_C.SOLVER.SCHEDULER = "ReduceLROnPlateau"
# for ReduceLR
_C.SOLVER.PATIENCE = 10
_C.SOLVER.GAMMA = 0.5
# for MultiSetpLR
_C.SOLVER.STEPS = (10, 20, 30, 40, 50, 60, 70, 80, 90)
# for SetpLR
_C.SOLVER.STEP_SIZE = 10
# for CosineAnnealingLR
_C.SOLVER.T_MAX = 10

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.MIXUP = 'off'

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
_C.TEST.IMS_PER_BATCH = 32
_C.TEST.TTA = 'off'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "/unsullied/sharefs/ouxiaoxuan/isilon/dl_backbone/tools/incep_bce_sgd"