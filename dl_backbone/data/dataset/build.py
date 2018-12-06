from .multilabel import TrainDataset


def build_dataset(cfg, transforms, is_train=True):
    """
    Arguments:
        cfg: model configuration
        transforms (callable): transforms to apply to each (image, target) sample
        is_train (bool): whether to setup the dataset for training or testing
    """
    if is_train:
        return TrainDataset(cfg.MODEL.NUM_CLASS, cfg.DATASETS.TRAIN_LABEL, cfg.DATASETS.TRAIN_ROOT, transforms)
    else:
        return TrainDataset(cfg.MODEL.NUM_CLASS, cfg.DATASETS.TEST_LABEL, cfg.DATASETS.TEST_ROOT, transforms)