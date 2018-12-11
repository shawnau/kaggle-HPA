from .multilabel import ProteinDataset


def build_dataset(cfg, transforms, is_train, dataset_name):
    """
    Arguments:
        cfg: model configuration
        transforms (callable): transforms to apply to each (image, target) sample
        is_train (bool): whether to setup the dataset for training or validation
        dataset_name: str in ['train', 'valid', 'test']
    """
    if dataset_name == cfg.DATASETS.TRAIN:
        return ProteinDataset(cfg.MODEL.NUM_CLASS, cfg.DATASETS.TRAIN_LABEL, cfg.DATASETS.TRAIN_ROOT, transforms, is_train)
    elif dataset_name == cfg.DATASETS.VALID:
        return ProteinDataset(cfg.MODEL.NUM_CLASS, cfg.DATASETS.VALID_LABEL, cfg.DATASETS.VALID_ROOT, transforms, is_train)
    elif dataset_name == cfg.DATASETS.TEST:
        return ProteinDataset(cfg.MODEL.NUM_CLASS, cfg.DATASETS.TEST_LABEL, cfg.DATASETS.TEST_ROOT, transforms, is_train)
