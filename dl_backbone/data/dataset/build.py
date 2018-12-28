from .multilabel import ProteinDataset


def build_dataset(cfg, transforms, is_train, dataset_name):
    """
    Arguments:
        cfg: model configuration
        transforms (callable): transforms to apply to each (image, target) sample
        is_train (bool): whether to setup the dataset for training or test
        dataset_name: str in ['train', 'valid', 'test']
    """
    if dataset_name == cfg.DATASETS.TRAIN:
        return ProteinDataset(cfg, cfg.DATASETS.TRAIN_LABEL, cfg.DATASETS.TRAIN_ROOT, transforms, is_train=True)
    elif dataset_name == cfg.DATASETS.VALID:
        return ProteinDataset(cfg, cfg.DATASETS.VALID_LABEL, cfg.DATASETS.VALID_ROOT, transforms, is_train=True)
    elif dataset_name == cfg.DATASETS.TEST:
        return ProteinDataset(cfg, cfg.DATASETS.TEST_LABEL, cfg.DATASETS.TEST_ROOT, transforms, is_train=False)
