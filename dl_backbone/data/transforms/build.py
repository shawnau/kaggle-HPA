from torchvision import transforms as T
from imgaug import augmenters as iaa


def build_transforms(cfg, dataset_name, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        mean = cfg.INPUT.TRAIN_PIXEL_MEAN
        std = cfg.INPUT.TRAIN_PIXEL_STD

        aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ]),
            # iaa.CropToFixedSize(min_size, max_size)
        ], random_order=True)
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        if dataset_name == cfg.DATASETS.TRAIN:
            mean = cfg.INPUT.TRAIN_PIXEL_MEAN
            std = cfg.INPUT.TRAIN_PIXEL_STD
        elif dataset_name == cfg.DATASETS.VALID:
            mean = cfg.INPUT.VALID_PIXEL_MEAN
            std = cfg.INPUT.VALID_PIXEL_STD
        elif dataset_name == cfg.DATASETS.TEST:
            mean = cfg.INPUT.TEST_PIXEL_MEAN
            std = cfg.INPUT.TEST_PIXEL_STD
        else:
            raise KeyError('dataset name not recognized')

        aug = iaa.Noop()

    transform = T.Compose(
        [
            aug.augment_image,
            T.ToPILImage(),  # magic fix for negative stride problem
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ]
    )
    return transform