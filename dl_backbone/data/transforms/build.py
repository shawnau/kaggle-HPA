from torchvision import transforms as T
from imgaug import augmenters as iaa


def build_transforms(cfg, dataset_name, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        mean = cfg.INPUT.TRAIN_PIXEL_MEAN
        std = cfg.INPUT.TRAIN_PIXEL_STD

        flip_aug = iaa.Sequential([
                iaa.OneOf([
                    iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270),
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                ])
            ])

        crop_aug = iaa.Sometimes(
            0.5,
            iaa.Sequential([
                iaa.OneOf([
                    iaa.CropToFixedSize(288, 288),
                    iaa.CropToFixedSize(320, 320),
                    iaa.CropToFixedSize(352, 352),
                    iaa.CropToFixedSize(384, 384),
                    iaa.CropToFixedSize(416, 416),
                    iaa.CropToFixedSize(448, 448),
                ])
            ])
        )

        blur_aug = iaa.Sometimes(0.5, iaa.GaussianBlur((0.0, 1.0)))

        if cfg.DATALOADER.AUGMENT == 'normal':
            aug = flip_aug
        elif cfg.DATALOADER.AUGMENT == 'heavy':
            aug = iaa.Sequential([flip_aug, crop_aug, iaa.Scale({"height": min_size, "width": max_size})])
        elif cfg.DATALOADER.AUGMENT == 'extreme':
            aug = iaa.Sequential([flip_aug, crop_aug, blur_aug])
        else:
            raise KeyError("aug mode not recognized")
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


def build_tta_transforms(cfg):
    """
    :param cfg:
    :param dataset_name: cfg.DATASETS.TEST
    :param is_train: False
    :return: List[transforms]
    """

    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    mean = cfg.INPUT.TEST_PIXEL_MEAN
    std = cfg.INPUT.TEST_PIXEL_STD

    augments = [
        iaa.Noop(),
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270),
        iaa.Fliplr(1.0),
        iaa.Flipud(1.0)
    ]

    transforms = []
    for aug in augments:
        transforms.append(T.Compose(
            [
                aug.augment_image,
                T.ToPILImage(),  # magic fix for negative stride problem
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ]
        ))

    return transforms