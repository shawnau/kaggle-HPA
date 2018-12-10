from .transforms import RandomClockwiseRotate, RandomHVFlip, Resize, RandomRotation
from torchvision import transforms as T
from imgaug import augmenters as iaa


def _build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        mean = cfg.INPUT.TRAIN_PIXEL_MEAN
        std = cfg.INPUT.TRAIN_PIXEL_STD
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        mean = cfg.INPUT.TEST_PIXEL_MEAN
        std = cfg.INPUT.TEST_PIXEL_STD
        flip_prob = 0

    transform = T.Compose(
        [
            RandomClockwiseRotate(0, flip_prob),
            RandomClockwiseRotate(1, flip_prob),
            RandomHVFlip(0, flip_prob),
            RandomHVFlip(1, flip_prob),
            # RandomRotation((-45, 45), is_train=is_train),
            Resize((min_size, max_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ]
    )
    return transform


def build_transforms(cfg, is_train=True):
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
                iaa.Flipud(0.5),
                # iaa.CropToFixedSize(min_size, max_size)
            ])], random_order=True)
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        mean = cfg.INPUT.TEST_PIXEL_MEAN
        std = cfg.INPUT.TEST_PIXEL_STD

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