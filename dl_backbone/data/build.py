import torch
import pickle
from .samplers import IterationBasedBatchSampler
from .transforms import build_transforms
from .dataset import build_dataset
from .collate_batch import BatchCollator


def make_data_sampler(cfg, dataset, shuffle):
    if shuffle:
        if cfg.DATALOADER.SAMPLER == "weighted":
            with open(cfg.DATALOADER.SAMPLER_WEIGHTS, 'rb') as f:
                weights = pickle.load(f)
            assert len(dataset) == len(weights)
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        else:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(sampler, images_per_batch, num_iters=None, start_iter=0):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=False
    )
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, dataset_name, is_train=True, start_iter=0, train_epoch=1):
    transforms = build_transforms(cfg, dataset_name, is_train)
    dataset = build_dataset(cfg, transforms, is_train, dataset_name)

    if is_train:
        shuffle = True
        images_per_gpu = cfg.SOLVER.IMS_PER_BATCH
        num_iters = int(train_epoch * len(dataset) // images_per_gpu)
    else:
        shuffle = False
        images_per_gpu = cfg.TEST.IMS_PER_BATCH
        num_iters = None

    sampler = make_data_sampler(cfg, dataset, shuffle)
    batch_sampler = make_batch_data_sampler(sampler, images_per_gpu, num_iters, start_iter)
    collator = BatchCollator()
    num_workers = cfg.DATALOADER.NUM_WORKERS

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )

    return data_loader
