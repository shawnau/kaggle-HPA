import torch
from .samplers import IterationBasedBatchSampler
from .transforms import build_transforms
from .dataset import build_dataset
from .collate_batch import BatchCollator


def make_data_sampler(dataset, shuffle):
    if shuffle:
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


def make_data_loader(cfg, dataset_name, is_train=True, start_iter=0):
    if is_train:
        images_per_gpu = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_gpu = cfg.TEST.IMS_PER_BATCH
        shuffle = False
        num_iters = None
        start_iter = 0

    transforms = build_transforms(cfg, is_train)
    dataset = build_dataset(cfg, transforms, is_train, dataset_name)
    sampler = make_data_sampler(dataset, shuffle)

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
