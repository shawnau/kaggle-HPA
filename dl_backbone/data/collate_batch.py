# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from dl_backbone.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images(ImageList) and targets(Tensor).
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = torch.stack(transposed_batch[1])
        img_ids = transposed_batch[2]
        return images, targets, img_ids
