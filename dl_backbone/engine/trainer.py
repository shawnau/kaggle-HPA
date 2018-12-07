# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
from tqdm import tqdm

import torch
from torch.distributed import deprecated as dist
import torch.nn.functional as F

from dl_backbone.utils.comm import get_world_size
from dl_backbone.utils.metric_logger import MetricLogger


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    train_data_loader,
    valid_data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("dl_backbone.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        data_time = time.time() - end
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.tensors.to(device)
        targets = targets.to(device)

        loss_dict = model(images, targets)
        # using nn.DataParallel will merge loss into one tensor
        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k].sum()

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values()) / float(torch.cuda.device_count())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 100 == 0 or iteration == (max_iter - 1):
            valid_loss = do_valid(model, valid_data_loader)
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "valid: {valid:.4f}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    valid=valid_loss,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0 and iteration > 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

    checkpointer.save("model_{:07d}".format(iteration), **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


def do_valid(model, valid_data_loader):
    model.eval()
    mean_acc, mean_loss, batches = 0, 0, 0
    for i, batch in tqdm(enumerate(valid_data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            logits = model(images.tensors)
            mean_loss += F.binary_cross_entropy_with_logits(logits.cuda(), targets.cuda())
            batches += 1
    model.train()
    return mean_loss.item() / float(batches)