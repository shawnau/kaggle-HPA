# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F

from dl_backbone.data.dataset.mertices import macro_f1
from dl_backbone.utils.metric_logger import MetricLogger


def do_train(
    model,
    loss_module,
    train_data_loader,
    optimizer,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    scheduler=None,
    valid_data_loader=None
):
    logger = logging.getLogger("model.trainer")
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
        if scheduler is not None:
            scheduler.step()

        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = loss_module(logits, targets)

        meters.update(**{"loss": loss})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 100 == 0 or iteration == (max_iter - 1):
            if valid_data_loader is not None:
                meters.update(**do_valid(model, valid_data_loader))
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}"
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"]
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
    with torch.no_grad():
        all_logits, all_targets = [], []
        for i, batch in tqdm(enumerate(valid_data_loader)):
            images, targets, image_ids = batch
            all_logits.append(model(images))
            all_targets.append(targets)
        all_logits  = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        bce_loss = F.binary_cross_entropy_with_logits(all_logits.cuda(), all_targets.cuda())
        f1 = macro_f1(all_logits.cuda(), all_targets.cuda(), th=0.15)
    model.train()
    return {"val loss": bce_loss, "val f1": f1}
