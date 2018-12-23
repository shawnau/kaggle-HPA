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
    valid_data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments
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

        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = loss_module(logits, targets)
        meters.update(**{"loss": loss})

        scheduler.step(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 200 == 0 or iteration == (max_iter - 1):
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

    checkpointer.save("model_{:07d}".format(max_iter), **arguments)
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
        weight = torch.Tensor([1.00, 3.29, 2.03, 3.21, 2.78, 2.63, 3.10,
         2.17, 5.94, 6.04, 6.12, 3.63, 3.61, 4.04,
         3.42, 7.18, 4.16, 5.22, 3.77, 3.11, 5.24,
         1.79, 3.41, 2.08, 5.26, 1.00, 4.76, 6.48]).cuda()
        bce_loss = F.binary_cross_entropy_with_logits(all_logits.cuda(), all_targets.cuda(), weight=weight)
        f1 = macro_f1(all_logits.cuda(), all_targets.cuda(), th=0.20)
    model.train()
    return {"val loss": bce_loss, "val f1": f1}
