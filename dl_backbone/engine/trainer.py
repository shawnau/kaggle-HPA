# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
from tqdm import tqdm

import torch
import numpy as np

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
    train_epoch,
    checkpoint_period,
    is_mixup,
    arguments
):
    logger = logging.getLogger("model.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    start_epoch = arguments["epoch"]
    model.train()
    end = time.time()
    last_loss, min_loss = 999, 999
    for epoch in range(start_epoch, train_epoch):
        max_iter = len(train_data_loader)
        arguments["epoch"] = epoch
        for iteration, (images, targets, _) in enumerate(train_data_loader):
            data_time = time.time() - end

            images = images.to(device)
            targets = targets.to(device)

            if is_mixup == 'on':
                loss = mixup(images, targets, model, loss_module)
            else:
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

            if iteration % 200 == 0 or iteration == (max_iter - 1):
                valid_meters = do_valid(model, loss_module, valid_data_loader)
                meters.update(**valid_meters)
                if valid_meters["val loss"] < min_loss:
                    min_loss = valid_meters["val loss"]
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch:.2f}",
                            "{meters}",
                            "lr: {lr:.6f}"
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch+float(iteration)/max_iter,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"]
                    )
                )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss)
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step(epoch)
        elif scheduler is not None:
            scheduler.step()

        if epoch % checkpoint_period == 0 and epoch > 0:
            checkpointer.save("model_{:07d}".format(epoch), **arguments)
        elif min_loss < last_loss:
            last_loss = min_loss
            checkpointer.save("model_minloss", **arguments)


def do_valid(model, loss_module, valid_data_loader):
    model.eval()
    with torch.no_grad():
        all_logits, all_targets = [], []
        for i, batch in tqdm(enumerate(valid_data_loader)):
            images, targets, image_ids = batch
            all_logits.append(model(images))
            all_targets.append(targets)
        all_logits  = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        loss = loss_module(all_logits.cuda(), all_targets.cuda())
        f1 = macro_f1(all_logits.cuda(), all_targets.cuda(), th=0.20)
    model.train()
    return {"val loss": loss.item(), "val f1": f1}


def mixup(images, targets, model, loss_module):
    """
    :param images: Tensor (N, 4, H, W)
    :param targets: Tensor (N, num_classes)
    :return:
    """
    alpha = 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0))
    inputs = lam * images + (1 - lam) * images[index, :]
    targets_a, targets_b = targets, targets[index]

    logits = model(inputs)
    loss = lam * loss_module(logits, targets_a) + (1-lam) * loss_module(logits, targets_b)
    return loss
