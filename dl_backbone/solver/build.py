# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .lr_scheduler import WarmupMultiStepLR
from dl_backbone.model.base import finetune_params


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_finetune_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.FINETUNE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.FINETUNE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        for name in finetune_params[cfg.MODEL.NAME]:
            if name in key:
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr=0, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def _make_lr_scheduler(cfg, optimizer):
    optimizer_dict = {
        "SetpLR": WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        ),
        "ReduceLROnPlateau": ReduceLROnPlateau(
            optimizer,
            factor=cfg.SOLVER.GAMMA,
            patience=cfg.SOLVER.PATIENCE)
    }
    return optimizer_dict[cfg.SOLVER.SCHEDULER]
