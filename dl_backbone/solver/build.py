# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR, CosineAnnealingLR
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

    if cfg.SOLVER.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(params, lr)
    else:
        raise KeyError("optimizer not supported")
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


def make_lr_scheduler(cfg, optimizer):
    if isinstance(optimizer, torch.optim.SGD):
        if cfg.SOLVER.SCHEDULER == "ReduceLROnPlateau":
            return ReduceLROnPlateau(
                optimizer,
                factor=cfg.SOLVER.GAMMA,
                patience=cfg.SOLVER.PATIENCE)
        elif cfg.SOLVER.SCHEDULER == "MultiStepLR":
            return MultiStepLR(
                optimizer,
                milestones=cfg.SOLVER.STEPS,
                gamma=cfg.SOLVER.GAMMA
            )
        elif cfg.SOLVER.SCHEDULER == "StepLR":
            return StepLR(
                optimizer,
                step_size=cfg.SOLVER.STEP_SIZE,
                gamma=cfg.SOLVER.GAMMA
            )
        elif cfg.SOLVER.SCHEDULER == "CosineAnnealingLR":
            return CosineAnnealingLR(
                optimizer,
                T_max=cfg.SOLVER.T_MAX,
                eta_min=1e-5
            )
        else:
            raise KeyError("LR Scheduler %s not recognized!"%cfg.SOLVER.SCHEDULER)
    elif isinstance(optimizer, torch.optim.Adam):
        return None
