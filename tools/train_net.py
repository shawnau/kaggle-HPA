import sys
sys.path.append('../')

import os, argparse, logging
import torch

from dl_backbone.config import cfg
from dl_backbone.data import make_data_loader
from dl_backbone.solver import make_optimizer, make_finetune_optimizer, make_lr_scheduler
from dl_backbone.model.network import NetWrapper
from dl_backbone.model.loss import make_loss_module
from dl_backbone.engine.trainer import do_train
from dl_backbone.utils.checkpoint import DetectronCheckpointer
from dl_backbone.utils.collect_env import collect_env_info
from dl_backbone.utils.logger import setup_logger


def train(cfg):
    logger = logging.getLogger("model.trainer")
    device = torch.device(cfg.MODEL.DEVICE)
    # define model and loss
    loss_module = make_loss_module(cfg)
    loss_module.to(device)
    model = NetWrapper(cfg)
    model.to(device)
    model = torch.nn.DataParallel(model)
    # define optimizer and scheduler
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    # load checkpoint
    arguments = {"epoch": 0}
    save_to_disk = True  # always true in one machine
    checkpointer = DetectronCheckpointer(
        cfg,
        model,
        optimizer,
        scheduler,
        cfg.OUTPUT_DIR,
        save_to_disk,
        logger
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)
    # make data loader
    train_data_loader = make_data_loader(
        cfg,
        cfg.DATASETS.TRAIN,
        is_train=True
    )
    valid_data_loader = make_data_loader(
        cfg,
        cfg.DATASETS.VALID,
        is_train=False
    )

    if cfg.SOLVER.FINETUNE == "on" and arguments["epoch"] == 0:
        finetune_optimizer = make_finetune_optimizer(cfg, model)
        finetune_data_loader = make_data_loader(
            cfg,
            cfg.DATASETS.TRAIN,
            is_train=True
        )
        do_train(
            model=model,
            loss_module=loss_module,
            train_data_loader=finetune_data_loader,
            valid_data_loader=valid_data_loader,
            optimizer=finetune_optimizer,
            scheduler=None,
            checkpointer=checkpointer,
            device=device,
            train_epoch=cfg.SOLVER.FINETUNE_EPOCH,
            checkpoint_period=cfg.SOLVER.CHECKPOINT_PERIOD,
            is_mixup=cfg.SOLVER.MIXUP,
            arguments=arguments
        )

    do_train(
        model=model,
        loss_module=loss_module,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpointer=checkpointer,
        device=device,
        train_epoch=cfg.SOLVER.TRAIN_EPOCH,
        checkpoint_period=cfg.SOLVER.CHECKPOINT_PERIOD,
        is_mixup=cfg.SOLVER.MIXUP,
        arguments=arguments
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Deep Learning Backbone")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("model", output_dir)
    logger.info("Using {} GPUs".format(torch.cuda.device_count()))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg)


if __name__ == "__main__":
    main()
