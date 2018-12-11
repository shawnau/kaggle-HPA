import sys
sys.path.append('../')

import os, argparse
import torch

from dl_backbone.config import cfg
from dl_backbone.data import make_data_loader
from dl_backbone.solver import make_lr_scheduler
from dl_backbone.solver import make_optimizer
from dl_backbone.model.network import NetWrapper
from dl_backbone.engine.trainer import do_train
from dl_backbone.utils.checkpoint import DetectronCheckpointer
from dl_backbone.utils.collect_env import collect_env_info
from dl_backbone.utils.logger import setup_logger


def train(is_valid=True):
    model = NetWrapper(cfg)
    loss_module = torch.nn.BCEWithLogitsLoss().cuda()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    model = torch.nn.DataParallel(model)

    arguments = {"iteration": 0}

    save_to_disk = True  # always true in one machine
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    train_data_loader = make_data_loader(
        cfg,
        cfg.DATASETS.TRAIN,
        is_train=True,
        start_iter=arguments["iteration"]
    )

    if is_valid:
        valid_data_loader = make_data_loader(
            cfg,
            cfg.DATASETS.VALID,
            is_train=False,
            start_iter=arguments["iteration"],
        )
    else:
        valid_data_loader = None

    do_train(
        model,
        loss_module,
        train_data_loader,
        optimizer,
        checkpointer,
        device,
        cfg.SOLVER.CHECKPOINT_PERIOD,
        arguments,
        scheduler=scheduler,
        valid_data_loader=valid_data_loader
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
    # logger.info("Running with config:\n{}".format(cfg))

    train()


if __name__ == "__main__":
    main()
