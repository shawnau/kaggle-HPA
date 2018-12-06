import sys
sys.path.append('../')

import os, argparse
import torch

from dl_backbone.config import cfg
from dl_backbone.data import make_data_loader
from dl_backbone.data.dataset.mertices import evaluation
from dl_backbone.solver import make_lr_scheduler
from dl_backbone.solver import make_optimizer
from dl_backbone.model.network import NetWrapper
from dl_backbone.engine.trainer import do_train
from dl_backbone.engine.inference import inference
from dl_backbone.utils.checkpoint import DetectronCheckpointer
from dl_backbone.utils.collect_env import collect_env_info
from dl_backbone.utils.comm import synchronize, get_rank
from dl_backbone.utils.logger import setup_logger


def train(cfg, local_rank, distributed):
    model = NetWrapper(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.deprecated.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    else:
        model = torch.nn.DataParallel(model)

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model


def validation(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps

    dataset_name = cfg.DATASETS.TEST
    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    inference(
        model,
        data_loader_val,
        device=cfg.MODEL.DEVICE,
        output_folder=output_folder,
    )
    synchronize()
    evaluation(cfg.DATASETS.TEST_LABEL, os.path.join(output_folder, "predictions.pth"))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.deprecated.init_process_group(
            backend="nccl", init_method="env://"
        )

    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("dl_backbone", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    # logger.info("Loaded configuration file {}".format(args.config_file))
    # with open(args.config_file, "r") as cf:
    #     config_str = "\n" + cf.read()
    #     logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        validation(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
