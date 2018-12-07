import os
import argparse
import torch

from dl_backbone.config import cfg
from dl_backbone.data import make_data_loader
from dl_backbone.engine.inference import inference
from dl_backbone.model.network import NetWrapper
from dl_backbone.utils.checkpoint import DetectronCheckpointer
from dl_backbone.utils.collect_env import collect_env_info
from dl_backbone.utils.comm import synchronize, get_rank
from dl_backbone.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Pytorch Inference")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.deprecated.init_process_group(
            backend="nccl", init_method="env://"
        )

    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("dl_backbone", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = NetWrapper(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    dataset_name = cfg.DATASETS.TEST
    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    inference(
        model,
        data_loader_val,
        device=cfg.MODEL.DEVICE,
        output_folder=output_folder
    )
    synchronize()


if __name__ == "__main__":
    main()