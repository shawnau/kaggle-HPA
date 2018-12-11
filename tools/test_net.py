import sys
sys.path.append('../')

import os
import argparse
import torch

from dl_backbone.config import cfg
from dl_backbone.data import make_data_loader
from dl_backbone.engine.inference import inference
from dl_backbone.model.network import NetWrapper
from dl_backbone.utils.checkpoint import DetectronCheckpointer
from dl_backbone.utils.collect_env import collect_env_info
from dl_backbone.utils.logger import setup_logger


def main():
    """
    :param mode: ["train valid test"]
    :return:
    """
    parser = argparse.ArgumentParser(description="Pytorch Inference")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--mode",
        default="valid",
        help="train, valid or test",
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("model", save_dir)
    logger.info("Using {} GPUs".format(torch.cuda.device_count()))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = NetWrapper(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model, logger=logger)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    if args.mode == "train":
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TRAIN)
        os.makedirs(output_folder, exist_ok=True)
        data_loader_test = make_data_loader(cfg, cfg.DATASETS.TRAIN, is_train=False)
    elif args.mode == "valid":
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.VALID)
        os.makedirs(output_folder, exist_ok=True)
        data_loader_test = make_data_loader(cfg, cfg.DATASETS.VALID, is_train=False)
    elif args.mode == "test":
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST)
        os.makedirs(output_folder, exist_ok=True)
        data_loader_test = make_data_loader(cfg, cfg.DATASETS.TEST, is_train=False)
    else:
        raise KeyError("mode not recognized")

    inference(
        model,
        data_loader_test,
        device=cfg.MODEL.DEVICE,
        output_folder=output_folder
    )


if __name__ == "__main__":
    main()