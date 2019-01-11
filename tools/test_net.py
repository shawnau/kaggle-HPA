import sys
sys.path.append('../')

import os
import argparse
import torch

from dl_backbone.config import cfg
from dl_backbone.data import make_data_loader, make_tta_data_loaders
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
    _ = checkpointer.load(os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.WEIGHT))

    for dataset_name in [cfg.DATASETS.TRAIN, cfg.DATASETS.VALID, cfg.DATASETS.TEST]:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        data_loader_test = make_data_loader(cfg, dataset_name, is_train=False)

        inference(
            model,
            data_loader_test,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder
        )

    if cfg.TEST.TTA == 'on':
        print('TTA start')
        tta_output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST, 'tta')
        os.makedirs(tta_output_folder, exist_ok=True)
        tta_data_loaders = make_tta_data_loaders(cfg)
        for idx, tta_data_loader in enumerate(tta_data_loaders):
            inference(
                model,
                tta_data_loader,
                device=cfg.MODEL.DEVICE,
                output_folder=tta_output_folder,
                output_name="tta_%d.pth" % idx
            )
        print("merging tta results")
        pths = [os.path.join(tta_output_folder, "tta_%d.pth" % idx) for idx in range(len(tta_data_loaders))]
        tta_tensors = [torch.stack(torch.load(pth), dim=0) for pth in pths]
        ensembled = sum(tta_tensors) / float(len(tta_tensors))
        pred_list = [ensembled[i] for i in range(len(ensembled))]
        tta_dump_path = os.path.join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST, 'predictions_tta.pth')
        torch.save(pred_list, tta_dump_path)
        print("model dumped into %s"%tta_dump_path)


if __name__ == "__main__":
    main()