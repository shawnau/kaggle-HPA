import os
from dl_backbone.config import cfg
from dl_backbone.data.dataset.mertices import evaluation


def main():
    dataset_name = cfg.DATASETS.TEST
    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
    evaluation(cfg.DATASETS.TEST_LABEL,
               os.path.join(output_folder, "predictions_valid.pth"),
               thresholds=[0.5, 0.4, 0.3, 0.2, 0.1])


if __name__ == "__main__":
    main()