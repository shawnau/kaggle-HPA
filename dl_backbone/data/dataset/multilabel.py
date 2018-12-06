import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, num_classes, label_file, root_folder, transforms):
        df_train = pd.read_csv(label_file)
        self.ids = df_train["Id"].tolist()
        raw_labels = df_train['Target'].tolist()
        self.labels = [list(map(int, item.split(' '))) for item in raw_labels]
        self.num_classes = num_classes
        self.root_folder = root_folder
        self.transforms = transforms

    def __getitem__(self, index):
        img_names = [self.ids[index] + "_" + color + ".png" for color in ["red", "green", "blue", "yellow"]]
        R, G, B, Y = (cv2.imread(os.path.join(self.root_folder, img_name), cv2.IMREAD_GRAYSCALE) for img_name in img_names)
        bgry = np.stack([B, G, R, Y], axis=-1)
        bgry = self.transforms(bgry)
        label_vec = torch.zeros(self.num_classes)
        labels = self.labels[index]
        label_vec[labels] = 1
        return bgry, label_vec, index

    def __len__(self):
        return len(self.ids)
