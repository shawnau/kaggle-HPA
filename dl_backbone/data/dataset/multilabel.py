import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ProteinDataset(Dataset):
    def __init__(self, train_root, transforms=None):
        pd_data = pd.read_csv("train.csv")
        ids = pd_data["Id"].tolist()
        raw_labels = pd_data['Target'].tolist()
        labels = [list(map(int, item.split(' '))) for item in raw_labels]
        self.data = list(zip(ids, labels))
        self.train_root = train_root
        self.transforms = transforms

    def __getitem__(self, index):
        img_names = [self.data[index][0] + "_" + color + ".png" for color in ["blue", "red", "yellow", "green"]]
        img = [Image.open(os.path.join(self.train_root, img_name)) for img_name in img_names]
        img = np.stack(img, axis=-1)
        if self.transforms is not None:
            img = self.transforms(img)
        labels = self.data[index][1]
        return img, np.array(labels)

    def __len__(self):
        return len(self.data)


