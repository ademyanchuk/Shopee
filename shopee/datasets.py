from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ShImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        image_id_col: str,
        target_col: str,
        is_test: bool,
        transform=None,
    ):
        self.df = df
        self.image_dir = image_dir
        self.image_id_col = image_id_col
        self.is_test = is_test
        self.labels = None
        if not is_test:
            self.labels = df[target_col].values
        self.transform = transform

    def _get_img_path(self, idx: int) -> Path:
        image_id = self.df.loc[idx, self.image_id_col]
        file_path = self.image_dir / f"{image_id}"
        return file_path

    @property
    def num_classes(self):
        if self.labels is not None:
            return self.labels.nunique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        data = {}

        file_path = self._get_img_path(idx)

        # here input will be greyscale
        image = Image.open(file_path).convert("RGB")

        if self.transform:
            image = np.array(image)  # albumentations need numpy
            image = self.transform(image=image)["image"]

        data["image"] = image
        if self.is_test:
            return data

        # train mode
        label = torch.tensor(self.labels[idx]).long()
        data["label"] = label

        return data
