from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .img_augs import make_albu_augs


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
            return len(np.unique(self.labels))

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


def init_datasets(
    Config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, image_dir: Path
):
    train_aug = make_albu_augs(
        img_size=Config["img_size"], crop_size=Config["crop_size"], mode="train"
    )
    val_aug = make_albu_augs(
        img_size=Config["img_size"], crop_size=Config["crop_size"], mode="val"
    )
    train_ds = ShImageDataset(
        train_df,
        image_dir,
        image_id_col=Config["image_id_col"],
        target_col=Config["target_col"],
        is_test=False,
        transform=train_aug,
    )
    val_ds = ShImageDataset(
        val_df,
        image_dir,
        image_id_col=Config["image_id_col"],
        target_col=Config["target_col"],
        is_test=False,
        transform=val_aug,
    )
    return train_ds, val_ds


def init_test_dataset(Config: dict, df: pd.DataFrame, image_dir: Path):
    test_aug = make_albu_augs(
        img_size=Config["img_size"], crop_size=Config["crop_size"], mode="test"
    )
    return ShImageDataset(
        df=df,
        image_dir=image_dir,
        image_id_col=Config["image_id_col"],
        target_col="",
        is_test=True,
        transform=test_aug,
    )


def init_dataloaders(train_ds: ShImageDataset, val_ds: ShImageDataset, Config: dict):
    return {
        "train": DataLoader(
            train_ds,
            batch_size=Config["bs"],
            shuffle=True,
            num_workers=Config["num_workers"],
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=Config["bs"],
            shuffle=False,
            num_workers=Config["num_workers"],
            pin_memory=True,
        ),
    }
