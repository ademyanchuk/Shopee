from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from shopee.rand_aug import make_aug

from .config import Config
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
        text: bool = False,
    ):
        self.df = df
        self.image_dir = image_dir
        self.image_id_col = image_id_col
        self.is_test = is_test
        self.labels = None
        if not is_test:
            self.labels = df[target_col].values
        self.transform = transform

        self.text = text
        if self.text:
            model_txt = TfidfVectorizer(**Config["tfidf_args"])
            self.text_embeds = (
                model_txt.fit_transform(df["title"]).toarray().astype(np.float32)
            )

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
        if self.text:
            data["text"] = torch.from_numpy(self.text_embeds[idx])
        if self.is_test:
            return data

        # train mode
        label = torch.tensor(self.labels[idx]).long()
        data["label"] = label

        return data


class ShTextDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        is_test: bool,
        tok_name_or_path: Union[str, Path],
    ):
        self.df = df
        self.is_test = is_test
        self.labels = None
        if not is_test:
            self.labels = df[target_col].values
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name_or_path)

    @property
    def num_classes(self):
        if self.labels is not None:
            return len(np.unique(self.labels))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        data = {}
        text = self.df.loc[idx, "title"]
        text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        data["input_ids"] = text["input_ids"][0]
        data["attention_mask"] = text["attention_mask"][0]
        if self.is_test:
            return data

        # train mode
        label = torch.tensor(self.labels[idx]).long()
        data["label"] = label

        return data


class ShMocoDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, image_dir: Path, image_id_col: str, transform=None,
    ):
        self.df = df
        self.image_dir = image_dir
        self.image_id_col = image_id_col
        self.transform = transform
        assert self.transform is not None

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

        image = np.array(image)  # albumentations need numpy
        image1 = self.transform(image=image)["image"]
        image2 = self.transform(image=image)["image"]

        data["image1"] = image1
        data["image2"] = image2
        return data


def init_augs(Config: dict):
    aug_type = Config["aug_type"]
    if aug_type == "albu":
        train_aug = make_albu_augs(
            img_size=Config["img_size"], crop_size=Config["crop_size"], mode="train"
        )
        val_aug = make_albu_augs(
            img_size=Config["img_size"], crop_size=Config["crop_size"], mode="val"
        )
    elif aug_type == "rand":
        train_aug = make_aug(
            img_size=Config["img_size"],
            crop_size=Config["crop_size"],
            starategy="rand",
            mode="train",
            severity=Config["rand_aug_severity"],
            width=Config["rand_aug_width"],
        )
        val_aug = make_aug(
            img_size=Config["img_size"],
            crop_size=Config["crop_size"],
            starategy="rand",
            mode="val",
        )
    else:
        raise NotImplementedError
    return train_aug, val_aug


def init_datasets(
    Config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: Path,
    txt_mod_name_or_path: Union[str, Path],
    use_text: bool = False,
):
    if use_text:
        # use bert dataset
        train_ds = ShTextDataset(
            df=train_df,
            target_col=Config["target_col"],
            is_test=False,
            tok_name_or_path=txt_mod_name_or_path,
        )
        val_ds = ShTextDataset(
            df=val_df,
            target_col=Config["target_col"],
            is_test=False,
            tok_name_or_path=txt_mod_name_or_path,
        )
    else:
        train_aug, val_aug = init_augs(Config)
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


def init_test_dataset(
    Config: dict,
    df: pd.DataFrame,
    image_dir: Path,
    txt_mod_name_or_path: Union[bool, str, Path],
    use_text: bool = False,
):
    if use_text:
        # use bert dataset
        return ShTextDataset(
            df=df, target_col="", is_test=True, tok_name_or_path=txt_mod_name_or_path,
        )
    else:
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


def init_dataloaders(
    train_ds: ShImageDataset,
    val_ds: ShImageDataset,
    Config: dict,
    is_moco: bool = False,
):
    return {
        "train": DataLoader(
            train_ds,
            batch_size=Config["bs"],
            shuffle=True,
            num_workers=Config["num_workers"],
            pin_memory=True,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
            drop_last=is_moco,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=Config["bs"],
            shuffle=False,
            num_workers=Config["num_workers"],
            pin_memory=True,
            drop_last=is_moco,
        ),
    }
