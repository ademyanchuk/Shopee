from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from .checkpoint_utils import resume_checkpoint
from .datasets import init_dataloaders, init_datasets
from .metric import validate_score
from .models import ArcFaceNet
from .paths import MODELS_PATH
from .train import validate_epoch


def validate_fold(
    exp_name: str, fold: int, Config: dict, df: pd.DataFrame, image_dir: Path
) -> Tuple[float, pd.DataFrame]:
    train_df = df[df["fold"] != fold].copy().reset_index(drop=True)
    val_df = df[df["fold"] == fold].copy().reset_index(drop=True)
    train_ds, val_ds = init_datasets(Config, train_df, val_df, image_dir)
    dataloaders = init_dataloaders(train_ds, val_ds, Config)
    num_classes = int(train_df[Config["target_col"]].max() + 1)

    model = ArcFaceNet(num_classes, Config, pretrained=False)
    model.cuda()
    if Config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    checkpoint_path = MODELS_PATH / f"{exp_name}_f{fold}_score.pth"
    epoch, _, _, th = resume_checkpoint(model, checkpoint_path)
    assert isinstance(epoch, int)
    assert isinstance(th, float)
    _, embeds, _ = validate_epoch(
        model, dataloaders["val"], epoch, Config, use_amp=True
    )
    score, pred_df = validate_score(val_df, embeds, th)
    return score, pred_df


def validate_fold_text(df: pd.DataFrame, fold: int, Config: dict):
    train_df = df[df["fold"] != fold].copy().reset_index(drop=True)
    val_df = df[df["fold"] == fold].copy().reset_index(drop=True)
    model = TfidfVectorizer(**Config["tfidf_args"])
    model.fit(train_df["title"])
    text_embeds = model.transform(val_df["title"]).toarray()
    score, pred_df = validate_score(val_df, text_embeds, th=None, chunk_sz=256)
    return score, pred_df


def validate_models_fold(
    exp_names: List[str], conf_dir: Path, fold: int, df: pd.DataFrame, image_dir: Path
) -> Tuple[float, pd.DataFrame]:
    train_df = df[df["fold"] != fold].copy().reset_index(drop=True)
    val_df = df[df["fold"] == fold].copy().reset_index(drop=True)

    embeds = []
    for exp_name in exp_names:
        with open(conf_dir / f"{exp_name}_conf.yaml", "r") as f:
            Config = yaml.safe_load(f)
        train_ds, val_ds = init_datasets(Config, train_df, val_df, image_dir)
        dataloaders = init_dataloaders(train_ds, val_ds, Config)
        num_classes = int(train_df[Config["target_col"]].max() + 1)

        model = ArcFaceNet(num_classes, Config, pretrained=False)
        model.cuda()
        if Config["channels_last"]:
            model = model.to(memory_format=torch.channels_last)
        checkpoint_path = MODELS_PATH / f"{exp_name}_f{fold}_score.pth"
        epoch, _, _, _ = resume_checkpoint(model, checkpoint_path)
        assert isinstance(epoch, int)
        _, embed, _ = validate_epoch(
            model, dataloaders["val"], epoch, Config, use_amp=True
        )
        embeds.append(embed)

    embeds = torch.cat(embeds, dim=1)
    score, pred_df = validate_score(val_df, embeds, th=None)
    return score, pred_df
