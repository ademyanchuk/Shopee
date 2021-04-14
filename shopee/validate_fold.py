import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from shopee.config import load_config_yaml

from .checkpoint_utils import resume_checkpoint
from .datasets import init_dataloaders, init_datasets
from .metric import row_wise_f1_score, validate_score
from .models import ArcFaceNet
from .paths import MODELS_PATH
from .train import validate_epoch


def validate_fold(
    exp_name: str, fold: int, Config: dict, df: pd.DataFrame, image_dir: Path,
) -> Tuple[float, pd.DataFrame]:
    """Config should have `tfidf_args` key"""
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
    # image predictions
    img_score, pred_df = validate_score(val_df, embeds, th)
    logging.info(f"Image model score: {img_score} [for exp: {exp_name}, fold: {fold}]")
    # text predictions
    text_score, text_df = validate_fold_text(val_df, Config["tfidf_args"])
    logging.info(f"Text model score: {text_score} [for exp: {exp_name}, fold: {fold}]")
    # finalize prediction data frame
    score, pred_df = finalize_df(pred_df, text_df)
    return score, pred_df


def validate_fold_text(df: pd.DataFrame, txt_model_args: dict):
    df = df.copy()
    model = TfidfVectorizer(**txt_model_args)
    text_embeds = model.fit_transform(df["title"]).toarray()
    text_embeds = torch.from_numpy(text_embeds)
    score, pred_df = validate_score(df, text_embeds, th=None)
    return score, pred_df


def finalize_df(img_df: pd.DataFrame, text_df: pd.DataFrame):
    # helper
    def combine_predictions(row):
        x = np.concatenate([row["pred_postings_img"], row["pred_postings_text"]])
        return np.unique(x).tolist()

    # rename img_df columns
    img_df = img_df.rename(
        columns={
            "best25_mean": "best25_mean_img",
            "f1": "f1_img",
            "pred_postings": "pred_postings_img",
        }
    )
    # add text columns
    img_df["best25_mean_text"] = text_df["best25_mean"]
    img_df["f1_text"] = text_df["f1"]
    img_df["pred_postings_text"] = text_df["pred_postings"]
    # combine predictions
    img_df["joined_pred"] = img_df.apply(combine_predictions, axis=1)
    # compute and add combined score
    score, f1mean = row_wise_f1_score(img_df["true_postings"], img_df["joined_pred"])
    img_df["f1_joined"] = score
    # change lists to strings before saving
    img_df["pred_postings_img"] = img_df["pred_postings_img"].apply(
        lambda x: " ".join(x)
    )
    img_df["pred_postings_text"] = img_df["pred_postings_text"].apply(
        lambda x: " ".join(x)
    )
    img_df["joined_pred"] = img_df["joined_pred"].apply(lambda x: " ".join(x))
    img_df["true_postings"] = img_df["true_postings"].apply(lambda x: " ".join(x))
    return f1mean, img_df


def validate_models_fold(
    exp_names: List[str], conf_dir: Path, fold: int, df: pd.DataFrame, image_dir: Path
) -> Tuple[float, pd.DataFrame]:
    train_df = df[df["fold"] != fold].copy().reset_index(drop=True)
    val_df = df[df["fold"] == fold].copy().reset_index(drop=True)

    embeds = []
    for exp_name in exp_names:
        Config = load_config_yaml(conf_dir, exp_name)
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
