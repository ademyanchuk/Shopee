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
from .models import init_model
from .paths import MODELS_PATH
from .train import validate_epoch


def validate_models_union(
    exp_names: List[str],
    conf_dir: Path,
    fold: int,
    df: pd.DataFrame,
    image_dir: Path,
    tfidf_args: dict,
) -> Tuple[float, pd.DataFrame]:
    """Validate several models and join their predictoins as a set union
       Add tfidf predictions as well"""
    train_df = df[df["fold"] != fold].copy().reset_index(drop=True)
    val_df = df[df["fold"] == fold].copy().reset_index(drop=True)
    pred_dfs = []
    for exp_name in exp_names:
        embedding = extract_embeedings(
            exp_name, conf_dir, image_dir, train_df, val_df, fold
        )
        score, pred_df = validate_score(val_df, embedding, th=None)
        print(f"DL model score: {score} [for exp: {exp_names}, fold: {fold}]")
        pred_dfs.append(pred_df)

    tfidf_score, text_df = validate_fold_text(val_df, tfidf_args)
    print(f"Tfidf model score: {tfidf_score} [for exp: {exp_names}, fold: {fold}]")
    pred_dfs.append(text_df)
    # merge here and return final score and df
    return finalize_df_v1(pred_dfs)


def validate_fold(
    exp_name: str, fold: int, Config: dict, df: pd.DataFrame, image_dir: Path,
) -> Tuple[float, pd.DataFrame]:
    """Config should have `tfidf_args` key"""
    train_df = df[df["fold"] != fold].copy().reset_index(drop=True)
    val_df = df[df["fold"] == fold].copy().reset_index(drop=True)

    # check if config (maybe used to train past models) has arc face text key
    try:
        use_text = Config["arc_face_text"]
        bert_name = Config["bert_name"]
    except KeyError:
        print("Old models: set text input to False")
        use_text = False
        bert_name = ""

    train_ds, val_ds = init_datasets(
        Config,
        train_df,
        val_df,
        image_dir,
        txt_mod_name_or_path=bert_name,
        use_text=use_text,
    )
    dataloaders = init_dataloaders(train_ds, val_ds, Config)
    num_classes = int(train_df[Config["target_col"]].max() + 1)
    model = init_model(num_classes, Config, pretrained=False)
    model.cuda()
    if Config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    checkpoint_path = MODELS_PATH / f"{exp_name}_f{fold}_score.pth"
    epoch, _, _, th = resume_checkpoint(model, checkpoint_path)
    assert isinstance(epoch, int)
    assert isinstance(th, float)
    _, embeds, _ = validate_epoch(
        model, dataloaders["val"], epoch, Config, use_amp=True, is_bert=use_text,
    )
    # image predictions
    img_score, pred_df = validate_score(val_df, embeds, th)
    logging.info(f"DL model score: {img_score} [for exp: {exp_name}, fold: {fold}]")
    # text predictions
    text_score, text_df = validate_fold_text(val_df, Config["tfidf_args"])
    logging.info(f"Tfidf model score: {text_score} [for exp: {exp_name}, fold: {fold}]")
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
    exp_names: List[str],
    conf_dir: Path,
    fold: int,
    df: pd.DataFrame,
    image_dir: Path,
    tfidf_args: dict,
) -> Tuple[float, pd.DataFrame]:
    train_df = df[df["fold"] != fold].copy().reset_index(drop=True)
    val_df = df[df["fold"] == fold].copy().reset_index(drop=True)

    embeds = []
    for exp_name in exp_names:
        embed = extract_embeedings(
            exp_name, conf_dir, image_dir, train_df, val_df, fold
        )
        embeds.append(embed)

    embeds = torch.cat(embeds, dim=1)  # concat embeedings from models
    # image predictions
    img_score, pred_df = validate_score(val_df, embeds, th=None)
    print(f"DL model score: {img_score} [for exp: {exp_names}, fold: {fold}]")
    # text predictions
    text_score, text_df = validate_fold_text(val_df, tfidf_args)
    print(f"Tfidf model score: {text_score} [for exp: {exp_names}, fold: {fold}]")
    # finalize prediction data frame
    score, pred_df = finalize_df(pred_df, text_df)
    return score, pred_df


def extract_embeedings(
    exp_name: str,
    conf_dir: Path,
    image_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fold: int,
):
    Config = load_config_yaml(conf_dir, exp_name)
    # check if config (maybe used to train past models) has arc face text key
    try:
        use_text = Config["arc_face_text"]
        bert_name = Config["bert_name"]
    except KeyError:
        print("Old models: set text input to False")
        use_text = False
        bert_name = ""
    train_ds, val_ds = init_datasets(
        Config,
        train_df,
        val_df,
        image_dir,
        txt_mod_name_or_path=bert_name,
        use_text=use_text,
    )
    dataloaders = init_dataloaders(train_ds, val_ds, Config)
    num_classes = int(train_df[Config["target_col"]].max() + 1)

    model = init_model(num_classes, Config, pretrained=False)
    model.cuda()
    if Config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    checkpoint_path = MODELS_PATH / f"{exp_name}_f{fold}_score.pth"
    epoch, _, _, _ = resume_checkpoint(model, checkpoint_path)
    assert isinstance(epoch, int)
    _, embed, _ = validate_epoch(
        model, dataloaders["val"], epoch, Config, use_amp=True, is_bert=use_text
    )
    return embed


def finalize_df_v1(dfs: List[pd.DataFrame]) -> Tuple[float, pd.DataFrame]:
    # helper
    def combine_predictions(row, cols_to_combine):
        x = np.concatenate([row[c] for c in cols_to_combine])
        return np.unique(x).tolist()

    assert len(dfs) >= 2
    final_df = dfs[0]
    unique_cols = ["best25_mean", "pred_postings", "true_postings", "f1"]
    same_cols = [c for c in final_df.columns if c not in unique_cols]
    for df in dfs[1:]:
        final_df = pd.merge(final_df, df, on=same_cols)

    # combine predictions
    cols_to_combine = [c for c in final_df.columns if c.startswith("pred_postings")]
    final_df["joined_pred"] = final_df.apply(
        lambda x: combine_predictions(x, cols_to_combine), axis=1
    )
    # compute and add combined score
    score, f1mean = row_wise_f1_score(
        final_df["true_postings_x"], final_df["joined_pred"]
    )
    final_df["f1_joined"] = score
    # change lists to strings before saving
    for col in (
        cols_to_combine
        + [c for c in final_df.columns if c.startswith("true_postings")]
        + ["joined_pred"]
    ):
        final_df[col] = final_df[col].apply(lambda x: " ".join(x))

    return f1mean, final_df
