from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from shopee.config import load_config_yaml
from shopee.metric import compute_thres, emb_sim, get_sim_stats, get_sim_stats_torch

from .checkpoint_utils import resume_checkpoint
from .datasets import init_test_dataset
from .models import ArcFaceNet

FOLD_NUM_CLASSES = {0: 11014, 1: 11014, 2: 11013, 3: 11014, 4: 11014}


def get_image_embeds(
    conf_dir: Path,
    exp_name: str,
    df: pd.DataFrame,
    image_dir: Path,
    model_dir: Path,
    on_fold: int,
):
    Config = load_config_yaml(conf_dir, exp_name)
    test_ds = init_test_dataset(Config, df, image_dir)
    test_dl = DataLoader(
        test_ds,
        batch_size=Config["bs"],
        shuffle=False,
        num_workers=Config["num_workers"],
        pin_memory=True,
    )
    num_classes = FOLD_NUM_CLASSES[on_fold]

    model = ArcFaceNet(num_classes, Config, pretrained=False)
    model.cuda()
    if Config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    checkpoint_path = model_dir / f"{exp_name}_f{on_fold}_score.pth"
    epoch, _, _, _ = resume_checkpoint(model, checkpoint_path)
    assert isinstance(epoch, int)
    _, img_embeds = test_epoch(model, test_dl, epoch, Config, use_amp=True)
    return img_embeds


def compute_matches(
    emb_tensor: torch.Tensor, df: pd.DataFrame, chunk_sz: int
) -> List[List[str]]:
    matches = []
    num_chunks = (len(emb_tensor) // chunk_sz) + 1

    for i in range(num_chunks):
        a = i * chunk_sz
        b = (i + 1) * chunk_sz
        b = min(b, len(emb_tensor))
        print(f"compute similarities for chunks {a} to {b}")
        sim = emb_tensor[a:b] @ emb_tensor.T
        stats = get_sim_stats_torch(sim)
        quants = torch.quantile(stats, q=torch.tensor([0.3, 0.6, 0.9]))
        threshold = torch.stack([compute_thres(x, quants) for x in stats])
        threshold = threshold[:, None].cuda()
        selection = (sim > threshold).cpu().numpy()
        for row in selection:
            matches.append(df.iloc[row].posting_id.tolist())
    return matches


def combine_predictions(row):
    x = np.concatenate([row["img_matches"], row["text_matches"]])
    return " ".join(np.unique(x))


def predict_img_text(
    exp_name: str,
    on_fold: int,
    df: pd.DataFrame,
    image_dir: Path,
    model_dir: Path,
    conf_dir: Path,
    text_model_args: dict,
):
    img_embeds = get_image_embeds(conf_dir, exp_name, df, image_dir, model_dir, on_fold)

    img_matches = compute_matches(img_embeds, df, chunk_sz=1024)

    # texts
    model_txt = TfidfVectorizer(**text_model_args)
    text_embeds = model_txt.fit_transform(df["title"]).toarray().astype(np.float32)
    text_embeds = torch.from_numpy(text_embeds).cuda()
    text_matches = compute_matches(text_embeds, df, chunk_sz=1024)

    tmp_df = pd.DataFrame({"img_matches": img_matches, "text_matches": text_matches})

    df["matches"] = tmp_df.apply(combine_predictions, axis=1)
    return df


def predict_text(
    df: pd.DataFrame, text_model_args: dict,
):
    model_txt = TfidfVectorizer(**text_model_args)
    text_embeds = model_txt.fit_transform(df["title"]).toarray().astype(np.float32)
    text_embeds = torch.from_numpy(text_embeds).cuda()
    text_matches = compute_matches(text_embeds, df, chunk_sz=1024)

    df["matches"] = text_matches
    df["matches"] = df["matches"].apply(lambda x: " ".join(x))
    return df


def predict_one_model(
    exp_name: str,
    on_fold: int,
    df: pd.DataFrame,
    image_dir: Path,
    model_dir: Path,
    conf_dir: Path,
    threshold: Optional[float] = None,
):
    Config = load_config_yaml(conf_dir, exp_name)
    test_ds = init_test_dataset(Config, df, image_dir)
    test_dl = DataLoader(
        test_ds,
        batch_size=Config["bs"],
        shuffle=False,
        num_workers=Config["num_workers"],
        pin_memory=True,
    )
    num_classes = FOLD_NUM_CLASSES[on_fold]

    model = ArcFaceNet(num_classes, Config, pretrained=False)
    model.cuda()
    if Config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    checkpoint_path = model_dir / f"{exp_name}_f{on_fold}_score.pth"
    epoch, _, _, th = resume_checkpoint(model, checkpoint_path)
    assert isinstance(epoch, int)
    assert isinstance(th, float)
    emb_list, emb_tensor = test_epoch(model, test_dl, epoch, Config, use_amp=True)
    # matching
    if threshold is None:
        threshold = th
    stats = []
    for batch in tqdm(emb_list):
        selection = (batch @ emb_tensor.T).cpu().numpy()
        batch_stats = get_sim_stats(selection)
        stats.append(batch_stats)
    quants = np.quantile(np.concatenate(stats), q=[0.3, 0.6, 0.9])
    matches = []
    for batch, stat in zip(emb_list, stats):
        threshold = pd.Series(stat).apply(lambda x: compute_thres(x, quants))
        threshold = threshold.values[:, None]
        batch_sims = (batch @ emb_tensor.T).cpu().numpy()
        selection = batch_sims > threshold
        for row in selection:
            matches.append(" ".join(df.iloc[row].posting_id.tolist()))
    df["matches"] = matches
    return df


def predict_2_models(
    exp_names: str,
    on_fold: int,
    df: pd.DataFrame,
    image_dir: Path,
    model_dir: Path,
    conf_dir: Path,
    threshold: Optional[float] = None,
):
    emb_lists, emb_tensors = [], []
    for exp_name in exp_names:
        Config = load_config_yaml(conf_dir, exp_name)
        test_ds = init_test_dataset(Config, df, image_dir)
        test_dl = DataLoader(
            test_ds,
            batch_size=Config["bs"],
            shuffle=False,
            num_workers=Config["num_workers"],
            pin_memory=True,
        )
        num_classes = FOLD_NUM_CLASSES[on_fold]

        model = ArcFaceNet(num_classes, Config, pretrained=False)
        model.cuda()
        if Config["channels_last"]:
            model = model.to(memory_format=torch.channels_last)
        checkpoint_path = model_dir / f"{exp_name}_f{on_fold}_score.pth"
        epoch, _, _, _ = resume_checkpoint(model, checkpoint_path)
        assert isinstance(epoch, int)
        emb_list, emb_tensor = test_epoch(model, test_dl, epoch, Config, use_amp=True)
        emb_lists.append(emb_list)
        emb_tensors.append(emb_tensor)
    emb_lists = [
        torch.cat([b1, b2], dim=1) for b1, b2 in zip(emb_lists[0], emb_lists[1])
    ]
    emb_tensors = torch.cat(emb_tensors, dim=1)
    # matching
    stats = []
    for batch in tqdm(emb_lists):
        selection = (batch @ emb_tensors.T).cpu().numpy()
        batch_stats = get_sim_stats(selection)
        stats.append(batch_stats)
    quants = np.quantile(np.concatenate(stats), q=[0.3, 0.6, 0.9])
    matches = []
    for batch, stat in zip(emb_lists, stats):
        threshold = pd.Series(stat).apply(lambda x: compute_thres(x, quants))
        threshold = threshold.values[:, None]
        batch_sims = (batch @ emb_tensors.T).cpu().numpy()
        selection = batch_sims > threshold
        for row in selection:
            matches.append(" ".join(df.iloc[row].posting_id.tolist()))
    df["matches"] = matches
    return df


def test_epoch(model, dataloader, epoch, Config, use_amp):
    model.eval()
    epoch_logits = []
    bar = tqdm(dataloader)
    for batch in bar:
        bar.set_description(f"Epoch {epoch} [validation]".ljust(20))
        batch_logits = test_batch(batch, model, Config, use_amp)
        epoch_logits.append(batch_logits)
    # on epoch end
    return epoch_logits, torch.cat(epoch_logits)


def test_batch(
    batch, model, Config, use_amp,
):
    """
    It returns also detached to cpu output tensor
    and targets tensor
    """
    inputs = batch["image"].cuda()
    if Config["channels_last"]:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        with autocast(enabled=use_amp):
            outputs = model(inputs)
    return outputs
