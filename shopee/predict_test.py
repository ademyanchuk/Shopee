from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from shopee.config import load_config_yaml
from shopee.metric import QUANTILES, compute_thres, get_sim_stats, get_sim_stats_torch

from .checkpoint_utils import resume_checkpoint
from .datasets import init_test_dataset
from .models import init_model

FOLD_NUM_CLASSES = {0: 11014, 1: 11014, 2: 11013, 3: 11014, 4: 11014}


def get_image_embeds(
    conf_dir: Path,
    exp_names: Union[str, List[str]],
    df: pd.DataFrame,
    image_dir: Path,
    model_dir: Path,
    on_fold: int,
):
    if isinstance(exp_names, str):
        exp_names = [exp_names]
    all_embeds = []
    for exp_name in exp_names:
        Config = load_config_yaml(conf_dir, exp_name)
        # check if config (maybe used to train past models) has arc face text key
        try:
            use_text = Config["arc_face_text"]
            bert_name = Config["bert_name"]
            kaggle_bert_dir = Path("/kaggle/input/sbert-hf") / bert_name.split("/")[-1]
        except KeyError:
            print("Old models: set text input to False")
            use_text = False
            kaggle_bert_dir = False
        test_ds = init_test_dataset(
            Config, df, image_dir, kaggle_bert_dir, use_text=use_text
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=Config["bs"],
            shuffle=False,
            num_workers=Config["num_workers"],
            pin_memory=True,
        )
        num_classes = FOLD_NUM_CLASSES[on_fold]

        model = init_model(num_classes, Config, pretrained=kaggle_bert_dir)
        model.cuda()
        if Config["channels_last"]:
            model = model.to(memory_format=torch.channels_last)
        checkpoint_path = model_dir / f"{exp_name}_f{on_fold}_score.pth"
        epoch, _, _, _ = resume_checkpoint(model, checkpoint_path)
        assert isinstance(epoch, int)
        _, img_embeds = test_epoch(
            model, test_dl, epoch, Config, use_amp=True, use_text=use_text
        )
        all_embeds.append(img_embeds)
    all_embeds = torch.cat(all_embeds, dim=1)  # concatenate
    return all_embeds


def compute_matches(
    emb_tensor: torch.Tensor,
    df: pd.DataFrame,
    chunk_sz: int,
    static_th: Optional[float] = None,
    coeff: torch.Tensor = torch.tensor([0.9, 0.9, 0.9]),
) -> List[List[str]]:
    # normalize
    emb_tensor = F.normalize(emb_tensor)
    matches = []
    num_chunks = (len(emb_tensor) // chunk_sz) + 1

    for i in range(num_chunks):
        a = i * chunk_sz
        b = (i + 1) * chunk_sz
        b = min(b, len(emb_tensor))
        print(f"compute similarities for chunks {a} to {b}")
        sim = emb_tensor[a:b] @ emb_tensor.T
        stats = get_sim_stats_torch(sim)
        quants = torch.quantile(stats, q=torch.tensor(QUANTILES))
        if static_th is not None:
            static_th = torch.tensor(static_th)

        threshold = torch.stack(
            [compute_thres(x, quants, static_th, coeff) for x in stats]
        )
        threshold = threshold[:, None].cuda()
        selection = (sim > threshold).cpu().numpy()
        best_2 = torch.argsort(sim)[-2:].cpu().numpy()
        for sel, b2 in zip(selection, best_2):
            row = sel
            if len(row) < 2:
                row = b2
            matches.append(df.iloc[row].posting_id.tolist())
    return matches


def combine_predictions(row):
    x = np.concatenate([row["img_matches"], row["text_matches"]])
    return " ".join(np.unique(x))


def predict_img_text(
    exp_name: Union[str, List[str]],
    on_fold: int,
    df: pd.DataFrame,
    image_dir: Path,
    model_dir: Path,
    conf_dir: Path,
    text_model_args: dict,
    static_ths: tuple = (None, None),
):
    img_embeds = get_image_embeds(
        conf_dir, exp_name[:-1], df, image_dir, model_dir, on_fold
    )

    img_matches = compute_matches(
        img_embeds,
        df,
        chunk_sz=1024,
        static_th=static_ths[0],
        coeff=torch.tensor([0.9, 0.85, 0.8]),
    )

    # texts
    model_txt = TfidfVectorizer(**text_model_args)
    text_embeds = model_txt.fit_transform(df["title"]).toarray().astype(np.float32)
    text_embeds = torch.from_numpy(text_embeds).cuda()

    bert_embeds = get_image_embeds(
        conf_dir, exp_name[-1], df, image_dir, model_dir, on_fold
    )
    text_embeds = torch.cat([text_embeds, bert_embeds], dim=1)

    text_matches = compute_matches(
        text_embeds,
        df,
        chunk_sz=1024,
        static_th=static_ths[1],
        coeff=torch.tensor([0.9, 0.85, 0.8]),
    )

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
    quants = np.quantile(np.concatenate(stats), q=QUANTILES)
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
    quants = np.quantile(np.concatenate(stats), q=QUANTILES)
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


def test_epoch(model, dataloader, epoch, Config, use_amp, use_text=False):
    model.eval()
    epoch_logits = []
    bar = tqdm(dataloader)
    test_batch_fn = test_batch
    if use_text:
        test_batch_fn = test_batch_bert
    for batch in bar:
        bar.set_description(f"Epoch {epoch} [validation]".ljust(20))
        batch_logits = test_batch_fn(batch, model, Config, use_amp)
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


def test_batch_bert(
    batch, model, Config, use_amp,
):
    """
    It returns also detached to cpu output tensor
    and targets tensor
    """
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    if Config["channels_last"]:
        raise NotImplementedError
    with torch.no_grad():
        with autocast(enabled=use_amp):
            outputs = model(input_ids, attention_mask)
    # index into only main task classes (if aux task is used)
    return outputs
