from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from shopee.metric import compute_thres, get_sim_stats

from .checkpoint_utils import resume_checkpoint
from .datasets import init_test_dataset
from .models import ArcFaceNet

FOLD_NUM_CLASSES = {0: 11014, 1: 11014, 2: 11013, 3: 11014, 4: 11014}


def predict_one_model(
    exp_name: str,
    on_fold: int,
    df: pd.DataFrame,
    image_dir: Path,
    model_dir: Path,
    conf_dir: Path,
    threshold: Optional[float] = None,
):
    with open(conf_dir / f"{exp_name}_conf.yaml", "r") as f:
        Config = yaml.safe_load(f)
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
