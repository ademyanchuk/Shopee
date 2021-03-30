from pathlib import Path
from typing import Tuple

import pandas as pd
import torch

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
