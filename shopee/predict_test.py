from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from .checkpoint_utils import resume_checkpoint
from .datasets import init_test_dataset
from .metric import make_submit_df
from .models import ArcFaceNet
from .paths import LOGS_PATH
from .train import validate_epoch

FOLD_NUM_CLASSES = {0: 11014, 1: 11014, 2: 11013, 3: 11014, 4: 11014}


def predict_one_model(
    exp_name: str, on_fold: int, df: pd.DataFrame, image_dir: Path, model_dir: Path
):
    with open(LOGS_PATH / f"{exp_name}_conf.yaml", "r") as f:
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
    _, embeds, _ = validate_epoch(
        model, test_dl, epoch, Config, use_amp=True
    )
    submit_df = make_submit_df(df, embeds, th)
    return submit_df
