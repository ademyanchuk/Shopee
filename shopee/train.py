import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from timm.models import load_checkpoint
from timm.utils import ModelEmaV2
from torch import nn, optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from shopee.amp_scaler import NativeScaler
from shopee.checkpoint_utils import resume_checkpoint, save_checkpoint
from shopee.datasets import init_dataloaders, init_datasets
from shopee.losses import ArcFaceLoss
from shopee.metric import treshold_finder
from shopee.models import ArcFaceNet
from shopee.optimizers import init_optimizer, init_scheduler
from shopee.paths import LOGS_PATH, MODELS_PATH, ON_DRIVE_PATH

ON_COLAB = "COLAB" in os.environ


def train_eval_fold(
    df: pd.DataFrame,
    image_dir: Path,
    args: argparse.Namespace,
    Config: dict,
    exp_name: str,
    use_amp: bool,
    checkpoint_path: Optional[Path],
) -> Optional[float]:
    """
    One full train/eval loop with validation on fold `fold`
    df: should have `fold` column
    """
    train_df = df[df["fold"] != args.fold].copy().reset_index(drop=True)
    # validation is allways on hard labels
    val_df = df[df["fold"] == args.fold].copy().reset_index(drop=True)

    train_ds, val_ds = init_datasets(Config, train_df, val_df, image_dir)
    dataloaders = init_dataloaders(train_ds, val_ds, Config)
    logging.info(f"Data: train size: {len(train_ds)}, val_size: {len(val_ds)}")

    num_classes = int(train_df[Config["target_col"]].max() + 1)
    model = ArcFaceNet(num_classes, Config)
    logging.info(
        f"Model {model} created, param count: {sum([m.numel() for m in model.parameters()]):_}"
    )
    model.cuda()

    if Config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    optimizer = init_optimizer(model.parameters(), Config["opt_conf"])
    logging.info(f"Using optimizer: {optimizer}")
    amp_scaler = NativeScaler() if use_amp else None
    logging.info(f"AMP: {amp_scaler}")

    # optionally resume from a checkpoint
    resume_epoch = None
    resume_loss = None
    resume_score = None
    if checkpoint_path:
        resume_epoch, resume_loss, resume_score = resume_checkpoint(
            model, checkpoint_path, optimizer=optimizer, loss_scaler=amp_scaler,
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if Config["model_ema"]:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model,
            decay=Config["model_ema_decay"],
            device="cpu" if Config["model_ema_force_cpu"] else None,
        )
        if checkpoint_path:
            load_checkpoint(model_ema.module, checkpoint_path, use_ema=True)

    # regular init of scheduler
    scheduler = init_scheduler(optimizer, Config["sch_conf"])
    if scheduler is not None and resume_epoch is not None:
        scheduler.step(resume_epoch, resume_loss)
        logging.info(
            f"""after resume and step: lr - {optimizer.param_groups[0]['lr']},
            initial lr -{optimizer.param_groups[0]['initial_lr']}"""
        )
    # Define loss function (no sense to check val loss in ArcFace setting with new groups)
    tr_criterion = ArcFaceLoss(num_classes, s=Config["s"], m=Config["m"])

    result = train_model(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        tr_criterion=tr_criterion,
        scheduler=scheduler,
        metrics_fn=treshold_finder,
        exp_name=f"{exp_name}_f{args.fold}",
        Config=Config,
        use_amp=use_amp,
        amp_scaler=amp_scaler,
        model_ema=model_ema,
        resume_epoch=resume_epoch,
        resume_loss=resume_loss,
        resume_score=resume_score,
    )
    return result


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    optimizer: optim.Optimizer,
    tr_criterion: nn.Module,
    scheduler: Optional[Any],
    metrics_fn: Optional[Callable],
    exp_name: str,
    Config: dict,
    use_amp: bool,
    amp_scaler: Optional[NativeScaler],
    model_ema: Optional[nn.Module],
    resume_epoch: Optional[int],
    resume_loss: Optional[float],
    resume_score: Optional[float],
) -> Optional[float]:
    """Train, validate, collect metrics, and save checkpoint"""

    save_on = Config["save_on"]
    assert save_on in [
        "loss",
        "score",
    ], f"Only [loss, score] modes allowed, {save_on} provided"

    if save_on == "score":
        assert metrics_fn is not None, "Provide metric function"

    # Initialize function-level state
    epoch_metrics = {
        "exp": "",
        "epoch": 0,
        "train_loss": np.inf,
        "val_loss": np.inf,
        "val_score": -np.inf,
    }
    epoch_metrics["exp"] = exp_name
    df_columns = list(epoch_metrics.keys())

    start_epoch = resume_epoch or 0
    best_loss = resume_loss or np.inf
    best_score = resume_score or -np.inf

    if start_epoch == 0:
        # init and write metrics logging dataframe
        write_metrics_df(
            df=pd.DataFrame(columns=df_columns), logs_path=LOGS_PATH, exp_name=exp_name
        )

    # Iterate over epochs
    for epoch in range(start_epoch, Config["num_epochs"]):
        train_loss = train_epoch(
            model,
            optimizer,
            tr_criterion,
            dataloaders["train"],
            epoch,
            Config,
            use_amp,
            amp_scaler,
            model_ema,
        )
        val_loss, val_logits, val_targets = validate_epoch(
            model, dataloaders["val"], epoch, Config, use_amp,
        )

        val_score = np.nan
        th = np.nan
        if metrics_fn is not None:
            val_score, th = metrics_fn(val_logits, val_targets)
        # Loging train and val results
        logging.info(f"Epoch {epoch} - avg train loss: {train_loss:.4f}")
        logging.info(f"Epoch {epoch} - avg val score: {val_score:.4f}, th: {th:.3f}")

        if model_ema is not None and not Config["model_ema_force_cpu"]:
            _, ema_logits, ema_targets = validate_epoch(
                model_ema.module, dataloaders["val"], epoch, Config, use_amp,
            )
            ema_score = np.nan
            if metrics_fn is not None:
                ema_score = metrics_fn(ema_logits, ema_targets)
            # Loging EMA results
            logging.info(f"Epoch {epoch} - avg EMA score: {ema_score:.4f}")

        # lr scheduler step
        if scheduler is not None:
            scheduler.step(epoch + 1)
        logging.info(
            f"epoch step: lr: {optimizer.param_groups[0]['lr']}, initial lr: {optimizer.param_groups[0]['initial_lr']}"
        )

        # update metrics dict
        epoch_metrics["epoch"] = epoch
        epoch_metrics["train_loss"] = train_loss
        epoch_metrics["val_loss"] = val_loss
        epoch_metrics["val_score"] = val_score

        # write epoch raw to experiment dataframe
        epoch_raw = get_epoch_raw(epoch_metrics)
        write_metrics_df(pd.DataFrame(epoch_raw), LOGS_PATH, exp_name, header=False)

        # save on validation loss improvement
        if epoch_metrics["val_loss"] < best_loss:
            best_loss = epoch_metrics["val_loss"]
            logging.info(f"Epoch {epoch} - Save @ loss: {best_loss:.4f}")
            save_checkpoint(
                epoch,
                model,
                optimizer,
                MODELS_PATH,
                exp_name + "_loss",
                epoch_metrics,
                model_ema,
                amp_scaler,
            )
            if ON_COLAB:
                # copy to persistent storage
                shutil.copy(
                    MODELS_PATH / f"{exp_name}_loss.pth", ON_DRIVE_PATH / "all_models"
                )
        # save on validation score improvement
        if epoch_metrics["val_score"] > best_score:
            best_score = epoch_metrics["val_score"]
            logging.info(f"Epoch {epoch} - Save @ score: {best_score:.4f}")
            save_checkpoint(
                epoch,
                model,
                optimizer,
                MODELS_PATH,
                exp_name + "_score",
                epoch_metrics,
                model_ema,
                amp_scaler,
            )
            if ON_COLAB:
                # copy to persistent storage
                shutil.copy(
                    MODELS_PATH / f"{exp_name}_score.pth", ON_DRIVE_PATH / "all_models"
                )
    if save_on == "loss":
        return best_loss
    if save_on == "score":
        return best_score


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    Config: dict,
    use_amp: bool,
    amp_scaler: Optional[NativeScaler],
    model_ema: Optional[nn.Module],
):
    model.train()
    optimizer.zero_grad()  # if use accum_grad > 1
    running_loss = 0.0
    bar = tqdm(dataloader)
    for step, batch in enumerate(bar, start=1):
        bar.set_description(f"Epoch {epoch} [train]".ljust(20))
        batch_loss = train_batch(
            batch,
            model,
            optimizer,
            criterion,
            Config,
            use_amp,
            amp_scaler,
            model_ema,
            step,
        )
        bar.set_postfix(loss=f"[train]: {batch_loss}")
        running_loss += batch_loss
    # on epoch end
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def train_batch(
    batch: Dict,
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    Config: dict,
    use_amp: bool,
    amp_scaler: Optional[NativeScaler],
    model_ema: Optional[nn.Module],
    step: int,
) -> float:
    inputs = batch["image"].cuda()
    if Config["channels_last"]:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    targets = batch["label"].cuda()
    # try to define weights
    try:
        weights = batch["weights"].cuda()
    except KeyError:
        # dataset doesn't provide weights = set to None
        weights = None
    with autocast(enabled=use_amp):
        outputs = model(inputs)
        criterion.weight = weights
        loss = criterion(outputs, targets)
    # backward and update
    if use_amp:
        amp_scaler(
            loss,
            optimizer,
            step,
            Config["accum_grad"],
            Config["clip_grad"],
            model.parameters(),
        )
    else:
        (loss / Config["accum_grad"]).backward()
        if step % Config["accum_grad"] == 0:
            optimizer.step()
            optimizer.zero_grad()

    if model_ema is not None:
        model_ema.update(model)
    return loss.item()


def validate_epoch(
    model: nn.Module, dataloader: DataLoader, epoch: int, Config: Any, use_amp: bool,
):
    """We don't compute validation loss here as it has not much sense
       to check it on data from completly different classes. Instead
       we validate the competition score and trying to achieve best val score"""
    epoch_loss = np.inf
    model.eval()
    epoch_logits = []
    epoch_targets = []
    bar = tqdm(dataloader)
    for batch in bar:
        bar.set_description(f"Epoch {epoch} [validation]".ljust(20))
        batch_logits, batch_targets = validate_batch(batch, model, Config, use_amp)
        epoch_logits.append(batch_logits)
        epoch_targets.append(batch_targets)
    # on epoch end
    epoch_logits = torch.cat(epoch_logits)
    epoch_targets = torch.cat(epoch_targets)
    return epoch_loss, epoch_logits, epoch_targets


def validate_batch(
    batch: Dict, model: nn.Module, Config: Any, use_amp: bool,
):
    """
    It returns also detached to cpu output tensor
    and targets tensor
    """
    inputs = batch["image"].cuda()
    if Config["channels_last"]:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    targets = batch["label"].cuda()
    with torch.no_grad():
        with autocast(enabled=use_amp):
            outputs = model(inputs)
    # index into only main task classes (if aux task is used)
    return outputs, targets


def write_metrics_df(
    df: pd.DataFrame, logs_path: Path, exp_name: str, **kwargs
) -> None:
    df.to_csv(logs_path / (exp_name + ".csv"), mode="a", index=False, **kwargs)


def get_epoch_raw(metrics: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Dict value should be list, so to append to dataframe"""
    return {key: [value] for key, value in metrics.items()}
