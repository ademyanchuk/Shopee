import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from timm.utils import get_state_dict, unwrap_model


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    models_path: Path,
    exp_name: str,
    epoch_metrics: Dict[str, float],
    model_ema: Optional[Any],
    amp_scaler: Optional[Any],
    scheduler: Optional[Any] = None,
) -> None:
    save_state = {
        "epoch": epoch + 1,  # increment epoch (to not repeat then resume)
        "state_dict": get_state_dict(model, unwrap_model),
        "optimizer": optimizer.state_dict(),
        "val_loss": epoch_metrics["val_loss"],
        "val_score": epoch_metrics["val_score"],
    }
    if model_ema is not None:
        save_state["state_dict_ema"] = get_state_dict(model_ema, unwrap_model)
    if amp_scaler is not None:
        save_state[amp_scaler.state_dict_key] = amp_scaler.state_dict()
    if scheduler is not None:
        save_state["lr_scheduler"] = scheduler.state_dict()
    torch.save(
        save_state, f"{models_path}/{exp_name}.pth",
    )


# copied from timm
def resume_checkpoint(
    model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True
):
    resume_epoch = None
    best_loss = None
    best_score = None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        if log_info:
            logging.info("Restoring model state from checkpoint...")
        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            name = k[7:] if k.startswith("module") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        if optimizer is not None and "optimizer" in checkpoint:
            if log_info:
                logging.info("Restoring optimizer state from checkpoint...")
            optimizer.load_state_dict(checkpoint["optimizer"])

        if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
            if log_info:
                logging.info("Restoring AMP loss scaler state from checkpoint...")
            loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

        if "epoch" in checkpoint:
            resume_epoch = checkpoint["epoch"]
        if "val_loss" in checkpoint:
            best_loss = checkpoint["val_loss"]
        if "val_score" in checkpoint:
            best_score = checkpoint["val_score"]

        if log_info:
            logging.info(
                "Loaded checkpoint '{}' (epoch {})".format(
                    checkpoint_path, checkpoint["epoch"]
                )
            )
    else:
        model.load_state_dict(checkpoint)
        if log_info:
            logging.info("Loaded checkpoint '{}'".format(checkpoint_path))
    return resume_epoch, best_loss, best_score
