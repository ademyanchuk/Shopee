import argparse
import logging
from pathlib import Path
from shopee.amp_scaler import NativeScaler
from shopee.optimizers import init_optimizer
from typing import Optional

import pandas as pd
from pandas.core.dtypes.missing import notna

from shopee.datasets import init_dataloaders, init_datasets
from shopee.models import ArcFaceNet


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

    assert train_ds.num_classes is not None
    model = ArcFaceNet(train_ds.num_classes, Config)
    logging.info(
        f"Model {model} created, param count:{sum([m.numel() for m in model.parameters()])}"
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
    if checkpoint_path:
        resume_epoch, resume_loss = resume_checkpoint(
            model, checkpoint_path, optimizer=optimizer, loss_scaler=amp_scaler,
        )
        logging.info(
            f"""after optim resume: lr - {optimizer.param_groups[0]['lr']},
            initial lr -{optimizer.param_groups[0]['initial_lr']}"""
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    # if Config.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    #     model_ema = ModelEmaV2(
    #         model,
    #         decay=Config.model_ema_decay,
    #         device="cpu" if Config.model_ema_force_cpu else None,
    #     )
    #     if checkpoint_path:
    #         load_checkpoint(model_ema.module, checkpoint_path, use_ema=True)

    # if checkpoint_path and ("plateau" in Config.sch_conf):
    #     scheduler = init_scheduler(optimizer, Config.sch_conf, initialize=True)
    #     logging.info(
    #         f"""on scheduler init: lr - {optimizer.param_groups[0]['lr']},
    #         initial lr -{optimizer.param_groups[0]['initial_lr']}"""
    #     )
    #     # need to load state for reduce lr on plateau scheduler
    #     resume_scheduler(checkpoint_path, scheduler)
    #     logging.info(
    #         f"""on scheduler resume: lr - {optimizer.param_groups[0]['lr']},
    #         initial lr -{optimizer.param_groups[0]['initial_lr']}"""
    #     )
    # else:
    #     # regular init of scheduler
    #     scheduler = init_scheduler(optimizer, Config.sch_conf)
    #     logging.info(
    #         f"""on scheduler init: lr - {optimizer.param_groups[0]['lr']},
    #         initial lr -{optimizer.param_groups[0]['initial_lr']}"""
    #     )
    # if scheduler is not None and resume_epoch is not None:
    #     scheduler.step(resume_epoch, resume_loss)
    #     logging.info(
    #         f"""after resume and step: lr - {optimizer.param_groups[0]['lr']},
    #         initial lr -{optimizer.param_groups[0]['initial_lr']}"""
    #     )
    # # Define loss functions
    # tr_criterion, val_criterion = init_losses(
    #     name=Config.tr_criterion, custom=kp, Config=Config
    # )

    # result = train_model(
    #     model=model,
    #     dataloaders=dataloaders,
    #     tr_criterion=tr_criterion,
    #     val_criterion=val_criterion,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     metrics_fn=get_auc_score,
    #     num_epochs=Config.num_epochs,
    #     exp_name=f"{exp_name}_f{fold}",
    #     logs_path=Config.LOGS_PATH,
    #     models_path=Config.MODELS_PATH,
    #     device=device,
    #     conf=Config,
    #     use_amp=use_amp,
    #     amp_scaler=amp_scaler,
    #     model_ema=model_ema,
    #     resume_epoch=resume_epoch,
    #     resume_loss=resume_loss,
    #     save_on=save_on,
    # )
    # return result
