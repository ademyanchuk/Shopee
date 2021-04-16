"""Atomic unit of experiment is one full train run on one fold
   Experiment will run only on machine with >= 1 GPU
"""
import argparse
import logging
import os
from shopee.validate_fold import validate_fold

import pandas as pd
import torch

from shopee.config import Config, save_config_yaml
from shopee.log_utils import get_commit_hash, get_exp_name, setup_logg
from shopee.paths import DATA_ROOT, LOGS_PATH, META_PATH, MODELS_PATH, ON_DRIVE_PATH
from shopee.train import ON_COLAB, train_eval_fold

parser = argparse.ArgumentParser(description="Experiment")
parser.add_argument(
    "--fold",
    "-F",
    type=int,
    default=0,
    choices=[0, 1, 2, 3],
    help="Fold # to use for validation. Default=0.",
)
parser.add_argument(
    "--train_csv",
    default="train_folds.csv",
    help="CSV file to use in training. Make sure it conforms to default file",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    help="Resume full model and optimizer state from checkpoint (default: none)",
)
# this arg is required to utilize DDP training
parser.add_argument("--local_rank", default=0, type=int)


def main():
    exp_name = get_exp_name(debug=Config["debug"])
    args = parser.parse_args()

    checkpoint_path = None
    # Resume Experiment is provided
    if args.resume:
        checkpoint_path = MODELS_PATH / f"{args.resume}_f{args.fold}.pth"
        assert (
            checkpoint_path.exists()
        ), f"Experiment {args.resume} checkpoint doesn't exist"
        exp_name = args.resume

    # Save training confiig
    save_config_yaml(Config, LOGS_PATH, exp_name)
    # Setup logging
    _ = setup_logg(LOGS_PATH, exp_name)
    logging.info(f"Experiment: {exp_name}")

    # Setup distributed training if avail
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        logging.info(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        logging.info("Training with a single process on 1 GPUs.")
    assert args.rank >= 0
    # seed
    torch.manual_seed(Config["seed"] + args.rank)

    logging.info(f"Git commit hash: {get_commit_hash()}")
    logging.info(f"Using CSV: {META_PATH / args.train_csv}")

    full_df = pd.read_csv(META_PATH / args.train_csv)
    if Config["debug"]:
        full_df = full_df.sample(n=2000, random_state=Config["seed"]).reset_index(
            drop=True
        )

    _ = train_eval_fold(
        df=full_df,
        image_dir=DATA_ROOT / "train_images",
        args=args,
        Config=Config,
        exp_name=exp_name,
        use_amp=True,
        checkpoint_path=checkpoint_path,
    )

    # validat fold's result and save predictions dataframe
    score, pred_df = validate_fold(
        exp_name, args.fold, Config, full_df, DATA_ROOT / "train_images"
    )
    logging.info(f"validation for fold: {args.fold}, score: {score}")
    if ON_COLAB:
        save_path = ON_DRIVE_PATH / "preds" / f"{exp_name}_f{args.fold}.csv"
        pred_df.to_csv(save_path, index=False, header=True)


if __name__ == "__main__":
    main()
