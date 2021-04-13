import argparse
import logging
import sys

import optuna
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from shopee.log_utils import get_exp_name
from shopee.metric import validate_score
from shopee.paths import META_PATH, ON_DRIVE_PATH


def positive_int(x):
    x = int(x)
    if x <= 0:
        raise argparse.ArgumentTypeError("Should be > 0")
    return x


def objective(trial: optuna.trial.Trial):
    df = pd.read_csv(META_PATH / "train_folds.csv")
    val_df = df[df["fold"] == 0].copy().reset_index(drop=True)

    txt_model_args = {
        "analyzer": trial.suggest_categorical("analyzer", ["word", "char", "char_wb"]),
        "ngram_range": (trial.suggest_int("ngram_range_low", low=1, high=3), 3),
        "max_features": trial.suggest_int(
            "max_features", low=1024, high=1024 * 20, step=1024
        ),
        "max_df": trial.suggest_float("max_df", low=0.9, high=1.0, step=0.01),
        "stop_words": "english",
        "binary": True,
    }

    model = TfidfVectorizer(**txt_model_args)
    text_embeds = model.fit_transform(val_df["title"]).toarray()
    text_embeds = torch.from_numpy(text_embeds)
    score, _ = validate_score(val_df, text_embeds, th=None)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HP optimize script")
    parser.add_argument(
        "--ntrials",
        type=positive_int,
        required=True,
        help="Number of optuna search trials",
    )
    args = parser.parse_args()
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = get_exp_name(debug=False)
    storage_name = f"sqlite:////{ON_DRIVE_PATH / study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=False,
    )
    study.optimize(objective, n_trials=args.ntrials, timeout=3600 * 23)
    print(study.best_value, study.best_params)
