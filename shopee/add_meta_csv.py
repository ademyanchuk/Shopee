"""Here I add image sizes and median hamming distance between images per group
   This could be useful later for error analysis
   Group-K-Folds added here as well. Prefer to use created file for experiments!
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PIL.Image import Image as PilImage
from sklearn.model_selection import GroupKFold


def create_meta_csv(data_path: Path, dest_path: Path):
    train_df = pd.read_csv(data_path / "train.csv")
    # add image sizes
    print("adding image sizes")
    sizes = [
        get_size(Image.open(data_path / "train_images" / fname))
        for fname in train_df["image"]
    ]
    train_df = pd.concat([train_df, pd.DataFrame(sizes)], axis=1)
    # add hamming median distace per group
    print("adding median hamming distance/group")
    hamm_medians = {}
    for group in train_df["label_group"].unique():
        group_phashes = train_df.loc[
            train_df["label_group"] == group, "image_phash"
        ].tolist()
        hamm_medians[group] = group_dist_median(group_phashes)
    train_df["hamming_median_dist"] = train_df["label_group"].apply(
        lambda x: hamm_medians[x]
    )
    # add folds
    print("adding folds")
    add_group_kfold(train_df)
    # save
    save_file = dest_path / "train_folds.csv"
    print(f"Saving to {save_file}")
    train_df.to_csv(save_file, header=True, index=False)


def get_size(img: PilImage):
    return {"hight": img.height, "width": img.width}


def hamming_dist(str1: str, str2: str):
    n1 = int(str1, 16)
    n2 = int(str2, 16)

    x = n1 ^ n2
    set_bits = 0

    while x > 0:
        set_bits += x & 1
        x >>= 1

    return set_bits


def group_dist_median(group_phashes):
    dists = []
    for i, one_phash in enumerate(group_phashes[:-1]):
        for other_phash in group_phashes[i + 1 :]:
            dists.append(hamming_dist(one_phash, other_phash))
    return np.median(dists)


def add_group_kfold(df: pd.DataFrame):
    gkf = GroupKFold(n_splits=5)
    df["fold"] = -1
    for fold, (_, valid_idx) in enumerate(gkf.split(df, None, df["label_group"])):
        df.loc[valid_idx, "fold"] = fold
