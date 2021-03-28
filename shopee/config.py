from pathlib import Path

import yaml

Config = {
    "debug": True,
    "seed": 42,
    # data
    "image_id_col": "image",
    "target_col": "target",
    "img_size": 384,  # resize
    "crop_size": None,  # if none == img_size
    "bs": 32,
    "num_workers": 6,
    # model bacbone
    "arch": "resnet50d",
    "pretrained": True,
    "global_pool": "avg",
    "embed_size": 512,
    "drop_rate": 0.0,
    "model_kwargs": {"drop_path_rate": None},
    "bn_momentum": 0.1,  # default 0.1
    "channels_last": False,
    # optimizer
    "opt_conf": {"adam": {"lr": 3e-4, "weight_decay": 0.0}},
    # loss
    "s": 10,  # arcface s scalar
    "m": 0.5,  # arcface margin
}


def save_config_yaml(config: dict, log_dir: Path, exp_name: str):
    with open(log_dir / f"{exp_name}_conf.yaml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)
