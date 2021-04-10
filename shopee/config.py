from pathlib import Path

import yaml

Config = {
    "debug": False,
    "seed": 42,
    # data
    "image_id_col": "image",
    "target_col": "target",
    "img_size": 512,  # resize
    "crop_size": None,  # if none == img_size
    "bs": 48,
    "num_workers": 4,
    # model
    "arch": "tf_efficientnet_b0_ns",
    "pretrained": True,
    "global_pool": "catavgmax",
    "embed_size": 256,
    "drop_rate": 0.0,
    "model_kwargs": {"drop_path_rate": 0.3},
    "bn_momentum": 0.1,  # default 0.1
    "channels_last": False,
    # ema
    "model_ema": False,
    "model_ema_decay": 0.999,
    "model_ema_force_cpu": False,
    # optimizer
    "opt_conf": {"adam": {"lr": 1e-3, "weight_decay": 0.0}},
    "sch_conf": {
        "cosine": {
            "t_initial": 30,
            "lr_min": 1e-7,
            "warmup_t": 10,
            "warmup_lr_init": 5e-6,
        }
    },
    # loss
    "s": 30,  # arcface s scalar
    "m": 0.7,  # arcface margin
    # train
    "aug_type": "rand",
    "rand_aug_severity": 5,
    "rand_aug_width": 5,
    "num_epochs": 30,
    "return_best": "score",
    "accum_grad": 1,
    "clip_grad": 1.0,  # norm of parameters grad
}


def save_config_yaml(config: dict, log_dir: Path, exp_name: str):
    with open(log_dir / f"{exp_name}_conf.yaml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)
