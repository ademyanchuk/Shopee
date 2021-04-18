from pathlib import Path

import yaml

Config = {
    "debug": False,
    "seed": 42,
    # data
    "image_id_col": "image",
    "target_col": "target",
    "img_size": 1024,  # resize
    "crop_size": None,  # if none == img_size
    "bs": 16,
    "num_workers": 4,
    # model
    "moco": False,
    "arch": "tf_efficientnet_b1_ns",
    "pretrained": True,
    "global_pool": "catavgmax",
    "embed_size": 256,
    "drop_rate": 0.0,
    "model_kwargs": {"drop_path_rate": 0.2},
    "bn_momentum": 0.01,  # default 0.1
    "channels_last": False,
    # ema
    "model_ema": False,
    "model_ema_decay": 0.999,
    "model_ema_force_cpu": False,
    # optimizer
    "opt_conf": {"adam": {"lr": 5e-4, "weight_decay": 0.0}},
    "sch_conf": {
        "cosine": {
            "t_initial": 20,
            "lr_min": 5e-7,
            "warmup_t": 10,
            "warmup_lr_init": 1e-6,
        }
    },
    # loss
    "s": 30,  # arcface s scalar
    "m": 0.5,  # arcface margin
    # train
    "aug_type": "albu",
    "rand_aug_severity": 5,
    "rand_aug_width": 5,
    "num_epochs": 20,
    "return_best": "score",
    "accum_grad": 1,
    "clip_grad": 1.0,  # norm of parameters grad
    # text
    "tfidf_args": {
        "analyzer": "char_wb",
        "ngram_range": (1, 3),
        "max_features": 5120,
        "max_df": 0.96,
        "binary": True,
    },  # provide valid tfidf args here
}


def save_config_yaml(config: dict, log_dir: Path, exp_name: str):
    with open(log_dir / f"{exp_name}_conf.yaml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)


class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", PrettySafeLoader.construct_python_tuple
)


def load_config_yaml(conf_dir: Path, exp_name: str):
    with open(conf_dir / f"{exp_name}_conf.yaml", "r") as f:
        Config = yaml.load(f, Loader=PrettySafeLoader)
    return Config
