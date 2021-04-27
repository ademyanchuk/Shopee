import torch.nn as nn
from typing import Any, Dict
import torch.optim as torch_optim
from timm import scheduler, optim


def init_optimizer(model: nn.Module, conf_dict: Dict, diff_lr: float = 0.0):
    """
    Initialize optimizer according a `conf_dict` params
    Only one key on outter level of config dict is allowed
    """
    optim_key = list(conf_dict.keys())[0]
    base_lr = conf_dict[optim_key].pop("lr")
    head_lr = base_lr
    if diff_lr != 0:
        head_lr *= diff_lr
    head_params = [
        model.bn1.parameters(),
        model.dropout.parameters(),
        model.fc1.parameters(),
        model.bn2.parameters(),
        model.margin.parameters(),
    ]
    params = [
        {"params": model.backbone.parameters(), "lr": base_lr},
        {"params": head_params, "lr": head_lr},
    ]
    if optim_key == "adam":
        return torch_optim.AdamW(params, **conf_dict[optim_key])
    elif optim_key == "radam":
        return optim.RAdam(params, **conf_dict[optim_key])
    elif optim_key == "rmsproptf":
        return optim.RMSpropTF(params, **conf_dict[optim_key])
    else:
        raise NotImplementedError


def init_scheduler(optimizer: Any, conf_dict: Dict, initialize: bool = True):
    """
    Initialize scheduler according a `conf_dict` params
    Only one key on outter level of config dict is allowed
    """
    if "step" in conf_dict:
        kwargs = conf_dict["step"]
        return scheduler.StepLRScheduler(optimizer, **kwargs)
    elif "cosine" in conf_dict:
        kwargs = conf_dict["cosine"]
        return scheduler.CosineLRScheduler(optimizer, **kwargs)
    else:
        raise NotImplementedError
