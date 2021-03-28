import torch.nn as nn
from typing import Any, Dict
import torch.optim as torch_optim
from timm import scheduler, optim


def init_optimizer(model_params: Any, conf_dict: Dict):
    """
    Initialize optimizer according a `conf_dict` params
    Only one key on outter level of config dict is allowed
    """
    if "adam" in conf_dict:
        kwargs = conf_dict["adam"]
        return torch_optim.AdamW(model_params, **kwargs)
    elif "radam" in conf_dict:
        kwargs = conf_dict["radam"]
        return optim.RAdam(model_params, **kwargs)
    elif "rmsproptf" in conf_dict:
        kwargs = conf_dict["rmsproptf"]
        return optim.RMSpropTF(model_params, **kwargs)
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