import os
import random
from typing import Tuple

import numpy as np
import torch


def seed_torch(seed=1982):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using pytorch device: {device}")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_properties(device))
    return device


# one img tensor reversed
def reverse_img_tensor(img: torch.Tensor, denorm: bool = True) -> np.ndarray:
    # Reverse all preprocessing
    img = img.numpy().transpose((1, 2, 0))
    if denorm:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = (img * 255).astype(np.uint8)
    return img