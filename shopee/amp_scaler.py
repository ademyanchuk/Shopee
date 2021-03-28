""" CUDA / AMP utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch.nn as nn
from torch.cuda.amp import GradScaler


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = GradScaler()

    def __repr__(self) -> str:
        return repr(self.__class__.__name__)

    def __call__(
        self,
        loss,
        optimizer,
        step,
        accum_grad,
        clip_grad=None,
        parameters=None,
        create_graph=False,
    ):
        self._scaler.scale(loss / accum_grad).backward(create_graph=create_graph)
        if step % accum_grad == 0:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                nn.utils.clip_grad_norm_(parameters, clip_grad)
            self._scaler.step(optimizer)
            self._scaler.update()
            optimizer.zero_grad()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
