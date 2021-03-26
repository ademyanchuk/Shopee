import torch
from torch import nn
from torch.nn import functional as F
import timm


def timm_backbone(
    f_out: int,
    arch: str,
    pretrained: bool,
    global_pool: str,
    drop_rate: float,
    bn_momentum: float,
    **kwargs,
) -> nn.Module:
    m = timm.create_model(arch, pretrained=pretrained, drop_rate=drop_rate, **kwargs)
    m.reset_classifier(f_out, global_pool=global_pool)
    # reset batch_norm momentum
    for module in m.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = bn_momentum
    return m


class ArcFaceLayer(nn.Module):
    def __init__(self, emb_size, output_classes):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(emb_size, output_classes))
        nn.init.xavier_normal_(self.W)

    def forward(self, x):
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.W, dim=0)
        return x_norm @ W_norm
