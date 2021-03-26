import timm
import torch
from torch import nn
from torch.nn import functional as F


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


class ArcFaceNet(nn.Module):
    def __init__(
        self, num_classes: int, Config: dict,
    ):
        super(ArcFaceNet, self).__init__()
        self.backbone = timm_backbone(
            f_out=0,
            arch=Config["arch"],
            pretrained=False,
            global_pool=Config["global_pool"],
            drop_rate=Config["drop_rate"],
            bn_momentum=Config["bn_momentum"],
            **Config["model_kwargs"],
        )
        num_features = self.backbone.num_features

        self.bn1 = nn.BatchNorm2d(num_features)
        self.dropout = nn.Dropout2d(Config["drop_rate"], inplace=True)
        self.fc1 = nn.Linear(num_features, Config["embed_size"])
        self.bn2 = nn.BatchNorm1d(Config["embed_size"])

        self.margin = ArcFaceLayer(
            emb_size=Config["embed_size"], output_classes=num_classes
        )

    def forward(self, x, labels=None):
        features = self.backbone(x)
        features = self.bn1(features)
        features = self.dropout(features)
        features = self.fc1(features)
        features = self.bn2(features)
        if labels is not None:
            return self.margin(features, labels)
        return F.normalize(features)
