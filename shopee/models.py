from typing import Union
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
        self,
        num_classes: int,
        Config: dict,
        pretrained: bool,
    ):
        super(ArcFaceNet, self).__init__()
        self.backbone = timm_backbone(
            f_out=0,
            arch=Config["arch"],
            pretrained=pretrained,
            global_pool=Config["global_pool"],
            drop_rate=Config["drop_rate"],
            bn_momentum=Config["bn_momentum"],
            **Config["model_kwargs"],
        )
        self.arch = Config["arch"]
        num_features = self.backbone.num_features
        if Config["global_pool"] == "catavgmax":
            num_features *= 2

        self.bn1 = nn.BatchNorm1d(num_features)
        self.dropout = nn.Dropout(Config["drop_rate"], inplace=True)
        self.fc1 = nn.Linear(num_features, Config["embed_size"])
        self.bn2 = nn.BatchNorm1d(Config["embed_size"])

        # check if config (maybe used to train past models) has arc face text key
        try:
            text = Config["arc_face_text"]
        except KeyError:
            print("Old models: set text input to False")
            text = False
        if text:
            assert isinstance(text, int)
            h_sz = text // 2
            self.text_head = nn.Sequential(
                nn.Linear(text, h_sz),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(h_sz),
                nn.Dropout(Config["drop_rate"], inplace=True),
                nn.Linear(h_sz, Config["embed_size"]),
            )
        arc_in = Config["embed_size"]
        if text:
            arc_in *= 2

        self.margin = ArcFaceLayer(emb_size=arc_in, output_classes=num_classes)

    def __repr__(self):
        return repr(self.__class__.__name__) + f" with backbone: {self.arch}"

    def forward(self, x, text=None):
        features = self.backbone(x)
        features = self.bn1(features)
        features = self.dropout(features)
        features = self.fc1(features)
        features = self.bn2(features)
        if text is not None:
            text_features = self.text_head(text)
            features = torch.cat([features, text_features], dim=1)
        if self.training:
            return self.margin(features)
        return F.normalize(features)
