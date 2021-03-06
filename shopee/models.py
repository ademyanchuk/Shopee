from pathlib import Path
from typing import Union
import timm
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


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
        self, num_classes: int, Config: dict, pretrained: bool,
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
        self.margin = ArcFaceLayer(
            emb_size=Config["embed_size"], output_classes=num_classes
        )

    def __repr__(self):
        return repr(self.__class__.__name__) + f" with backbone: {self.arch}"

    def forward(self, x):
        features = self.backbone(x)
        features = self.bn1(features)
        features = self.dropout(features)
        features = self.fc1(features)
        features = self.bn2(features)
        if self.training:
            return self.margin(features)
        return F.normalize(features)


class ArcFaceBert(nn.Module):
    def __init__(
        self, num_classes: int, Config: dict, pretrained: Union[bool, Path],
    ):
        super(ArcFaceBert, self).__init__()
        self.arch = Config["bert_name"]
        if pretrained:
            self.arch = pretrained
        self.backbone = AutoModel.from_pretrained(self.arch)
        num_features = self.backbone.config.hidden_size

        self.bn1 = nn.BatchNorm1d(num_features)
        self.dropout = nn.Dropout(Config["drop_rate"], inplace=True)
        self.fc1 = nn.Linear(num_features, Config["embed_size"])
        self.bn2 = nn.BatchNorm1d(Config["embed_size"])

        self.margin = ArcFaceLayer(
            emb_size=Config["embed_size"], output_classes=num_classes
        )

    def __repr__(self):
        return repr(self.__class__.__name__) + f" with backbone: {self.arch}"

    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        features = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        features = self.mean_pooling(features, attention_mask)
        features = self.bn1(features)
        features = self.dropout(features)
        features = self.fc1(features)
        features = self.bn2(features)
        if self.training:
            return self.margin(features)
        return F.normalize(features)


def init_model(num_classes: int, Config: dict, pretrained: Union[bool, Path]):
    # work with old and new configs
    try:
        is_bert = Config["arc_face_text"]
    except KeyError:
        print("Old Config: setting BERT to False")
        is_bert = False
    if is_bert:
        return ArcFaceBert(num_classes, Config, pretrained)
    else:
        if not isinstance(pretrained, bool):
            pretrained = False
        return ArcFaceNet(num_classes, Config, pretrained)
