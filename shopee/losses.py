from torch import nn
from torch.nn import functional as F


class ArchFaceLoss(nn.Module):
    def __init__(self, num_classes, s=10, m=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.s = s
        self.m = m

    def forward(self, cosine, target):
        # this prevents nan when a value slightly crosses 1.0 due to numerical error
        cosine = cosine.clip(-1 + 1e-7, 1 - 1e-7)
        arcosine = cosine.arccos()
        arcosine += F.one_hot(target, num_classes=self.num_classes) * self.m
        cosine2 = arcosine.cos()
        cosine2 = cosine2 * self.s
        return F.cross_entropy(cosine2, target)
