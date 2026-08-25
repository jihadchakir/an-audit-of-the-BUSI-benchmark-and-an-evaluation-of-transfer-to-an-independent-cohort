from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork


class ResNetFPNEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        fpn_channels: int = 256,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_bn: bool = False,
    ):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.body = IntermediateLayerGetter(
            backbone, return_layers={"layer2": "c3", "layer3": "c4", "layer4": "c5"}
        )
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[512, 1024, 2048], out_channels=fpn_channels
        )
        self.head = nn.Sequential(
            nn.Linear(3 * fpn_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )
        self.embedding_dim = embedding_dim
        self.freeze_bn = freeze_bn

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.body.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: Dict[str, torch.Tensor] = self.body(x)
        pyramid = self.fpn(feats)
        pooled: List[torch.Tensor] = [
            F.adaptive_avg_pool2d(pyramid[k], 1).flatten(1) for k in ("c3", "c4", "c5")
        ]
        z = torch.cat(pooled, dim=1)
        # fp32 from here: distances downstream must not be computed in fp16
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = self.head(z.float())
            z = F.normalize(z, p=2, dim=1)
        return z

    def param_groups(self, lr: float, backbone_lr_mult: float):
        return [
            {"params": self.body.parameters(), "lr": lr * backbone_lr_mult},
            {"params": list(self.fpn.parameters()) + list(self.head.parameters()), "lr": lr},
        ]


@torch.no_grad()
def extract_embeddings(model: nn.Module, loader, device: str = "cuda") -> tuple:
    """Returns (E, y) as numpy arrays. Batched, unlike the original code which
    called .predict() once per image inside a Python loop."""
    model.eval()
    embs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x)
        embs.append(z.float().cpu().numpy())
        labels.append(y.numpy())
    import numpy as np

    return np.concatenate(embs), np.concatenate(labels)
