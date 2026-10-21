"""Standard baseline: torchvision ResNet-50 (ImageNet-pretrained) with a single-logit binary head."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_model(pretrained: bool = True) -> nn.Module:
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def weighted_bce(pos_weight: float, device) -> nn.Module:
    """BCE-with-logits with positives up-weighted by n_neg / n_pos of the training set."""
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
