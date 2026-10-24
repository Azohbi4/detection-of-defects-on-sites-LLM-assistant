"""Architectures pré-entraînées (ImageNet) adaptées en classification multi-label."""

import torch
from torch import nn
from torchvision import models

ARCHS = ("resnet18", "resnet50", "efficientnet_b0")


def build_model(arch: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    else:
        raise ValueError(f"Architecture inconnue : {arch} (choix : {ARCHS})")
    return model


def _head(model: nn.Module, arch: str) -> list[nn.Module]:
    return [model.fc] if arch.startswith("resnet") else [model.classifier]


def _last_block(model: nn.Module, arch: str) -> list[nn.Module]:
    return [model.layer4] if arch.startswith("resnet") else list(model.features[-3:])


def set_trainable(model: nn.Module, arch: str, stage: str) -> list[nn.Parameter]:
    """stage « head » : seule la tête apprend ; stage « finetune » : tête + dernier bloc dégelés."""
    for p in model.parameters():
        p.requires_grad = False
    modules = _head(model, arch) + (_last_block(model, arch) if stage == "finetune" else [])
    params = [p for m in modules for p in m.parameters()]
    for p in params:
        p.requires_grad = True
    return params


class FocalLoss(nn.Module):
    """Focal loss multi-label (Lin et al., 2017) : réduit le poids des exemples faciles (le fond béton)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


def make_criterion(kind: str, train_targets: torch.Tensor) -> nn.Module:
    if kind == "bce":
        return nn.BCEWithLogitsLoss()
    if kind == "weighted":
        pos = train_targets.sum(0)
        pos_weight = ((len(train_targets) - pos) / pos.clamp(min=1)).clamp(1.0, 50.0)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if kind == "focal":
        return FocalLoss()
    raise ValueError(f"Perte inconnue : {kind}")
