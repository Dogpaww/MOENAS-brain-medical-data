"""Registry of external baseline model builders (branch: benchmark).

Each entry maps a CLI-facing model name to a builder function
`(num_classes) -> nn.Module` returning an ImageNet-pretrained torchvision
model with its classifier head replaced for `num_classes` outputs.

Kept as one flat registry rather than one branch per model (see
HANDOFF.md's "should I create separate branches" discussion): every
baseline needs to coexist in a single file so a fix to shared training code
never has to be reapplied model-by-model, and so the comparison reported in
the paper stays genuinely controlled -- only the model differs, nothing
else silently drifts between baselines.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from torchvision import models


def _resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


MODEL_REGISTRY: dict[str, Callable[[int], nn.Module]] = {
    "resnet18": _resnet18,
}


def build_baseline_model(name: str, num_classes: int) -> nn.Module:
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown baseline model {name!r}. Available: {available}.") from None
    return builder(num_classes)
