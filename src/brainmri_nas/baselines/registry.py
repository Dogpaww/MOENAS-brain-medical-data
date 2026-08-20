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

Most builders wrap torchvision (resnet18/34, vgg16, efficientnet_v2_s/m);
senet and deit_small come from timm instead, since neither is in
torchvision.models. "deit_small" stands in for T2T-ViT, which turned out to
be genuinely absent from timm 1.0.28 (see the builder's docstring for why
DeiT is a well-justified substitute rather than a downgrade).
"""

from __future__ import annotations

from collections.abc import Callable

import timm
import torch.nn as nn
from torchvision import models

# DeiT bakes its position embeddings into a fixed patch grid, so unlike every
# conv backbone below, "the model" genuinely depends on input resolution.
# Every controlled-comparison run in this project trains at
# DatasetConfig.image_size (96, see run_baseline.py's --image-size default);
# 224 is only ever used for a separately-named, explicitly-labeled secondary
# comparison. Hardcoding 96 here avoids threading an image_size argument
# through MODEL_REGISTRY's (num_classes) -> nn.Module signature, and thus
# every other builder and every call site, for the sake of the one model
# that needs it.
_DEIT_IMAGE_SIZE = 96


def _resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _resnet34(num_classes: int) -> nn.Module:
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _vgg16(num_classes: int) -> nn.Module:
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    # classifier[6] is the final 4096->1000 layer; everything before it
    # (the two 4096-unit hidden layers + their dropout) is left pretrained.
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def _densenet121(num_classes: int) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def _efficientnet_v2_s(num_classes: int) -> nn.Module:
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _efficientnet_v2_m(num_classes: int) -> nn.Module:
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _senet(num_classes: int) -> nn.Module:
    # legacy_seresnet50: SE-ResNet-50 (Hu et al. 2018) -- the SENet variant
    # most commonly used as a classification baseline in brain-tumor-MRI
    # papers (e.g. BTSC-TNAS). Adds squeeze-and-excitation channel attention
    # on a ResNet-50 backbone, so the comparison against the plain
    # resnet18/resnet34 already in this registry isolates what SE attention
    # buys. timm's list_models(pretrained=True) filter misses this family --
    # confirmed real pretrained weights instead via get_pretrained_cfg().
    return timm.create_model("legacy_seresnet50", pretrained=True, num_classes=num_classes)


def _deit_small(num_classes: int) -> nn.Module:
    # Substitute for T2T-ViT (Yuan et al. 2021): confirmed completely absent
    # from timm 1.0.28 under any name (list_models('*t2t*') returns []).
    # DeiT (Touvron et al. 2021) is arguably a better-justified pick for this
    # project anyway -- it was designed specifically for data-efficient
    # training on datasets far smaller than JFT-300M/ImageNet-21k, directly
    # relevant to this ~2,600-image corpus, whereas T2T-ViT's contribution
    # was mainly compute-efficient tokenization for ImageNet-scale training.
    # img_size tells timm to interpolate the pretrained position embeddings
    # from their native 224x224/14x14 patch grid down to 96x96/6x6.
    return timm.create_model(
        "deit_small_patch16_224", pretrained=True, img_size=_DEIT_IMAGE_SIZE, num_classes=num_classes
    )


MODEL_REGISTRY: dict[str, Callable[[int], nn.Module]] = {
    "resnet18": _resnet18,
    "resnet34": _resnet34,
    "vgg16": _vgg16,
    "densenet121": _densenet121,
    "efficientnet_v2_s": _efficientnet_v2_s,
    "efficientnet_v2_m": _efficientnet_v2_m,
    "senet": _senet,
    "deit_small": _deit_small,
}


def build_baseline_model(name: str, num_classes: int) -> nn.Module:
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown baseline model {name!r}. Available: {available}.") from None
    return builder(num_classes)
