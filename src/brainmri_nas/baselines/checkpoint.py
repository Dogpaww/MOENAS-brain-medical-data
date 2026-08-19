"""Checkpoint save/load for external baseline models (branch: benchmark).

Deliberately separate from `training/checkpoint.py`, which bundles NAS-
specific reconstruction data (genotype, chromosome, `build_model` kwargs)
that a plain torchvision model doesn't have. Reusing that module here would
mean `rebuild_model_from_checkpoint` unconditionally calling
`NetworkGenotype.from_dict` on a payload that never had a genotype to begin
with. This mirrors its cpu_state_dict-snapshot discipline (never save a live
model reference; detach/clone every tensor) without the NAS-specific parts.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from brainmri_nas.baselines.registry import build_baseline_model


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    model_name: str,
    class_to_idx: dict,
    image_size: int,
    epoch: int,
    validation_metrics: dict,
) -> None:
    payload = {
        "model_state": cpu_state_dict(model),
        "model_name": model_name,
        "class_to_idx": dict(class_to_idx),
        "image_size": image_size,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict:
    # weights_only=False: our own pipeline's output, not an untrusted download.
    return torch.load(path, map_location=map_location, weights_only=False)


def rebuild_model_from_checkpoint(payload: dict) -> nn.Module:
    model = build_baseline_model(payload["model_name"], num_classes=len(payload["class_to_idx"]))
    model.load_state_dict(payload["model_state"])
    return model
