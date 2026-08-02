"""Checkpoint save/load (handoff §28).

Never `best_model = model` -- that's a live reference to a model object
that keeps mutating every subsequent epoch, so "the best checkpoint" would
silently become whatever the model looked like when training finished, not
whatever epoch actually had the best validation metric. `cpu_state_dict`
detaches, moves to CPU, and clones every tensor, so what gets saved is a
frozen snapshot independent of anything that happens to `model` afterward.

The checkpoint bundles everything needed to reconstruct and re-evaluate the
model in a completely separate process: model state, chromosome, genotype,
the exact `build_model` kwargs, class mapping, image size, augmentation
policy, epoch, and validation metrics (handoff §28).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from brainmri_nas.model.network import build_model
from brainmri_nas.search_space.genotype import NetworkGenotype


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    genotype: NetworkGenotype,
    model_config: dict,
    class_to_idx: dict,
    image_size: int,
    epoch: int,
    validation_metrics: dict,
    chromosome: list[float] | None = None,
    augmentation_policy: list[dict] | None = None,
) -> None:
    payload = {
        "model_state": cpu_state_dict(model),
        "chromosome": list(chromosome) if chromosome is not None else None,
        "genotype": genotype.to_dict(),
        "model_config": dict(model_config),
        "class_to_idx": dict(class_to_idx),
        "image_size": image_size,
        "augmentation_policy": augmentation_policy,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict:
    # weights_only=False: this file is produced by our own pipeline, not an
    # untrusted download, and its payload legitimately mixes tensors with
    # plain dicts/lists (genotype, config, metrics) beyond torch's
    # weights-only-safe allowlist.
    return torch.load(path, map_location=map_location, weights_only=False)


def rebuild_model_from_checkpoint(payload: dict) -> nn.Module:
    genotype = NetworkGenotype.from_dict(payload["genotype"])
    model = build_model(genotype, **payload["model_config"])
    model.load_state_dict(payload["model_state"])
    return model
