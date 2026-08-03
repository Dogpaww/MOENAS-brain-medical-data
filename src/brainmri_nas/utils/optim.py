"""Optimizer/scheduler construction shared by augmentation-policy trial
training and final training.

Adam + `MultiStepLR(milestones=[0.5*epochs, 0.75*epochs], gamma=0.1)`,
matching the legacy repo's `evaluate.py::train()` shape. The base
`learning_rate` itself is passed in by the caller (`configs/search.yaml`),
not hardcoded here -- it was originally set to 0.025 to match the legacy
value exactly, but that number is the standard DARTS *SGD* recipe's LR
(~25x too high for Adam, whose own PyTorch default is 1e-3) and was the
likely cause of large epoch-to-epoch validation swings in real runs;
`TrainingConfig`/`AugmentationConfig` now default to 1e-3 instead. Centralized
here, instead of duplicated in both training loops, so they can't silently
drift apart on this again.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer_and_scheduler(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    gamma: float = 0.1,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Degrades to "no decay" for very short runs (e.g. a 1-epoch smoke test)
    # instead of MultiStepLR raising/double-firing on a milestone of 0.
    milestones = sorted({m for m in (int(0.5 * epochs), int(0.75 * epochs)) if 0 < m < epochs})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    return optimizer, scheduler
