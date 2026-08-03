"""Optimizer/scheduler construction shared by augmentation-policy trial
training and final training.

SGD(momentum) + CosineAnnealingLR -- the standard DARTS training recipe
(lr=0.025, weight_decay=3e-4, momentum=0.9), not Adam. The legacy repo's
`lr=0.025` value is this exact SGD recipe's learning rate; it was originally
carried over unchanged when this project's training loop used Adam, which
is ~25x too high for Adam's own normal range and a likely cause of large
epoch-to-epoch validation swings seen in real training runs. Rather than
just rescale the learning rate for Adam, the optimizer itself was switched
to match the recipe those numbers actually belong to: on a small,
overfitting-prone dataset with a BatchNorm-heavy DARTS-style architecture,
SGD+momentum's flatter-minima tendency and undistorted weight decay (Adam's
non-decoupled weight decay gets rescaled unevenly per-parameter by its own
adaptive step normalization) both favor generalization over Adam's faster
but often sharper convergence. Centralized here, instead of duplicated in
both training loops, so they can't silently drift apart on this again.
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
    momentum: float = 0.9,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    return optimizer, scheduler
