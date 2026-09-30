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

`optimizer_name="sgd"` is the default and what every baseline uses, including
the DeiT-Small transformer baseline. "adamw" is kept only as an explicit,
labeled opt-in (run_baseline.py's --optimizer flag).

A correction, since this docstring previously argued the opposite for DeiT.
It claimed SGD is a documented mismatch for fine-tuning vision transformers,
citing a DeiT-Small run whose train_loss "froze" around 0.2916. That was a
misreading on two counts. First, 0.2916 is the label-smoothing floor: with
smoothing 0.1 over 3 classes the lowest achievable cross-entropy is 0.2911,
and the CNN baselines settle at the same value -- the model had fit the
training set, not stalled. Second, the DeiT paper does not support the claim:
it fine-tunes "with either AdamW or SGD. These optimizers have a similar
performance for the fine-tuning stage" (Touvron et al. 2021, Sec. 6; Table 8
shows SGD fine-tuning matching AdamW, 83.1 vs 83.1). SGD only hurts DeiT
during pre-training from scratch, which no baseline here does.
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
    optimizer_name: str = "sgd",
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer_name {optimizer_name!r}. Expected 'sgd' or 'adamw'.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    return optimizer, scheduler
