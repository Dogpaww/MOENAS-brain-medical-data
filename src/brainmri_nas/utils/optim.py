"""Optimizer/scheduler construction shared by augmentation-policy trial
training and final training.

Matches the legacy repo's `evaluate.py::train()` exactly: `Adam(lr=0.025,
weight_decay=3e-4)` + `MultiStepLR(milestones=[0.5*epochs, 0.75*epochs],
gamma=0.1)`. The handoff spec doesn't mandate a specific optimizer, and the
project's goal is to preserve the original methodology while fixing only
the bugs it explicitly calls out -- optimizer choice isn't one of them, so
this deliberately matches the original rather than substituting a different
(if more fashionable) default. Centralized here, instead of duplicated in
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
    gamma: float = 0.1,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Degrades to "no decay" for very short runs (e.g. a 1-epoch smoke test)
    # instead of MultiStepLR raising/double-firing on a milestone of 0.
    milestones = sorted({m for m in (int(0.5 * epochs), int(0.75 * epochs)) if 0 < m < epochs})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    return optimizer, scheduler
