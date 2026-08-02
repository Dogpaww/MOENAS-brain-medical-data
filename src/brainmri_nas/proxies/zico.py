"""ZiCO zero-cost proxy (handoff §18).

Loss: CrossEntropyLoss, matching the project's single-label 4-class task
everywhere else. Only `nn.Conv2d`/`nn.Linear` weight gradients are collected
(same layer scope as SynFlow). Model runs in `.train()` mode so BatchNorm
sees per-batch statistics, matching how gradients would actually look during
real training. `batches` must be the same fixed set for every candidate in a
search run (handoff §19) -- this function doesn't select them, it just
consumes whatever list of (x, y) tensors it's given.

Per-layer score: for each Conv2d/Linear weight, gather one gradient per
batch, then `log(sum(mean_abs / std))` over parameter entries whose std
across batches is meaningfully nonzero (an `eps` threshold rather than exact
`!= 0`, so near-constant-but-nonzero gradients can't blow the ratio up).
Final score is the sum of finite, positive per-layer scores.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

STD_EPSILON = 1e-12


def _aggregate_zico(grad_history: dict[str, list[np.ndarray]]) -> float:
    total = 0.0
    for grads in grad_history.values():
        if len(grads) < 2:
            continue  # need >= 2 batches to have a meaningful std

        grad_array = np.stack(grads, axis=0)
        std = np.std(grad_array, axis=0)
        mean_abs = np.mean(np.abs(grad_array), axis=0)

        significant = std > STD_EPSILON
        if not np.any(significant):
            continue

        layer_sum = float(np.sum(mean_abs[significant] / std[significant]))
        if layer_sum > 0 and math.isfinite(layer_sum):
            total += math.log(layer_sum)

    return total if math.isfinite(total) else 0.0


def compute_zico(
    model: nn.Module,
    *,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict:
    if len(batches) < 2:
        raise ValueError(f"ZiCO needs at least 2 batches to compute a gradient std, got {len(batches)}.")

    was_training = model.training
    model.to(device)
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    grad_history: dict[str, list[np.ndarray]] = {}

    try:
        for x, y in batches:
            x = x.to(device)
            y = y.to(device)

            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight.grad is not None:
                    grad = module.weight.grad.detach().to("cpu").reshape(-1).numpy()
                    grad_history.setdefault(name, []).append(grad)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    return {"zico": _aggregate_zico(grad_history)}
