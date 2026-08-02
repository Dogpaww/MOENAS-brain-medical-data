"""One training epoch, with gradient accumulation and CUDA-only AMP
(handoff §26-27).

AMP is only ever actually engaged when `use_amp` is True, which callers
should only set when the resolved device is CUDA -- this module doesn't
decide that itself, it just does what it's told, so the "AMP only really
runs on CUDA" decision stays visible in one place (`final_training.py`).

Gradient accumulation: the optimizer only steps once every
`accumulation_steps` micro-batches, *and* on the final micro-batch of the
epoch regardless of where that falls in the accumulation cycle -- otherwise
a trailing partial group's gradients would silently be dropped. AMP
gradients are unscaled before clipping (handoff §26); a per-epoch LR
scheduler is stepped by the caller after this returns, which is also why
"advance step-based schedulers only with optimizer updates" doesn't need
special handling here -- there's no per-iteration scheduler in this design.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    use_amp: bool,
    scaler: "torch.amp.GradScaler | None",
    accumulation_steps: int,
    grad_clip_norm: float,
    loss_fn: nn.Module | None = None,
) -> float:
    if use_amp and scaler is None:
        raise ValueError("use_amp=True requires a GradScaler.")

    model.train()
    loss_fn = loss_fn or nn.CrossEntropyLoss()

    num_batches = len(train_loader)
    total_loss = 0.0
    total_samples = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        scaled_loss = loss / accumulation_steps
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        is_flush_step = ((step + 1) % accumulation_steps == 0) or ((step + 1) == num_batches)
        if is_flush_step:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / max(total_samples, 1)
