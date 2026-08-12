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

Batch progress is logged through the same named logger `final_training.py`
already attaches handlers to, so it needs no extra plumbing to reach both
the console and `training.log`; called from a bare unit test with no
handler configured, these log calls are simply silent no-ops.

`loss_cache`, when given, makes this sample-adaptive (SapAugment, Hu et al.
2021) -- see `augment/trial_training.py`'s docstring for the fuller
rationale, which applies identically here. `train_loader` must then yield
`(x, y, index)` triples instead of `(x, y)` pairs. Passing `loss_cache=None`
preserves plain 2-tuple-batch behavior, used both for final training
without a selected augmentation policy and by tests that don't exercise
sample-adaptivity at all.

`label_smoothing` (>0) trains against softened targets instead of hard
one-hot ones, which puts a floor under how low cross-entropy can go and so
keeps regularizing at full strength for the whole run -- unlike weight
decay, whose per-step pull is `lr * weight_decay * w` and therefore fades
out along with CosineAnnealingLR exactly when late-epoch overfitting is
worst (real 250-epoch runs bottom out around train_loss=0.04 despite a
raised weight decay).

It is applied to the *optimized* loss only, never to what LossCache
records, for the same reason the cancer penalty below is: smoothing
penalizes confident predictions, which does not preserve the difficulty
*ordering* LossCache ranks on. Concretely, a correct p_y=0.99 sample
(plain CE 0.0101) and a correct p_y=0.90 one (plain CE 0.1054) swap places
under smoothing (0.4423 vs 0.3578) -- so the genuinely easiest samples
would be misread as hardest and handed mild augmentation instead of
strong, inverting SapAugment for exactly the samples it most wants to push.

A `cancer_no_tumor_penalty` term and balanced `class_weights` were tried
here and removed -- see the git history for the rationale and the evidence
that retired them. Both assumed a `no_tumor` class, which the figshare
dataset does not have.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brainmri_nas.utils.loss_cache import LossCache
from brainmri_nas.utils.progress import batch_log_interval, maybe_log_batch_progress

DEFAULT_LOGGER_NAME = "brainmri_nas.final_training"


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
    epoch: int = 1,
    total_epochs: int = 1,
    loss_cache: LossCache | None = None,
    label_smoothing: float = 0.0,
    logger: logging.Logger | None = None,
) -> float:
    if use_amp and scaler is None:
        raise ValueError("use_amp=True requires a GradScaler.")

    logger = logger or logging.getLogger(DEFAULT_LOGGER_NAME)

    model.train()
    loss_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)
    # Unsmoothed twin, used only to feed LossCache a difficulty signal whose
    # ordering smoothing would otherwise scramble (see module docstring).
    # `is loss_fn` when smoothing is off, so the common path stays one CE call.
    cache_loss_fn = nn.CrossEntropyLoss(reduction="none") if label_smoothing > 0.0 else loss_fn

    num_batches = len(train_loader)
    log_interval = batch_log_interval(num_batches)
    total_loss = 0.0
    total_samples = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader):
        if loss_cache is not None:
            x, y, indices = batch
        else:
            x, y = batch
            indices = None

        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            per_sample_loss = loss_fn(logits, y)
            loss = per_sample_loss.mean()

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

        if loss_cache is not None:
            if cache_loss_fn is loss_fn:
                cached_loss = per_sample_loss.detach()
            else:
                with torch.no_grad():
                    # .float(): logits may be half under autocast, and this runs outside it.
                    cached_loss = cache_loss_fn(logits.detach().float(), y)
            loss_cache.record_batch(indices.tolist(), cached_loss.cpu().tolist())

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        maybe_log_batch_progress(
            logger,
            prefix="train",
            epoch=epoch,
            total_epochs=total_epochs,
            batch_idx=step,
            total_batches=num_batches,
            running_loss=total_loss / total_samples,
            interval=log_interval,
        )

    if loss_cache is not None:
        loss_cache.end_epoch()

    return total_loss / max(total_samples, 1)
