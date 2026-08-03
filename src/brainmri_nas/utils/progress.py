"""Periodic batch-progress logging shared by both training loops
(`training/engine.py` and `augment/trial_training.py`), so neither has to
reinvent "which batch counts as a checkpoint."

Emits roughly `target_updates_per_epoch` lines per epoch regardless of how
many batches an epoch actually has -- a dataset with 10 batches/epoch and
one with 300 batches/epoch both get proportionate visibility instead of one
line per batch (unusable spam) or silence for an entire epoch (which is
what both loops did before this existed).
"""

from __future__ import annotations

import logging


def batch_log_interval(total_batches: int, target_updates_per_epoch: int = 5) -> int:
    return max(1, total_batches // target_updates_per_epoch)


def maybe_log_batch_progress(
    logger: logging.Logger,
    *,
    prefix: str,
    epoch: int,
    total_epochs: int,
    batch_idx: int,
    total_batches: int,
    running_loss: float,
    interval: int,
) -> None:
    is_last_batch = (batch_idx + 1) == total_batches
    if (batch_idx + 1) % interval == 0 or is_last_batch:
        logger.info(
            "%s epoch %d/%d batch %d/%d (%.0f%%) running_loss=%.4f",
            prefix,
            epoch,
            total_epochs,
            batch_idx + 1,
            total_batches,
            100.0 * (batch_idx + 1) / total_batches,
            running_loss,
        )
