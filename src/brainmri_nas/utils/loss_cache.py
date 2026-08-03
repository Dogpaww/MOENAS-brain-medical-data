"""Per-sample loss cache for sample-adaptive augmentation (SapAugment, Hu et
al. 2021 -- "learning a sample-adaptive policy for data augmentation").

Two deliberate departures from taking the current epoch's raw per-sample
loss at face value, both discussed and agreed with the user before
implementing:

1. **Refresh cadence is a fraction of the run, not a fixed epoch count.**
   Refreshing every epoch is too noisy to be a trustworthy "this sample is
   hard" signal -- besides the well-known early-training problem (an
   untrained model makes every sample look equally hard), per-epoch loss
   is also noisy epoch-to-epoch from batch composition and the model being
   a moving target. `target_refresh_cycles=40` calibrates the interval so
   a 200-epoch run refreshes roughly every 5 epochs (the reference point
   the user gave); a short 10-epoch search trial scales down to every
   epoch automatically, rather than needing a special case.
2. **The cache stores the windowed *average* loss, not a single-epoch
   snapshot** -- averaging over the just-completed window suppresses the
   same epoch-to-epoch noise for roughly free, since the per-batch losses
   are already being computed regardless.

Ranks (not raw loss values) are what callers actually consume, exactly as
in the paper -- normalizing to a relative rank keeps the signal comparable
across training stages, since a rank of "hardest 10%" means the same thing
whether the model's absolute loss scale is high (early training) or low
(late training).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_TARGET_REFRESH_CYCLES = 40
NEUTRAL_RANK = 0.5  # used before the cache has any data at all (cold start)


class LossCache:
    def __init__(
        self,
        num_samples: int,
        total_epochs: int,
        target_refresh_cycles: int = DEFAULT_TARGET_REFRESH_CYCLES,
    ):
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        self.num_samples = num_samples
        self.refresh_interval = max(1, round(total_epochs / target_refresh_cycles))

        self._pending_sum = np.zeros(num_samples, dtype=np.float64)
        self._pending_count = np.zeros(num_samples, dtype=np.int64)
        self._cached_loss = np.full(num_samples, np.nan, dtype=np.float64)
        self._epochs_since_refresh = 0
        self.has_data = False
        self._ranks_cache: np.ndarray | None = None  # memoized get_loss_ranks(); see there for why

    def record_batch(self, indices: Sequence[int], losses: Sequence[float]) -> None:
        idx = np.asarray(indices)
        self._pending_sum[idx] += np.asarray(losses, dtype=np.float64)
        self._pending_count[idx] += 1

    def end_epoch(self) -> bool:
        """Call once per epoch, after all of that epoch's batches. Returns
        True if this call actually refreshed the cache (i.e. `refresh_interval`
        epochs' worth of data had accumulated)."""
        self._epochs_since_refresh += 1
        if self._epochs_since_refresh < self.refresh_interval:
            return False

        seen = self._pending_count > 0
        if np.any(seen):
            self._cached_loss[seen] = self._pending_sum[seen] / self._pending_count[seen]
            self.has_data = True
            self._ranks_cache = None  # invalidate -- only recomputed lazily, on next get_loss_ranks() call

        self._pending_sum[:] = 0.0
        self._pending_count[:] = 0
        self._epochs_since_refresh = 0
        return True

    def get_loss_ranks(self) -> np.ndarray:
        """Per-sample rank normalized to [0, 1]: 0 = lowest cached loss
        (easiest), 1 = highest cached loss (hardest). Samples with no cached
        loss yet (cold start, or never seen in a completed window) get the
        neutral rank so they aren't arbitrarily treated as easy or hard.

        Memoized between refreshes: a `Dataset.__getitem__` calls this once
        per sample *read* (thousands of times per epoch), so recomputing a
        full argsort every call -- rather than once per actual refresh --
        would be needless, real overhead.
        """
        if self._ranks_cache is not None:
            return self._ranks_cache

        ranks = np.full(self.num_samples, NEUTRAL_RANK, dtype=np.float64)
        valid = ~np.isnan(self._cached_loss)
        num_valid = int(np.sum(valid))
        if num_valid == 0:
            self._ranks_cache = ranks
            return ranks

        order = np.argsort(self._cached_loss[valid])
        valid_ranks = np.empty(num_valid, dtype=np.float64)
        valid_ranks[order] = np.arange(num_valid, dtype=np.float64) / max(1, num_valid - 1)
        ranks[valid] = valid_ranks
        self._ranks_cache = ranks
        return ranks
