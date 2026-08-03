from __future__ import annotations

import numpy as np
import pytest

from brainmri_nas.utils.loss_cache import NEUTRAL_RANK, LossCache


def test_refresh_interval_calibrated_to_40_cycles():
    assert LossCache(num_samples=10, total_epochs=200).refresh_interval == 5
    assert LossCache(num_samples=10, total_epochs=250).refresh_interval == 6
    # Short search trials scale down automatically rather than needing a special case.
    assert LossCache(num_samples=10, total_epochs=10).refresh_interval == 1


def test_refresh_interval_never_zero_even_for_tiny_runs():
    assert LossCache(num_samples=10, total_epochs=1).refresh_interval >= 1


def test_cold_start_returns_neutral_rank_for_everyone():
    cache = LossCache(num_samples=5, total_epochs=200)
    ranks = cache.get_loss_ranks()
    assert np.all(ranks == NEUTRAL_RANK)
    assert cache.has_data is False


def test_cache_does_not_refresh_before_interval_elapses():
    cache = LossCache(num_samples=3, total_epochs=200)  # interval = 5
    cache.record_batch([0, 1, 2], [1.0, 2.0, 3.0])
    for _ in range(4):  # only 4 of the 5 required epochs
        refreshed = cache.end_epoch()
        assert refreshed is False
    assert cache.has_data is False
    assert np.all(cache.get_loss_ranks() == NEUTRAL_RANK)


def test_cache_refreshes_with_windowed_average_not_a_snapshot():
    cache = LossCache(num_samples=2, total_epochs=200)  # interval = 5
    # Sample 0's loss trends 1.0 -> 5.0 across 5 epochs (avg = 3.0); sample 1 is constant at 10.0.
    for loss_0 in (1.0, 2.0, 3.0, 4.0, 5.0):
        cache.record_batch([0, 1], [loss_0, 10.0])
        refreshed = cache.end_epoch()
    assert refreshed is True
    assert cache.has_data is True
    assert cache._cached_loss[0] == pytest.approx(3.0)  # windowed average, not the last (5.0) or first (1.0)
    assert cache._cached_loss[1] == pytest.approx(10.0)


def test_pending_accumulation_resets_after_refresh():
    cache = LossCache(num_samples=1, total_epochs=1)  # interval = 1, refreshes every epoch
    cache.record_batch([0], [2.0])
    cache.end_epoch()
    assert cache._cached_loss[0] == pytest.approx(2.0)

    cache.record_batch([0], [8.0])
    cache.end_epoch()
    assert cache._cached_loss[0] == pytest.approx(8.0)  # not (2.0+8.0)/2 -- old window must not leak into the new one


def test_ranks_order_lowest_loss_to_zero_highest_to_one():
    cache = LossCache(num_samples=4, total_epochs=4)  # interval = 1
    cache.record_batch([0, 1, 2, 3], [5.0, 1.0, 3.0, 9.0])
    cache.end_epoch()

    ranks = cache.get_loss_ranks()
    assert ranks[1] == pytest.approx(0.0)  # loss=1.0, easiest
    assert ranks[3] == pytest.approx(1.0)  # loss=9.0, hardest
    assert ranks[2] < ranks[0]  # loss=3.0 ranks below loss=5.0


def test_samples_never_recorded_stay_neutral_after_a_refresh():
    cache = LossCache(num_samples=3, total_epochs=1)  # interval = 1
    cache.record_batch([0, 2], [1.0, 5.0])  # index 1 never recorded
    cache.end_epoch()

    ranks = cache.get_loss_ranks()
    assert ranks[1] == NEUTRAL_RANK
    assert ranks[0] != NEUTRAL_RANK
    assert ranks[2] != NEUTRAL_RANK


def test_rejects_non_positive_num_samples():
    with pytest.raises(ValueError):
        LossCache(num_samples=0, total_epochs=200)


def test_ranks_are_memoized_between_refreshes():
    cache = LossCache(num_samples=3, total_epochs=3)  # interval = 1
    cache.record_batch([0, 1, 2], [1.0, 2.0, 3.0])
    cache.end_epoch()

    ranks_a = cache.get_loss_ranks()
    ranks_b = cache.get_loss_ranks()
    assert ranks_a is ranks_b  # same object -- not recomputed on the second call

    # A real refresh must invalidate the memoized ranks, not silently keep stale ones.
    cache.record_batch([0, 1, 2], [9.0, 5.0, 1.0])  # reverse the ordering entirely
    cache.end_epoch()
    ranks_c = cache.get_loss_ranks()
    assert ranks_c is not ranks_a
    assert ranks_c[0] == pytest.approx(1.0)  # sample 0 is now the hardest, was easiest before
    assert ranks_c[2] == pytest.approx(0.0)  # sample 2 is now the easiest, was hardest before
