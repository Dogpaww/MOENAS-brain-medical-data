from __future__ import annotations

import logging

from brainmri_nas.utils.progress import batch_log_interval, maybe_log_batch_progress


def test_batch_log_interval_scales_with_batch_count():
    assert batch_log_interval(5) == 1  # max(1, 5 // 5)
    assert batch_log_interval(100) == 20  # 100 // 5
    assert batch_log_interval(1) == 1  # never zero, would break "% interval"


def test_maybe_log_batch_progress_fires_at_interval_and_on_last_batch(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test.progress.cadence")

    total_batches = 10
    interval = batch_log_interval(total_batches)  # 10 // 5 == 2
    fired_batches = []

    for batch_idx in range(total_batches):
        caplog.clear()
        maybe_log_batch_progress(
            logger,
            prefix="test",
            epoch=1,
            total_epochs=1,
            batch_idx=batch_idx,
            total_batches=total_batches,
            running_loss=0.5,
            interval=interval,
        )
        if caplog.records:
            fired_batches.append(batch_idx)

    # (batch_idx + 1) % 2 == 0 -> fires at 1,3,5,7,9 (0-indexed); 9 is also the last batch.
    assert fired_batches == [1, 3, 5, 7, 9]


def test_maybe_log_batch_progress_message_content(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test.progress.content")

    maybe_log_batch_progress(
        logger,
        prefix="train",
        epoch=3,
        total_epochs=10,
        batch_idx=9,
        total_batches=10,
        running_loss=0.6789,
        interval=1,
    )

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "train" in message
    assert "3/10" in message
    assert "10/10" in message
    assert "0.6789" in message
