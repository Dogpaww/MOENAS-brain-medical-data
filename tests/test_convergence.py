"""The pre-registered convergence rule: window arithmetic, thresholds, and
seed averaging per augmentation arm."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from brainmri_nas.training.convergence import describe_run, flag_models, is_flagged, late_improvement


def _rows(accuracies: list[float], aucs: list[float]) -> list[dict]:
    return [{"val_accuracy": a, "val_macro_auc": u} for a, u in zip(accuracies, aucs)]


def test_late_improvement_compares_the_last_fifth_with_the_fifth_before_it():
    # 50 epochs -> windows are epochs 31-40 and 41-50.
    accuracies = [0.80] * 30 + [0.90] * 10 + [0.92] * 10
    aucs = [0.95] * 30 + [0.97] * 10 + [0.975] * 10
    accuracy_rise, auc_rise = late_improvement(_rows(accuracies, aucs))
    assert accuracy_rise == pytest.approx(2.0)
    assert auc_rise == pytest.approx(0.5)


def test_thresholds_are_strict_and_either_metric_can_flag():
    assert not is_flagged(0.5, 0.2)
    assert is_flagged(0.51, 0.0)
    assert is_flagged(0.0, 0.21)


def _write_run(root: Path, name: str, *, augmentation: str | None, seed: int, late_accuracy: float) -> Path:
    run = root / name
    run.mkdir()
    config = {"model_name": "deit_small", "epochs": 50, "image_size": 96, "seed": seed}
    if augmentation is not None:
        config["augmentation"] = augmentation
    (run / "baseline_config.json").write_text(json.dumps(config))
    accuracies = [0.90] * 40 + [late_accuracy] * 10
    with open(run / "training_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "val_accuracy", "val_macro_auc"])
        writer.writeheader()
        for epoch, accuracy in enumerate(accuracies, start=1):
            writer.writerow({"epoch": epoch, "val_accuracy": accuracy, "val_macro_auc": 0.95})
    return run


def test_model_flag_uses_the_seed_average_not_any_single_seed(tmp_path: Path):
    runs = [
        # Adaptive arm: one seed rises 1.0 pt, the other 0.0 -> average 0.5, not above threshold.
        describe_run(_write_run(tmp_path, "a1", augmentation="sample_adaptive", seed=1, late_accuracy=0.91)),
        describe_run(_write_run(tmp_path, "a2", augmentation="sample_adaptive", seed=2, late_accuracy=0.90)),
        # Pre-flag run with no "augmentation" field counts as the none arm.
        describe_run(_write_run(tmp_path, "n1", augmentation=None, seed=1, late_accuracy=0.90)),
    ]
    assert runs[0]["flagged"] and not runs[1]["flagged"]

    (model,) = flag_models(runs)
    assert model["optimizer_name"] == "sgd"
    assert model["arms"]["sample_adaptive"]["num_seeds"] == 2
    assert model["arms"]["sample_adaptive"]["accuracy_rise"] == pytest.approx(0.5)
    assert model["arms"]["none"]["num_seeds"] == 1
    assert not model["flagged"]


def test_model_is_flagged_when_either_arm_average_exceeds_the_threshold(tmp_path: Path):
    runs = [
        describe_run(_write_run(tmp_path, "a1", augmentation="sample_adaptive", seed=1, late_accuracy=0.92)),
        describe_run(_write_run(tmp_path, "n1", augmentation="none", seed=1, late_accuracy=0.90)),
    ]
    (model,) = flag_models(runs)
    assert model["arms"]["sample_adaptive"]["flagged"]
    assert not model["arms"]["none"]["flagged"]
    assert model["flagged"]


def test_two_runs_with_the_same_seed_are_refused_not_averaged(tmp_path: Path):
    # An original run (no "augmentation" field) and its rerun with the same seed.
    runs = [
        describe_run(_write_run(tmp_path, "old", augmentation=None, seed=1, late_accuracy=0.90)),
        describe_run(_write_run(tmp_path, "rerun", augmentation="none", seed=1, late_accuracy=0.90)),
    ]
    with pytest.raises(ValueError, match="same seed"):
        flag_models(runs)
