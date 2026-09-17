"""Per-sample test predictions: written by baseline training, and recreated
from a checkpoint only when the recomputed metrics match the saved ones."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

import brainmri_nas.baselines.train_baseline as train_baseline
from brainmri_nas.baselines import registry
from brainmri_nas.baselines.predictions import PREDICTIONS_FILENAME, backfill_run_predictions
from brainmri_nas.data.loader import build_dataset_bundle
from brainmri_nas.utils.config import Config, DatasetConfig, TrainingConfig

IMAGE_SIZE = 16


def _tiny_baseline(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, num_classes),
    )


@pytest.fixture()
def tiny_registry(monkeypatch):
    monkeypatch.setitem(registry.MODEL_REGISTRY, "tiny", _tiny_baseline)


def _config(data_root: Path) -> Config:
    return Config(
        dataset=DatasetConfig(data_root=str(data_root), image_size=IMAGE_SIZE, batch_size=4, num_workers=0),
        training=TrainingConfig(device="cpu"),
    )


def _trained_run(data_root: Path, tmp_path: Path) -> Path:
    split_path = tmp_path / "split_indices.json"
    build_dataset_bundle(
        data_root, image_size=IMAGE_SIZE, validation_fraction=0.2, split_seed=1, batch_size=4, split_indices_path=split_path
    )
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "split_indices.json").write_text(split_path.read_text())
    train_baseline.run_baseline_training(
        _config(data_root),
        model_name="tiny",
        split_indices_path=output_dir / "split_indices.json",
        output_dir=output_dir,
        image_size=IMAGE_SIZE,
        epochs=2,
    )
    return output_dir


def test_training_writes_one_prediction_row_per_test_image(synthetic_dataset_root: Path, tmp_path: Path, tiny_registry):
    run = _trained_run(synthetic_dataset_root, tmp_path)

    predictions = pd.read_csv(run / PREDICTIONS_FILENAME)
    metrics = json.loads((run / "test_metrics.json").read_text())
    assert len(predictions) == metrics["num_samples"]

    # Rows are in dataset order, paths are relative to the data root, and they agree with the saved aggregates.
    test_files = sorted(p.relative_to(synthetic_dataset_root).as_posix() for p in (synthetic_dataset_root / "Testing").rglob("*.png"))
    assert predictions["image"].tolist() == test_files
    assert predictions["true_label"].tolist() == [Path(f).parent.name for f in test_files]
    assert predictions["correct"].mean() == pytest.approx(metrics["accuracy"])
    probability_columns = [c for c in predictions.columns if c.startswith("prob_")]
    assert len(probability_columns) == len(metrics["per_class"]["support"])
    assert predictions[probability_columns].sum(axis=1).to_numpy() == pytest.approx(1.0, abs=1e-5)


def test_backfill_recreates_the_same_predictions_from_the_checkpoint(
    synthetic_dataset_root: Path, tmp_path: Path, tiny_registry
):
    run = _trained_run(synthetic_dataset_root, tmp_path)
    original = pd.read_csv(run / PREDICTIONS_FILENAME)
    (run / PREDICTIONS_FILENAME).unlink()

    result = backfill_run_predictions(run, _config(synthetic_dataset_root), torch.device("cpu"))

    assert result["status"] == "written"
    recreated = pd.read_csv(run / PREDICTIONS_FILENAME)
    pd.testing.assert_frame_equal(recreated, original)


def test_backfill_skips_runs_that_already_have_predictions(synthetic_dataset_root: Path, tmp_path: Path, tiny_registry):
    run = _trained_run(synthetic_dataset_root, tmp_path)
    result = backfill_run_predictions(run, _config(synthetic_dataset_root), torch.device("cpu"))
    assert result["status"] == "skipped"


def test_backfill_writes_nothing_when_saved_metrics_do_not_reproduce(
    synthetic_dataset_root: Path, tmp_path: Path, tiny_registry
):
    run = _trained_run(synthetic_dataset_root, tmp_path)
    (run / PREDICTIONS_FILENAME).unlink()
    metrics_path = run / "test_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    # Move one image between cells, as if the checkpoint came from different weights.
    cm = metrics["confusion_matrix"]
    row = next(i for i, r in enumerate(cm) if sum(r) > 0)
    col = next(j for j, v in enumerate(cm[row]) if v > 0)
    cm[row][col] -= 1
    cm[row][(col + 1) % len(cm[row])] += 1
    metrics_path.write_text(json.dumps(metrics))

    result = backfill_run_predictions(run, _config(synthetic_dataset_root), torch.device("cpu"))

    assert result["status"] == "mismatch"
    assert "confusion matrix" in result["reason"]
    assert not (run / PREDICTIONS_FILENAME).exists()


def test_backfill_skips_runs_without_a_checkpoint(synthetic_dataset_root: Path, tmp_path: Path, tiny_registry):
    run = _trained_run(synthetic_dataset_root, tmp_path)
    (run / PREDICTIONS_FILENAME).unlink()
    (run / "best_checkpoint.pt").unlink()

    result = backfill_run_predictions(run, _config(synthetic_dataset_root), torch.device("cpu"))

    assert result["status"] == "skipped"
    assert "best_checkpoint.pt" in result["reason"]
