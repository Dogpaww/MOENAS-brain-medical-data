"""Baselines trained under the searched architecture's sample-adaptive policy.

Without a policy a baseline sees only resize + grayscale + normalize, while
the searched architecture trains under its selected SapAugment policy, so the
two differ in augmentation as well as architecture. These tests pin down that
`selected_policy_path` routes a baseline through the same sample-adaptive
loader and LossCache as final training, and that omitting it leaves the
unaugmented path exactly as it was.

A tiny stand-in model is registered for the duration of each test so nothing
downloads pretrained ImageNet weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn

import brainmri_nas.baselines.train_baseline as train_baseline
from brainmri_nas.augment.search_space import chromosome_length, decode_chromosome
from brainmri_nas.baselines import registry
from brainmri_nas.data.loader import build_dataset_bundle
from brainmri_nas.utils.config import Config, DatasetConfig, TrainingConfig
from brainmri_nas.utils.loss_cache import NEUTRAL_RANK, LossCache

IMAGE_SIZE = 16
EPOCHS = 2


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


@pytest.fixture()
def training_calls(monkeypatch):
    """Records what each epoch's train_one_epoch call was given, then runs it for real."""
    calls = []
    real = train_baseline.train_one_epoch

    def spy(model, train_loader, optimizer, **kwargs):
        calls.append({"loader": train_loader, "loss_cache": kwargs["loss_cache"]})
        return real(model, train_loader, optimizer, **kwargs)

    monkeypatch.setattr(train_baseline, "train_one_epoch", spy)
    return calls


def _config(data_root: Path) -> Config:
    return Config(
        dataset=DatasetConfig(data_root=str(data_root), image_size=IMAGE_SIZE, batch_size=4, num_workers=0),
        training=TrainingConfig(device="cpu"),
    )


def _saved_split(data_root: Path, tmp_path: Path) -> Path:
    split_path = tmp_path / "split_indices.json"
    build_dataset_bundle(
        data_root,
        image_size=IMAGE_SIZE,
        validation_fraction=0.2,
        split_seed=1,
        batch_size=4,
        split_indices_path=split_path,
    )
    return split_path


def _policy_file(tmp_path: Path) -> Path:
    chromosome = [((i * 53) % 101) / 101.0 for i in range(chromosome_length())]
    record = {"chromosome": chromosome, "policy": decode_chromosome(chromosome).to_dict(), "val_macro_auc": 0.5}
    path = tmp_path / "selected_policy.json"
    path.write_text(json.dumps(record))
    return path


def test_policy_routes_baseline_through_sample_adaptive_loader_and_loss_cache(
    synthetic_dataset_root: Path, tmp_path: Path, tiny_registry, training_calls
):
    policy_path = _policy_file(tmp_path)
    output_dir = tmp_path / "baseline_aug"

    train_baseline.run_baseline_training(
        _config(synthetic_dataset_root),
        model_name="tiny",
        split_indices_path=_saved_split(synthetic_dataset_root, tmp_path),
        output_dir=output_dir,
        image_size=IMAGE_SIZE,
        epochs=EPOCHS,
        selected_policy_path=policy_path,
    )

    assert len(training_calls) == EPOCHS
    cache = training_calls[0]["loss_cache"]
    assert isinstance(cache, LossCache)
    # One cache for the whole run -- a fresh one per epoch would never accumulate ranks.
    assert all(call["loss_cache"] is cache for call in training_calls)

    x, y, indices = next(iter(training_calls[0]["loader"]))
    assert x.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert indices.shape == y.shape

    # 2 epochs -> refresh_interval 1, so real per-sample ranks replaced the cold-start value.
    assert not np.all(cache.get_loss_ranks() == NEUTRAL_RANK)

    baseline_config = json.loads((output_dir / "baseline_config.json").read_text())
    assert baseline_config["selected_policy_path"] == str(policy_path)
    assert json.loads((output_dir / "selected_policy.json").read_text()) == json.loads(policy_path.read_text())


def test_no_policy_keeps_the_unaugmented_baseline_path(
    synthetic_dataset_root: Path, tmp_path: Path, tiny_registry, training_calls
):
    output_dir = tmp_path / "baseline_plain"

    train_baseline.run_baseline_training(
        _config(synthetic_dataset_root),
        model_name="tiny",
        split_indices_path=_saved_split(synthetic_dataset_root, tmp_path),
        output_dir=output_dir,
        image_size=IMAGE_SIZE,
        epochs=EPOCHS,
    )

    assert all(call["loss_cache"] is None for call in training_calls)
    batch = next(iter(training_calls[0]["loader"]))
    assert len(batch) == 2  # (x, y) pairs, no sample index

    baseline_config = json.loads((output_dir / "baseline_config.json").read_text())
    assert baseline_config["selected_policy_path"] is None
    assert not (output_dir / "selected_policy.json").exists()
