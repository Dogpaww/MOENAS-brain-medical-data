"""Constant-strength control: the sample-adaptive policy's operators and
probabilities, each held at its rank-averaged strength for every sample."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn

import brainmri_nas.baselines.train_baseline as train_baseline
from brainmri_nas.augment.genotype import AugmentationPolicy, AugmentationStep
from brainmri_nas.augment.search_space import (
    MAGNITUDE_RANGES,
    chromosome_length,
    decode_chromosome,
    rank_averaged_strengths,
    resolve_step,
    resolve_step_at_strength,
    sap_strength,
)
from brainmri_nas.augment.transform_builder import build_constant_strength_transform
from brainmri_nas.baselines import registry
from brainmri_nas.data.loader import build_dataset_bundle
from brainmri_nas.search.nsga2_runner import run_search
from brainmri_nas.training.checkpoint import load_checkpoint
from brainmri_nas.training.final_training import run_final_training
from brainmri_nas.utils.config import (
    Config,
    DatasetConfig,
    NSGA2Config,
    ProxyConfig,
    SearchSpaceConfig,
    TrainingConfig,
)

IMAGE_SIZE = 16


def _policy() -> AugmentationPolicy:
    chromosome = [((i * 53) % 101) / 101.0 for i in range(chromosome_length())]
    return decode_chromosome(chromosome)


def _policy_file(tmp_path: Path) -> Path:
    chromosome = [((i * 53) % 101) / 101.0 for i in range(chromosome_length())]
    path = tmp_path / "selected_policy.json"
    path.write_text(json.dumps({"chromosome": chromosome, "policy": decode_chromosome(chromosome).to_dict()}))
    return path


def test_rank_average_matches_the_mean_over_the_loss_cache_rank_grid():
    policy = _policy()
    n = 101
    strengths = rank_averaged_strengths(policy, n)
    for step in policy.ordered_steps():
        expected = np.mean([sap_strength(step.strength_s, step.strength_a, i / (n - 1)) for i in range(n)])
        assert strengths[step.name] == pytest.approx(expected)


def test_a_symmetric_curve_averages_to_one_half():
    step = AugmentationStep(name="rotation", order=0, probability=1.0, strength_s=10.0, strength_a=0.5)
    strengths = rank_averaged_strengths(AugmentationPolicy(steps=(step,)), 1001)
    assert strengths["rotation"] == pytest.approx(0.5, abs=1e-3)


def test_adaptive_resolution_is_unchanged_by_the_refactor():
    for step in _policy().ordered_steps():
        for rank in (0.0, 0.3, 0.7, 1.0):
            by_rank = resolve_step(step, rank)
            by_strength = resolve_step_at_strength(step, sap_strength(step.strength_s, step.strength_a, rank))
            assert by_rank == by_strength


def test_constant_transform_keeps_probabilities_and_uses_the_given_strength():
    policy = _policy()
    strengths = {step.name: 0.4 for step in policy.ordered_steps()}
    transform = build_constant_strength_transform(policy, IMAGE_SIZE, strengths)
    ops = {type(t.transforms[0]).__name__ if hasattr(t, "transforms") else type(t).__name__: t for t in transform.transforms}

    rotation = ops["RandomRotation"]
    rotation_step = next(s for s in policy.ordered_steps() if s.name == "rotation")
    assert rotation.p == pytest.approx(rotation_step.probability)
    assert rotation.transforms[0].degrees == pytest.approx([-0.4 * MAGNITUDE_RANGES["rotation"][1], 0.4 * MAGNITUDE_RANGES["rotation"][1]])
    assert "RandomErasing" in ops  # tensor-space op still placed after normalization


def _tiny_baseline(num_classes: int) -> nn.Module:
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, num_classes))


def test_baseline_constant_strength_arm_uses_one_transform_and_no_loss_cache(
    synthetic_dataset_root: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.setitem(registry.MODEL_REGISTRY, "tiny", _tiny_baseline)
    calls = []
    real = train_baseline.train_one_epoch
    monkeypatch.setattr(
        train_baseline,
        "train_one_epoch",
        lambda model, loader, optimizer, **kw: (calls.append((loader, kw["loss_cache"])), real(model, loader, optimizer, **kw))[1],
    )
    split_path = tmp_path / "split_indices.json"
    build_dataset_bundle(synthetic_dataset_root, image_size=IMAGE_SIZE, validation_fraction=0.2, split_seed=1, batch_size=4, split_indices_path=split_path)
    policy_path = _policy_file(tmp_path)
    output_dir = tmp_path / "baseline_constant"

    train_baseline.run_baseline_training(
        Config(dataset=DatasetConfig(data_root=str(synthetic_dataset_root), image_size=IMAGE_SIZE, batch_size=4, num_workers=0), training=TrainingConfig(device="cpu")),
        model_name="tiny",
        split_indices_path=split_path,
        output_dir=output_dir,
        image_size=IMAGE_SIZE,
        epochs=2,
        constant_strength_policy_path=policy_path,
    )

    assert all(cache is None for _, cache in calls)
    loader = calls[0][0]
    assert len(next(iter(loader))) == 2  # (x, y): not sample-adaptive
    assert "RandomRotation" in repr(loader.dataset.dataset.transform)

    config = json.loads((output_dir / "baseline_config.json").read_text())
    assert config["augmentation"] == "constant_strength"
    assert config["constant_strength_policy_path"] == str(policy_path)
    record = json.loads((output_dir / "constant_strength_policy.json").read_text())
    num_train = len(json.loads(split_path.read_text())["train_indices"])
    assert record["strengths"] == pytest.approx(rank_averaged_strengths(_policy(), num_train))
    assert not (output_dir / "selected_policy.json").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"selected_policy_path": "a.json", "constant_strength_policy_path": "b.json"},
        {"selected_legacy_policy_path": "a.json", "constant_strength_policy_path": "b.json"},
    ],
)
def test_constant_strength_is_exclusive_with_the_other_policies(tmp_path: Path, kwargs):
    config = Config(dataset=DatasetConfig(data_root=str(tmp_path)), training=TrainingConfig(device="cpu"))
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_baseline.run_baseline_training(config, model_name="tiny", split_indices_path=tmp_path / "s.json", output_dir=tmp_path / "o", **kwargs)
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_final_training(config, selected_architecture_path=tmp_path / "a.json", split_indices_path=tmp_path / "s.json", output_dir=tmp_path / "o", **kwargs)


def test_final_training_constant_strength_arm_records_the_strengths(synthetic_dataset_root: Path, tmp_path: Path):
    dataset = DatasetConfig(data_root=str(synthetic_dataset_root), image_size=IMAGE_SIZE, batch_size=4, num_workers=0)
    search_space = SearchSpaceConfig(initial_channels_min=4, initial_channels_max=4, number_of_cells_min=3, number_of_cells_max=3)
    search_dir = tmp_path / "search_run"
    run_search(
        Config(dataset=dataset, search_space=search_space, proxies=ProxyConfig(zico_batch_size=2, zico_num_batches=2), nsga2=NSGA2Config(population_size=4, num_generations=1, seed=1, device="cpu")),
        search_dir,
    )
    training_dir = tmp_path / "training_run"
    run_final_training(
        Config(dataset=dataset, search_space=search_space, training=TrainingConfig(physical_batch_size=4, final_epochs=1, precision="fp32", device="cpu")),
        selected_architecture_path=search_dir / "selected_architecture.json",
        split_indices_path=search_dir / "split_indices.json",
        constant_strength_policy_path=_policy_file(tmp_path),
        output_dir=training_dir,
    )

    record = json.loads((training_dir / "constant_strength_policy.json").read_text())
    payload = load_checkpoint(training_dir / "best_checkpoint.pt")
    assert {s["name"]: s["constant_strength"] for s in payload["augmentation_policy"]} == pytest.approx(record["strengths"])
    assert "constant strength" in (training_dir / "training.log").read_text()
    assert not (training_dir / "selected_policy.json").exists()
