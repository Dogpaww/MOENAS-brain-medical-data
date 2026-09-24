"""Per-run magnitude bounds: validation, effect on resolved transforms, and
what a training run records about the bounds it used."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brainmri_nas.augment.genotype import AugmentationPolicy, AugmentationStep
from brainmri_nas.augment.search_space import (
    MAGNITUDE_RANGES,
    chromosome_length,
    decode_chromosome,
    resolve_bounds,
    resolve_step,
    resolve_step_at_strength,
)
from brainmri_nas.augment.transform_builder import build_constant_strength_transform, build_sample_adaptive_transform
from brainmri_nas.search.nsga2_runner import run_search
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
WIDE = {"rotation": [0.0, 45.0], "affine_translation": [0.0, 0.16], "random_resized_crop": [0.0, 0.35]}


def _policy() -> AugmentationPolicy:
    return decode_chromosome([((i * 53) % 101) / 101.0 for i in range(chromosome_length())])


def test_defaults_are_used_when_nothing_is_overridden():
    assert resolve_bounds(None) == MAGNITUDE_RANGES
    assert resolve_bounds({})["rotation"] == MAGNITUDE_RANGES["rotation"]


def test_overrides_replace_only_the_named_operators():
    bounds = resolve_bounds(WIDE)
    assert bounds["rotation"] == (0.0, 45.0)
    assert bounds["affine_translation"] == (0.0, 0.16)
    assert bounds["brightness"] == MAGNITUDE_RANGES["brightness"]


@pytest.mark.parametrize(
    "overrides, message",
    [({"rotate": [0.0, 30.0]}, "Unknown augmentation operation"), ({"rotation": [0.3, 0.1]}, "0 <= low <= high")],
)
def test_bad_bounds_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        resolve_bounds(overrides)


def test_strength_maps_onto_the_widened_range():
    step = AugmentationStep(name="rotation", order=0, probability=1.0, strength_s=10.0, strength_a=0.5)
    wide = resolve_bounds(WIDE)
    assert resolve_step_at_strength(step, 1.0).magnitude == pytest.approx(15.0)  # default bound
    assert resolve_step_at_strength(step, 1.0, wide).magnitude == pytest.approx(45.0)
    assert resolve_step_at_strength(step, 0.5, wide).magnitude == pytest.approx(22.5)
    # An easy sample (rank 0) gets the full bound; a hard one (rank 1) still gets nothing.
    assert resolve_step(step, 0.0, wide).magnitude == pytest.approx(45.0)
    assert resolve_step(step, 1.0, wide).magnitude == pytest.approx(0.0, abs=1e-9)


def test_both_transform_builders_honour_the_bounds():
    policy = _policy()
    wide = resolve_bounds(WIDE)
    adaptive = repr(build_sample_adaptive_transform(policy, IMAGE_SIZE, 0.0, wide))
    assert "RandomRotation(degrees=[-45.0, 45.0]" in adaptive
    constant = repr(build_constant_strength_transform(policy, IMAGE_SIZE, {s.name: 1.0 for s in policy.steps}, wide))
    assert "RandomRotation(degrees=[-45.0, 45.0]" in constant


def _tiny_configs(data_root: Path):
    dataset = DatasetConfig(data_root=str(data_root), image_size=IMAGE_SIZE, batch_size=4, num_workers=0)
    search_space = SearchSpaceConfig(
        initial_channels_min=4, initial_channels_max=4, number_of_cells_min=3, number_of_cells_max=3
    )
    search = Config(
        dataset=dataset,
        search_space=search_space,
        proxies=ProxyConfig(zico_batch_size=2, zico_num_batches=2),
        nsga2=NSGA2Config(population_size=4, num_generations=1, seed=1, device="cpu"),
    )
    training = Config(
        dataset=dataset,
        search_space=search_space,
        training=TrainingConfig(physical_batch_size=4, final_epochs=1, precision="fp32", device="cpu"),
    )
    return search, training


def _policy_file(tmp_path: Path) -> Path:
    path = tmp_path / "selected_policy.json"
    path.write_text(json.dumps({"policy": _policy().to_dict()}))
    return path


def test_final_training_records_the_bounds_it_used(synthetic_dataset_root: Path, tmp_path: Path):
    search_config, training_config = _tiny_configs(synthetic_dataset_root)
    search_dir = tmp_path / "search_run"
    run_search(search_config, search_dir)
    bounds_file = tmp_path / "bounds.json"
    bounds_file.write_text(json.dumps(WIDE))

    for name, path in [("wide", bounds_file), ("default", None)]:
        output_dir = tmp_path / f"training_{name}"
        run_final_training(
            training_config,
            selected_architecture_path=search_dir / "selected_architecture.json",
            split_indices_path=search_dir / "split_indices.json",
            selected_policy_path=_policy_file(tmp_path),
            magnitude_bounds_path=path,
            output_dir=output_dir,
        )
        recorded = json.loads((output_dir / "magnitude_bounds.json").read_text())
        expected = 45.0 if name == "wide" else MAGNITUDE_RANGES["rotation"][1]
        assert recorded["bounds"]["rotation"][1] == pytest.approx(expected)
        assert recorded["source"] == (str(path) if path else "defaults")
        assert set(recorded["bounds"]) == set(MAGNITUDE_RANGES)
