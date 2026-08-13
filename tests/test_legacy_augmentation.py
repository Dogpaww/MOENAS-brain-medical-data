from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from brainmri_nas.augment.legacy_policy_search import run_legacy_augmentation_search
from brainmri_nas.augment.legacy_search_space import (
    chromosome_length,
    decode_legacy_chromosome,
    legacy_steps,
    resolve_fixed_step,
    validate_chromosome,
)
from brainmri_nas.augment.legacy_transform_builder import build_legacy_transform
from brainmri_nas.augment.search_space import AUGMENTATION_OPS, MAGNITUDE_RANGES
from brainmri_nas.model.network import build_model
from brainmri_nas.search.nsga2_runner import run_search
from brainmri_nas.search_space.chromosome import decode_chromosome as decode_arch_chromosome
from brainmri_nas.training.final_training import run_final_training
from brainmri_nas.utils.config import (
    AugmentationConfig,
    Config,
    DatasetConfig,
    NSGA2Config,
    ProxyConfig,
    SearchSpaceConfig,
    TrainingConfig,
)

N_VAR = chromosome_length()


def _sample_chromosome(offset=0.0):
    return [((i * 0.37 + offset) % 1.0) * 0.999 for i in range(N_VAR)]


def test_chromosome_length_is_two():
    assert N_VAR == 2


def test_decode_never_repeats_across_the_full_gene_grid():
    """Exhaustive over a coarse grid: every (a, b) must decode to 2 DISTINCT
    ops from the real menu, deterministically. A repeat would mean the same
    op got applied twice, silently doing nothing the second time."""
    seen = set()
    steps = 25
    for i in range(steps):
        for j in range(steps):
            a, b = i / steps, j / steps
            op1, op2 = decode_legacy_chromosome([a, b])
            assert op1 != op2
            assert op1 in AUGMENTATION_OPS
            assert op2 in AUGMENTATION_OPS
            assert decode_legacy_chromosome([a, b]) == (op1, op2)  # deterministic
            seen.add((op1, op2))
    max_possible = len(AUGMENTATION_OPS) * (len(AUGMENTATION_OPS) - 1)
    assert len(seen) == max_possible, f"only {len(seen)}/{max_possible} ordered pairs reachable"


def test_decode_handles_the_open_upper_boundary():
    # Genes are valid on [0, 1) -- the highest legal value must not index out of range.
    op1, op2 = decode_legacy_chromosome([1.0 - 1e-9, 1.0 - 1e-9])
    assert op1 != op2
    assert {op1, op2} <= set(AUGMENTATION_OPS)


def test_validate_chromosome_rejects_wrong_length_and_out_of_range():
    with pytest.raises(ValueError, match="length"):
        validate_chromosome([0.1])
    with pytest.raises(ValueError, match="range"):
        validate_chromosome([1.0, 0.5])


def test_resolved_step_is_fixed_at_max_safe_magnitude_and_always_applied():
    for name in AUGMENTATION_OPS:
        resolved = resolve_fixed_step(name, order=0)
        assert resolved.probability == 1.0
        assert resolved.magnitude == MAGNITUDE_RANGES[name][1]
        # Never depends on anything sample-specific -- calling it again must
        # give an identical result, unlike the sample-adaptive resolve_step.
        again = resolve_fixed_step(name, order=0)
        assert resolved == again


def test_legacy_steps_round_trip_through_to_dict():
    op1, op2 = "rotation", "random_erasing"
    s1, s2 = legacy_steps(op1, op2)
    assert s1.name == op1 and s1.order == 0
    assert s2.name == op2 and s2.order == 1
    assert s1.to_dict()["probability"] == 1.0


def test_legacy_transform_runs_end_to_end_and_routes_ops_correctly():
    from PIL import Image

    # random_erasing must land in tensor space (after ToTensor); rotation in
    # PIL space (before it) -- get this wrong and the run crashes on a
    # tensor being handed a PIL-only op or vice versa. Running the transform
    # end to end is a real check of that routing, not just a unit test of
    # which set contains which string.
    transform = build_legacy_transform("rotation", "random_erasing", image_size=32)
    img = Image.new("RGB", (40, 40), color=(100, 50, 150))
    out = transform(img)
    assert out.shape == torch.Size([3, 32, 32])
    assert out.dtype == torch.float32


def test_legacy_transform_is_not_the_identity():
    """A transform that silently no-ops would pass every shape-only test
    while doing nothing -- assert the pixels actually differ from a plain
    resize+normalize of the same image."""
    from PIL import Image

    from brainmri_nas.data.transforms import build_train_transform

    img = Image.new("RGB", (48, 48), color=(30, 200, 90))
    plain = build_train_transform(32)(img)
    torch.manual_seed(0)
    augmented = build_legacy_transform("brightness", "contrast", image_size=32)(img)
    assert not torch.allclose(plain, augmented)


def _tiny_config(synthetic_dataset_root: Path) -> Config:
    return Config(
        dataset=DatasetConfig(data_root=str(synthetic_dataset_root), image_size=16, batch_size=4, num_workers=0),
        search_space=SearchSpaceConfig(
            initial_channels_min=4, initial_channels_max=4, number_of_cells_min=3, number_of_cells_max=3
        ),
        proxies=ProxyConfig(zico_batch_size=2, zico_num_batches=2),
        nsga2=NSGA2Config(population_size=4, num_generations=1, seed=1, device="cpu"),
        augmentation=AugmentationConfig(population_size=2, num_generations=1, trial_epochs=1, seed=1, device="cpu"),
    )


def test_end_to_end_tiny_legacy_augmentation_search(synthetic_dataset_root: Path, tmp_path: Path):
    config = _tiny_config(synthetic_dataset_root)
    search_dir = tmp_path / "search_run"
    run_search(config, search_dir)

    augmentation_dir = tmp_path / "augmentation_run"
    result = run_legacy_augmentation_search(
        config,
        selected_architecture_path=search_dir / "selected_architecture.json",
        split_indices_path=search_dir / "split_indices.json",
        output_dir=augmentation_dir,
    )

    for filename in (
        "legacy_policy_archive.json",
        "selected_legacy_policy.json",
        "legacy_augmentation_search.log",
        "augmentation_config.yaml",
        "run_manifest.json",
    ):
        assert (augmentation_dir / filename).exists(), f"missing output file: {filename}"

    assert len(result["archive"]) >= config.augmentation.population_size
    selected = result["selected_policy"]
    assert len(selected["ops"]) == 2
    assert selected["ops"][0] != selected["ops"][1]
    assert 0.0 <= selected["val_macro_auc"] <= 1.0


def test_final_training_consumes_a_legacy_policy_unmodified(synthetic_dataset_root: Path, tmp_path: Path):
    config = _tiny_config(synthetic_dataset_root)
    search_dir = tmp_path / "search_run"
    run_search(config, search_dir)

    augmentation_dir = tmp_path / "augmentation_run"
    run_legacy_augmentation_search(
        config,
        selected_architecture_path=search_dir / "selected_architecture.json",
        split_indices_path=search_dir / "split_indices.json",
        output_dir=augmentation_dir,
    )

    training_config = Config(
        dataset=config.dataset,
        search_space=config.search_space,
        training=TrainingConfig(
            physical_batch_size=4, gradient_accumulation_steps=1, final_epochs=1, precision="fp32", device="cpu"
        ),
    )
    result = run_final_training(
        training_config,
        selected_architecture_path=search_dir / "selected_architecture.json",
        split_indices_path=search_dir / "split_indices.json",
        selected_legacy_policy_path=augmentation_dir / "selected_legacy_policy.json",
        output_dir=tmp_path / "training_run",
    )
    assert result["best_epoch"] == 1
    assert 0.0 <= result["test_metrics"]["accuracy"] <= 1.0

    checkpoint = (tmp_path / "training_run" / "selected_legacy_policy.json").exists()
    assert checkpoint


def test_final_training_rejects_both_policy_kinds_at_once(synthetic_dataset_root: Path, tmp_path: Path):
    config = _tiny_config(synthetic_dataset_root)
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_final_training(
            config,
            selected_architecture_path=tmp_path / "does_not_matter.json",
            split_indices_path=tmp_path / "does_not_matter.json",
            selected_policy_path=tmp_path / "a.json",
            selected_legacy_policy_path=tmp_path / "b.json",
            output_dir=tmp_path / "training_run",
        )


def test_final_training_checkpoint_records_the_legacy_ops(synthetic_dataset_root: Path, tmp_path: Path):
    """The checkpoint's augmentation_policy field must reflect what was
    ACTUALLY applied during training, not silently stay None or carry over
    the sample-adaptive schema's fields as if they meant something here."""
    config = _tiny_config(synthetic_dataset_root)
    search_dir = tmp_path / "search_run"
    run_search(config, search_dir)
    augmentation_dir = tmp_path / "augmentation_run"
    run_legacy_augmentation_search(
        config,
        selected_architecture_path=search_dir / "selected_architecture.json",
        split_indices_path=search_dir / "split_indices.json",
        output_dir=augmentation_dir,
    )
    selected_ops = tuple(__import__("json").loads((augmentation_dir / "selected_legacy_policy.json").read_text())["ops"])

    training_config = Config(
        dataset=config.dataset,
        search_space=config.search_space,
        training=TrainingConfig(
            physical_batch_size=4, gradient_accumulation_steps=1, final_epochs=1, precision="fp32", device="cpu"
        ),
    )
    training_dir = tmp_path / "training_run"
    run_final_training(
        training_config,
        selected_architecture_path=search_dir / "selected_architecture.json",
        split_indices_path=search_dir / "split_indices.json",
        selected_legacy_policy_path=augmentation_dir / "selected_legacy_policy.json",
        output_dir=training_dir,
    )

    from brainmri_nas.training.checkpoint import load_checkpoint

    payload = load_checkpoint(training_dir / "best_checkpoint.pt")
    recorded_ops = {step["name"] for step in payload["augmentation_policy"]}
    assert recorded_ops == set(selected_ops)
    assert all(step["probability"] == 1.0 for step in payload["augmentation_policy"])
