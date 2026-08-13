from __future__ import annotations

from pathlib import Path

import torch

from brainmri_nas.model.network import build_model
from brainmri_nas.search.random_baseline import run_random_baseline
from brainmri_nas.search_space.genotype import NetworkGenotype
from brainmri_nas.training.final_training import run_final_training
from brainmri_nas.utils.config import Config, DatasetConfig, NSGA2Config, ProxyConfig, SearchSpaceConfig, TrainingConfig


def _tiny_config(synthetic_dataset_root: Path, seed: int = 1) -> Config:
    return Config(
        dataset=DatasetConfig(data_root=str(synthetic_dataset_root), image_size=16, batch_size=4, num_workers=0),
        search_space=SearchSpaceConfig(
            initial_channels_min=4, initial_channels_max=4, number_of_cells_min=3, number_of_cells_max=3
        ),
        proxies=ProxyConfig(zico_batch_size=2, zico_num_batches=2),
        nsga2=NSGA2Config(seed=seed, device="cpu"),
    )


def test_random_baseline_writes_the_files_stage_2_and_3_require(synthetic_dataset_root: Path, tmp_path: Path):
    config = _tiny_config(synthetic_dataset_root)
    output_dir = tmp_path / "search_run"
    result = run_random_baseline(config, output_dir, seed=7)

    for filename in (
        "config.yaml",
        "dataset_report.json",
        "class_mapping.json",
        "split_indices.json",  # required by run_augmentation_search / run_final_training
        "proxy_sample_indices.json",
        "selected_architecture.json",
        "search.log",
        "run_manifest.json",
    ):
        assert (output_dir / filename).exists(), f"missing output file: {filename}"

    # No NSGA-II/TOPSIS artifacts -- there was nothing to select among.
    for filename in ("search_archive.json", "pareto_front.json", "topsis_ranking.json", "candidate_cache.json"):
        assert not (output_dir / filename).exists()

    selected = result["selected_architecture"]
    assert selected["valid"] is True
    assert "topsis_score" not in selected  # never ranked, so must never claim to be

    genotype = NetworkGenotype.from_dict(selected["genotype"])
    model = build_model(
        genotype,
        input_channels=config.dataset.input_channels,
        num_classes=config.dataset.num_classes,
        image_size=config.dataset.image_size,
        initial_channels=selected["initial_channels"],
        number_of_cells=selected["number_of_cells"],
        drop_path_probability=0.0,
        stem_type=config.search_space.stem_type,
    )
    logits = model(torch.randn(2, config.dataset.input_channels, config.dataset.image_size, config.dataset.image_size))
    assert logits.shape == (2, config.dataset.num_classes)


def test_same_seed_draws_the_same_architecture_different_seed_differs(synthetic_dataset_root: Path, tmp_path: Path):
    config = _tiny_config(synthetic_dataset_root)
    r1 = run_random_baseline(config, tmp_path / "a", seed=42)
    r2 = run_random_baseline(config, tmp_path / "b", seed=42)
    r3 = run_random_baseline(config, tmp_path / "c", seed=43)

    assert r1["selected_architecture"]["candidate_hash"] == r2["selected_architecture"]["candidate_hash"]
    assert r1["selected_architecture"]["candidate_hash"] != r3["selected_architecture"]["candidate_hash"]


def test_final_training_consumes_a_random_baseline_architecture_unmodified(
    synthetic_dataset_root: Path, tmp_path: Path
):
    """The point of the whole design: stage 3 must not need to know or care
    that this architecture came from a random draw rather than NSGA-II."""
    config = _tiny_config(synthetic_dataset_root)
    search_dir = tmp_path / "search_run"
    run_random_baseline(config, search_dir, seed=3)

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
        output_dir=tmp_path / "training_run",
    )
    assert result["best_epoch"] == 1
    assert 0.0 <= result["test_metrics"]["accuracy"] <= 1.0
