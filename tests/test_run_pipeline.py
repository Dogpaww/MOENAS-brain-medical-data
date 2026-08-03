"""Regression test for scripts/run_pipeline.py: the single-command orchestrator
that chains architecture search -> augmentation search -> final training.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from brainmri_nas.utils.config import (
    AugmentationConfig,
    Config,
    DatasetConfig,
    NSGA2Config,
    ProxyConfig,
    SearchSpaceConfig,
    TrainingConfig,
    save_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_pipeline.py"


def _write_tiny_config(path: Path, data_root: Path) -> None:
    config = Config(
        dataset=DatasetConfig(data_root=str(data_root), image_size=16, batch_size=4, num_workers=0),
        search_space=SearchSpaceConfig(
            initial_channels_min=4, initial_channels_max=4, number_of_cells_min=3, number_of_cells_max=3
        ),
        proxies=ProxyConfig(zico_batch_size=2, zico_num_batches=2),
        nsga2=NSGA2Config(population_size=4, num_generations=1, seed=1, device="cpu"),
        augmentation=AugmentationConfig(population_size=2, num_generations=1, trial_epochs=1, seed=1, device="cpu"),
        training=TrainingConfig(physical_batch_size=4, final_epochs=1, precision="amp", device="cpu"),
    )
    save_config(config, path)


def _run(config_path: Path, output_dir: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-u", str(PIPELINE_SCRIPT), "--config", str(config_path), "--output-dir", str(output_dir), *extra_args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_run_pipeline_runs_all_three_stages_and_is_resumable(synthetic_dataset_root: Path, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    _write_tiny_config(config_path, synthetic_dataset_root)
    output_dir = tmp_path / "outputs"

    first = _run(config_path, output_dir)
    assert first.returncode == 0, f"first run failed:\nSTDOUT:\n{first.stdout}\nSTDERR:\n{first.stderr}"
    assert "STAGE 1/3" in first.stdout
    assert "STAGE 2/3" in first.stdout
    assert "STAGE 3/3" in first.stdout
    assert "Test accuracy:" in first.stdout

    for path in (
        output_dir / "search_run" / "selected_architecture.json",
        output_dir / "augmentation_run" / "selected_policy.json",
        output_dir / "training_run" / "test_metrics.json",
        output_dir / "training_run" / "best_checkpoint.pt",
    ):
        assert path.exists(), f"missing expected output: {path}"

    # Re-running must skip every stage, not redo the work.
    second = _run(config_path, output_dir)
    assert second.returncode == 0
    assert "[skip] architecture search already done" in second.stdout
    assert "[skip] augmentation search already done" in second.stdout
    assert "[skip] final training already done" in second.stdout
    assert "STAGE 1/3" not in second.stdout


def test_run_pipeline_skip_augmentation(synthetic_dataset_root: Path, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    _write_tiny_config(config_path, synthetic_dataset_root)
    output_dir = tmp_path / "outputs"

    result = _run(config_path, output_dir, "--skip-augmentation")
    assert result.returncode == 0, f"run failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "augmentation search disabled" in result.stdout
    assert not (output_dir / "augmentation_run").exists()
    assert (output_dir / "training_run" / "test_metrics.json").exists()
