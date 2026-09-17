"""scripts/run_final_training.py: what the command line passes to
run_final_training -- seed override and which augmentation policy, if any."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from brainmri_nas.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_final_training.py"
# Parsed only; no data is read, and run_final_training is stubbed out.
CONFIG = REPO / "configs" / "figshare.yaml"


@pytest.fixture()
def cli(monkeypatch):
    spec = importlib.util.spec_from_file_location("run_final_training_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = []

    def fake_run_final_training(config, **kwargs):
        calls.append({"config": config, **kwargs})
        return {"best_epoch": 1, "test_metrics": {"accuracy": 0.5, "macro_f1": 0.5, "macro_auc": 0.5}}

    monkeypatch.setattr(module, "run_final_training", fake_run_final_training)
    return module, calls


@pytest.fixture()
def dirs(tmp_path: Path) -> dict[str, Path]:
    adaptive = tmp_path / "augmentation_run"
    adaptive.mkdir()
    (adaptive / "selected_policy.json").write_text(json.dumps({}))
    fixed = tmp_path / "legacy_augmentation_run"
    fixed.mkdir()
    (fixed / "selected_legacy_policy.json").write_text(json.dumps({"ops": ["brightness", "rotation"]}))
    return {"config": CONFIG, "adaptive": adaptive, "fixed": fixed, "search": tmp_path / "search_run", "out": tmp_path / "out"}


def _base_args(dirs):
    return ["--config", str(dirs["config"]), "--search-output-dir", str(dirs["search"]), "--output-dir", str(dirs["out"])]


def test_no_flags_trains_without_augmentation_and_keeps_config_seed(cli, dirs):
    module, calls = cli
    module.main(_base_args(dirs))
    (call,) = calls
    assert call["selected_policy_path"] is None
    assert call["selected_legacy_policy_path"] is None
    assert call["config"].training.seed == load_config(CONFIG).training.seed


def test_seed_flag_overrides_config_seed(cli, dirs):
    module, calls = cli
    config_seed = load_config(CONFIG).training.seed
    new_seed = config_seed + 2
    module.main(_base_args(dirs) + ["--seed", str(new_seed)])
    assert calls[0]["config"].training.seed == new_seed


def test_adaptive_flag_passes_selected_policy(cli, dirs):
    module, calls = cli
    module.main(_base_args(dirs) + ["--augmentation-output-dir", str(dirs["adaptive"])])
    assert calls[0]["selected_policy_path"] == dirs["adaptive"] / "selected_policy.json"
    assert calls[0]["selected_legacy_policy_path"] is None


def test_legacy_flag_passes_fixed_policy(cli, dirs):
    module, calls = cli
    module.main(_base_args(dirs) + ["--legacy-augmentation-output-dir", str(dirs["fixed"])])
    assert calls[0]["selected_legacy_policy_path"] == dirs["fixed"] / "selected_legacy_policy.json"
    assert calls[0]["selected_policy_path"] is None


def test_both_augmentation_flags_are_rejected(cli, dirs):
    module, calls = cli
    with pytest.raises(SystemExit):
        module.main(
            _base_args(dirs)
            + ["--augmentation-output-dir", str(dirs["adaptive"]), "--legacy-augmentation-output-dir", str(dirs["fixed"])]
        )
    assert not calls


def test_missing_policy_file_fails_before_training(cli, dirs, tmp_path):
    module, calls = cli
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="does not exist"):
        module.main(_base_args(dirs) + ["--legacy-augmentation-output-dir", str(empty)])
    assert not calls
