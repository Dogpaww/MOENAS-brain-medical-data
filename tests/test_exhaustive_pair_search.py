"""Exhaustive fixed-pair search: every unordered pair, repeated, reseeded per
trial, resumable, selected by mean."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import brainmri_nas.augment.exhaustive_pair_search as exhaustive
from brainmri_nas.augment.search_space import AUGMENTATION_OPS
from brainmri_nas.search.nsga2_runner import run_search
from brainmri_nas.utils.config import (
    AugmentationConfig,
    Config,
    DatasetConfig,
    NSGA2Config,
    ProxyConfig,
    SearchSpaceConfig,
)

PAIRS = [("rotation", "brightness"), ("rotation", "horizontal_flip"), ("brightness", "random_erasing")]


def _tiny_config(data_root: Path, *, trial_epochs: int = 1) -> Config:
    return Config(
        dataset=DatasetConfig(data_root=str(data_root), image_size=16, batch_size=4, num_workers=0),
        search_space=SearchSpaceConfig(
            initial_channels_min=4, initial_channels_max=4, number_of_cells_min=3, number_of_cells_max=3
        ),
        proxies=ProxyConfig(zico_batch_size=2, zico_num_batches=2),
        nsga2=NSGA2Config(population_size=4, num_generations=1, seed=1, device="cpu"),
        augmentation=AugmentationConfig(trial_epochs=trial_epochs, seed=7, device="cpu"),
    )


@pytest.fixture()
def search_dir(synthetic_dataset_root: Path, tmp_path: Path) -> Path:
    directory = tmp_path / "search_run"
    run_search(_tiny_config(synthetic_dataset_root), directory)
    return directory


@pytest.fixture()
def fake_trials(monkeypatch):
    """Replaces trial training with a fixed score per pair, recording each call."""
    scores = {PAIRS[0]: 0.80, PAIRS[1]: 0.90, PAIRS[2]: 0.70}
    calls = []

    def fake_train_trial_model(model, train_loader, val_loader, **kwargs):
        transform = repr(train_loader.dataset.dataset.transform)
        pair = next(p for p in PAIRS if _pair_in_transform(p, transform))
        calls.append(pair)
        return {"val_macro_auc": scores[pair]}

    monkeypatch.setattr(exhaustive, "train_trial_model", fake_train_trial_model)
    return calls


def _pair_in_transform(pair, transform_repr: str) -> bool:
    markers = {"rotation": "RandomRotation", "brightness": "brightness=(", "horizontal_flip": "RandomHorizontalFlip", "random_erasing": "RandomErasing"}
    return all(markers[op] in transform_repr for op in pair) and sum(m in transform_repr for m in markers.values()) == 2


def _run(config, search_dir, output_dir, repeats, pairs=PAIRS):
    return exhaustive.run_exhaustive_pair_search(
        config,
        selected_architecture_path=search_dir / "selected_architecture.json",
        split_indices_path=search_dir / "split_indices.json",
        output_dir=output_dir,
        repeats=repeats,
        pairs=pairs,
    )


def test_all_pairs_are_the_21_unordered_pairs_in_menu_order():
    pairs = exhaustive.all_pairs()
    assert len(pairs) == 21 == len({frozenset(p) for p in pairs})
    assert all(AUGMENTATION_OPS.index(a) < AUGMENTATION_OPS.index(b) for a, b in pairs)


def test_summary_ranks_by_mean_and_reports_spread():
    archive = [
        {"ops": ["rotation", "brightness"], "val_macro_auc": 0.80},
        {"ops": ["rotation", "brightness"], "val_macro_auc": 0.90},
        {"ops": ["rotation", "contrast"], "val_macro_auc": 0.86},
    ]
    best, second = exhaustive.summarize(archive)
    assert best["ops"] == ["rotation", "contrast"]  # 0.86 beats the 0.85 mean even though 0.90 is the single best trial
    assert second["val_macro_auc_mean"] == pytest.approx(0.85)
    assert second["val_macro_auc_std"] == pytest.approx(0.0707107, abs=1e-6)
    assert second["num_evaluations"] == 2


def test_every_pair_runs_each_repeat_with_that_repeats_seed(synthetic_dataset_root, search_dir, tmp_path, fake_trials, monkeypatch):
    seeds = []
    real_seed_everything = exhaustive.seed_everything
    monkeypatch.setattr(exhaustive, "seed_everything", lambda s: (seeds.append(s), real_seed_everything(s)))

    result = _run(_tiny_config(synthetic_dataset_root), search_dir, tmp_path / "out", repeats=2)

    assert len(result["archive"]) == 6
    assert fake_trials == PAIRS + PAIRS  # repeat 1 of every pair, then repeat 2
    # One seed call for the shared initial weights, then one per trial: 7,7,7 then 8,8,8.
    assert seeds == [7, 7, 7, 7, 8, 8, 8]
    assert [r["seed"] for r in result["archive"]] == [7, 7, 7, 8, 8, 8]


def test_selection_is_written_in_the_legacy_format(synthetic_dataset_root, search_dir, tmp_path, fake_trials):
    output_dir = tmp_path / "out"
    _run(_tiny_config(synthetic_dataset_root), search_dir, output_dir, repeats=2)

    selected = json.loads((output_dir / "selected_legacy_policy.json").read_text())
    assert selected["ops"] == ["rotation", "horizontal_flip"]
    assert selected["val_macro_auc"] == pytest.approx(0.90)
    assert selected["num_evaluations"] == 2
    summary = json.loads((output_dir / "exhaustive_pair_summary.json").read_text())
    assert [s["ops"] for s in summary] == [list(PAIRS[1]), list(PAIRS[0]), list(PAIRS[2])]


def test_rerun_resumes_and_only_runs_missing_trials(synthetic_dataset_root, search_dir, tmp_path, fake_trials):
    output_dir = tmp_path / "out"
    _run(_tiny_config(synthetic_dataset_root), search_dir, output_dir, repeats=1)
    assert len(fake_trials) == 3

    result = _run(_tiny_config(synthetic_dataset_root), search_dir, output_dir, repeats=2)
    assert len(fake_trials) == 6  # only repeat 2 ran the second time
    assert len(result["archive"]) == 6


def test_fewer_repeats_on_rerun_select_from_those_repeats_only(synthetic_dataset_root, search_dir, tmp_path, fake_trials):
    output_dir = tmp_path / "out"
    _run(_tiny_config(synthetic_dataset_root), search_dir, output_dir, repeats=2)
    result = _run(_tiny_config(synthetic_dataset_root), search_dir, output_dir, repeats=1)
    assert all(s["num_evaluations"] == 1 for s in result["summary"])


def test_rerun_with_different_trial_settings_is_refused(synthetic_dataset_root, search_dir, tmp_path, fake_trials):
    output_dir = tmp_path / "out"
    _run(_tiny_config(synthetic_dataset_root), search_dir, output_dir, repeats=1)
    with pytest.raises(ValueError, match="different settings"):
        _run(_tiny_config(synthetic_dataset_root, trial_epochs=2), search_dir, output_dir, repeats=1)


def test_interrupted_search_keeps_finished_trials_and_writes_no_selection(
    synthetic_dataset_root, search_dir, tmp_path, monkeypatch
):
    calls = {"n": 0}

    def flaky_train_trial_model(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("VM went away")
        return {"val_macro_auc": 0.5}

    monkeypatch.setattr(exhaustive, "train_trial_model", flaky_train_trial_model)
    output_dir = tmp_path / "out"
    with pytest.raises(RuntimeError):
        _run(_tiny_config(synthetic_dataset_root), search_dir, output_dir, repeats=1)

    assert len(json.loads((output_dir / "exhaustive_pair_archive.json").read_text())) == 2
    assert not (output_dir / "selected_legacy_policy.json").exists()


def test_real_trials_run_end_to_end(synthetic_dataset_root, search_dir, tmp_path):
    result = _run(_tiny_config(synthetic_dataset_root), search_dir, tmp_path / "out", repeats=1, pairs=PAIRS[:2])
    assert len(result["archive"]) == 2
    assert all(0.0 <= r["val_macro_auc"] <= 1.0 for r in result["archive"])
    assert result["selected_policy"]["ops"] in [list(p) for p in PAIRS[:2]]


def test_trials_start_from_the_same_initial_weights_as_the_legacy_search(
    synthetic_dataset_root, search_dir, tmp_path, monkeypatch
):
    import brainmri_nas.augment.legacy_policy_search as legacy
    from brainmri_nas.utils.config import AugmentationConfig

    captured = {}

    def capture(name):
        def fake_train_trial_model(model, *args, **kwargs):
            captured.setdefault(name, {k: v.clone() for k, v in model.state_dict().items()})
            return {"val_macro_auc": 0.5}
        return fake_train_trial_model

    config = _tiny_config(synthetic_dataset_root)
    config.augmentation = AugmentationConfig(population_size=2, num_generations=1, trial_epochs=1, seed=7, device="cpu")
    monkeypatch.setattr(legacy, "train_trial_model", capture("legacy"))
    legacy.run_legacy_augmentation_search(
        config,
        selected_architecture_path=search_dir / "selected_architecture.json",
        split_indices_path=search_dir / "split_indices.json",
        output_dir=tmp_path / "legacy",
    )
    monkeypatch.setattr(exhaustive, "train_trial_model", capture("exhaustive"))
    _run(config, search_dir, tmp_path / "exhaustive", repeats=1, pairs=PAIRS[:1])

    assert captured["legacy"].keys() == captured["exhaustive"].keys()
    assert all((captured["legacy"][k] == captured["exhaustive"][k]).all() for k in captured["legacy"])
