from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from brainmri_nas.augment.policy_search import run_augmentation_search
from brainmri_nas.model.network import build_model
from brainmri_nas.search.nsga2_runner import run_search
from brainmri_nas.search_space.chromosome import decode_chromosome
from brainmri_nas.training.engine import train_one_epoch
from brainmri_nas.training.evaluate import evaluate_model
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
from brainmri_nas.utils.optim import build_optimizer_and_scheduler

INPUT_CHANNELS, IMAGE_SIZE, NUM_CLASSES = 3, 16, 4


def _tiny_model():
    from brainmri_nas.search_space.chromosome import chromosome_length

    length = chromosome_length()
    chromosome = [((i * 41) % 89) / 89.0 for i in range(length)]
    genotype = decode_chromosome(chromosome)
    return build_model(
        genotype,
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        initial_channels=4,
        number_of_cells=3,
        drop_path_probability=0.0,
        stem_type="cifar",
    )


def _tiny_loader(num_samples=8, batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_samples, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, generator=g)
    y = torch.randint(0, NUM_CLASSES, (num_samples,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_default_gradient_accumulation_is_disabled():
    assert TrainingConfig().gradient_accumulation_steps == 1


def test_optimizer_and_scheduler_match_darts_recipe():
    # SGD(momentum) + CosineAnnealingLR over the full epoch budget -- the
    # standard DARTS training recipe these lr/weight_decay/momentum values
    # belong to (see utils/optim.py docstring for why this replaced Adam).
    model = _tiny_model()
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, learning_rate=0.025, weight_decay=3e-4, momentum=0.9, epochs=200
    )

    assert isinstance(optimizer, torch.optim.SGD)
    param_group = optimizer.param_groups[0]
    assert param_group["lr"] == 0.025
    assert param_group["weight_decay"] == 3e-4
    assert param_group["momentum"] == 0.9

    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 200


def test_cancer_no_tumor_penalty_increases_reported_loss_when_active():
    # lr=0.0 freezes the weights across both calls -- an apples-to-apples
    # comparison of the loss value itself, not training dynamics.
    torch.manual_seed(0)
    model = _tiny_model()
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    loader = _tiny_loader()

    loss_without_penalty = train_one_epoch(
        model,
        loader,
        torch.optim.SGD(model.parameters(), lr=0.0),
        device=torch.device("cpu"),
        use_amp=False,
        scaler=None,
        accumulation_steps=1,
        grad_clip_norm=5.0,
        no_tumor_class_index=2,
        cancer_no_tumor_penalty=0.0,
    )

    model.load_state_dict(initial_state)
    loss_with_penalty = train_one_epoch(
        model,
        loader,
        torch.optim.SGD(model.parameters(), lr=0.0),
        device=torch.device("cpu"),
        use_amp=False,
        scaler=None,
        accumulation_steps=1,
        grad_clip_norm=5.0,
        no_tumor_class_index=2,
        cancer_no_tumor_penalty=0.5,
    )

    assert loss_with_penalty > loss_without_penalty


def test_scheduler_degrades_gracefully_for_short_runs():
    model = _tiny_model()
    # CosineAnnealingLR needs T_max >= 1 -- a 1-epoch smoke-test run must not
    # raise or produce a degenerate (e.g. zero-length) cosine cycle.
    optimizer, scheduler = build_optimizer_and_scheduler(model, learning_rate=0.025, weight_decay=3e-4, epochs=1)
    assert scheduler.T_max == 1

    x = torch.randn(2, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    optimizer.zero_grad()
    model(x).sum().backward()
    optimizer.step()
    scheduler.step()  # must not raise

    assert optimizer.param_groups[0]["lr"] >= 0.0


def test_evaluate_model_returns_full_metric_suite():
    model = _tiny_model()
    loader = _tiny_loader()
    metrics = evaluate_model(model, loader, device=torch.device("cpu"), num_classes=NUM_CLASSES)

    for key in ("loss", "accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc", "num_samples"):
        assert key in metrics
    assert len(metrics["per_class"]["precision"]) == NUM_CLASSES
    assert len(metrics["per_class"]["recall"]) == NUM_CLASSES
    assert len(metrics["per_class"]["f1"]) == NUM_CLASSES
    assert len(metrics["confusion_matrix"]) == NUM_CLASSES
    assert all(len(row) == NUM_CLASSES for row in metrics["confusion_matrix"])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_evaluate_model_moves_predictions_to_cpu_before_numpy():
    # MPS tensors raise on .numpy() unless moved to CPU first -- if evaluate_model
    # ever forgot to do that, this would crash instead of merely being "slow".
    model = _tiny_model().to("mps")
    loader = _tiny_loader()
    metrics = evaluate_model(model, loader, device=torch.device("mps"), num_classes=NUM_CLASSES)
    assert isinstance(metrics["accuracy"], float)


def test_train_one_epoch_updates_weights_and_returns_finite_loss():
    model = _tiny_model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    loss = train_one_epoch(
        model,
        _tiny_loader(),
        optimizer,
        device=torch.device("cpu"),
        use_amp=False,
        scaler=None,
        accumulation_steps=1,
        grad_clip_norm=5.0,
    )

    assert loss == loss  # not NaN
    changed = any(not torch.equal(p.detach(), before[name]) for name, p in model.named_parameters())
    assert changed


def test_train_one_epoch_emits_batch_progress_logs(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="brainmri_nas.final_training")
    model = _tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # 8 samples, batch_size=1 -> 8 batches/epoch -> interval = max(1, 8//5) = 1 -> logs every batch.
    train_one_epoch(
        model,
        _tiny_loader(num_samples=8, batch_size=1),
        optimizer,
        device=torch.device("cpu"),
        use_amp=False,
        scaler=None,
        accumulation_steps=1,
        grad_clip_norm=5.0,
        epoch=3,
        total_epochs=10,
    )

    messages = [r.getMessage() for r in caplog.records]
    assert any("3/10" in m and "batch" in m for m in messages)
    assert any("8/8" in m for m in messages)  # the final batch always logs


def test_train_one_epoch_amp_without_scaler_raises():
    model = _tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError):
        train_one_epoch(
            model,
            _tiny_loader(),
            optimizer,
            device=torch.device("cpu"),
            use_amp=True,
            scaler=None,
            accumulation_steps=1,
            grad_clip_norm=5.0,
        )


def test_gradient_accumulation_flushes_final_partial_group(monkeypatch):
    model = _tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    step_calls = {"count": 0}
    original_step = optimizer.step

    def counting_step(*args, **kwargs):
        step_calls["count"] += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", counting_step)

    # 5 samples, batch_size=1 -> 5 micro-batches, accumulation_steps=2 -> flush at 2, 4, 5.
    train_one_epoch(
        model,
        _tiny_loader(num_samples=5, batch_size=1),
        optimizer,
        device=torch.device("cpu"),
        use_amp=False,
        scaler=None,
        accumulation_steps=2,
        grad_clip_norm=5.0,
    )

    assert step_calls["count"] == 3


def _linear_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, NUM_CLASSES))


def test_gradient_accumulation_matches_full_batch_gradient_for_a_bn_free_model():
    # No BatchNorm in this tiny model, so accumulated micro-batch gradients
    # should sum to *exactly* the full-batch gradient (handoff §26's BN caveat
    # doesn't apply here -- this isolates the accumulation math itself).
    x = torch.randn(8, 3, 8, 8, generator=torch.Generator().manual_seed(1))
    y = torch.randint(0, NUM_CLASSES, (8,), generator=torch.Generator().manual_seed(2))

    model_full = _linear_model()
    initial_state = copy.deepcopy(model_full.state_dict())
    optimizer_full = torch.optim.SGD(model_full.parameters(), lr=0.1, momentum=0.0)
    train_one_epoch(
        model_full,
        DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False),
        optimizer_full,
        device=torch.device("cpu"),
        use_amp=False,
        scaler=None,
        accumulation_steps=1,
        grad_clip_norm=1e9,  # effectively disable clipping for this comparison
    )

    model_accum = _linear_model()
    model_accum.load_state_dict(initial_state)
    optimizer_accum = torch.optim.SGD(model_accum.parameters(), lr=0.1, momentum=0.0)
    train_one_epoch(
        model_accum,
        DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False),
        optimizer_accum,
        device=torch.device("cpu"),
        use_amp=False,
        scaler=None,
        accumulation_steps=2,
        grad_clip_norm=1e9,
    )

    for name in initial_state:
        assert torch.allclose(
            model_full.state_dict()[name], model_accum.state_dict()[name], atol=1e-6
        ), f"{name} diverged between full-batch and accumulated updates"


def test_end_to_end_tiny_final_training(synthetic_dataset_root: Path, tmp_path: Path):
    search_config = Config(
        dataset=DatasetConfig(data_root=str(synthetic_dataset_root), image_size=16, batch_size=4, num_workers=0),
        search_space=SearchSpaceConfig(
            initial_channels_min=4, initial_channels_max=4, number_of_cells_min=3, number_of_cells_max=3
        ),
        proxies=ProxyConfig(zico_batch_size=2, zico_num_batches=2),
        nsga2=NSGA2Config(population_size=4, num_generations=1, seed=1, device="cpu"),
    )
    search_output_dir = tmp_path / "search_run"
    run_search(search_config, search_output_dir)

    augmentation_config = Config(
        dataset=search_config.dataset,
        search_space=search_config.search_space,
        augmentation=AugmentationConfig(population_size=2, num_generations=1, trial_epochs=1, seed=1, device="cpu"),
    )
    augmentation_output_dir = tmp_path / "augmentation_run"
    run_augmentation_search(
        augmentation_config,
        selected_architecture_path=search_output_dir / "selected_architecture.json",
        split_indices_path=search_output_dir / "split_indices.json",
        output_dir=augmentation_output_dir,
    )

    training_config = Config(
        dataset=search_config.dataset,
        search_space=search_config.search_space,
        training=TrainingConfig(
            physical_batch_size=4,
            gradient_accumulation_steps=2,
            final_epochs=2,
            precision="amp",  # requested, but device=cpu -> should silently fall back to fp32, not crash
            device="cpu",
        ),
    )
    training_output_dir = tmp_path / "training_run"
    result = run_final_training(
        training_config,
        selected_architecture_path=search_output_dir / "selected_architecture.json",
        split_indices_path=search_output_dir / "split_indices.json",
        selected_policy_path=augmentation_output_dir / "selected_policy.json",
        output_dir=training_output_dir,
    )

    for filename in (
        "training_config.yaml",
        "selected_architecture.json",
        "selected_policy.json",
        "training_history.csv",
        "best_checkpoint.pt",
        "validation_metrics.json",
        "test_metrics.json",
        "per_class_metrics.csv",
        "confusion_matrix.png",
        "memory_profile.json",
        "training.log",
        "run_manifest.json",
    ):
        assert (training_output_dir / filename).exists(), f"missing output file: {filename}"

    assert result["best_epoch"] >= 1
    assert 0.0 <= result["test_metrics"]["accuracy"] <= 1.0
    assert len(result["history"]) == training_config.training.final_epochs

    import pandas as pd

    history_df = pd.read_csv(training_output_dir / "training_history.csv")
    assert len(history_df) == training_config.training.final_epochs
    assert history_df["is_best"].any()
