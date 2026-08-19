"""External baseline training orchestration (branch: benchmark).

Mirrors `training/final_training.py`'s flow (build data -> train with
validation-only per-epoch evaluation -> track the best checkpoint via a
smoothed metric -> reload it independently -> evaluate the test set exactly
once) but trimmed to what a plain, ImageNet-pretrained torchvision model
needs: no chromosome/genotype, no selected augmentation policy, no
LossCache sample-adaptivity. Every lower-level piece this reuses
(`build_dataset_bundle`, `train_one_epoch`, `evaluate_model`,
`build_optimizer_and_scheduler`, `profile_flops_and_params`) is the exact
same code the NAS pipeline itself calls, and is generic on `nn.Module` --
none of it is NASNetwork-specific -- so this comparison is controlled by
construction: only the model differs, not the data, split, metrics, or
profiling method.

Deliberately fine-tunes pretrained weights rather than training from
scratch, and uses far fewer epochs and a lower learning rate than the NAS
pipeline's from-scratch recipe (`TrainingConfig.final_epochs=200`,
`learning_rate=0.025`): that recipe was tuned for a randomly-initialized
custom network and would either waste most of a fine-tuning run's epochs
past convergence or actively overfit a pretrained model on ~2,100 images.
See HANDOFF.md for the fuller rationale.
"""

from __future__ import annotations

import gc
import logging
import math
from pathlib import Path

import pandas as pd
import torch

from brainmri_nas.baselines.checkpoint import load_checkpoint, rebuild_model_from_checkpoint, save_checkpoint
from brainmri_nas.baselines.registry import build_baseline_model
from brainmri_nas.data.loader import build_dataset_bundle
from brainmri_nas.proxies.profiling import profile_flops_and_params
from brainmri_nas.training.engine import train_one_epoch
from brainmri_nas.training.evaluate import evaluate_model, save_confusion_matrix_plot, save_per_class_metrics_csv
from brainmri_nas.utils.config import Config
from brainmri_nas.utils.determinism import seed_everything
from brainmri_nas.utils.device import resolve_device
from brainmri_nas.utils.git_info import get_run_manifest
from brainmri_nas.utils.optim import build_optimizer_and_scheduler
from brainmri_nas.utils.serialization import dump_json

# Matches final_training.py's CHECKPOINT_SMOOTHING_WINDOW -- same rationale
# (a single epoch's validation score is a noisy "best" signal), kept equal
# on purpose so checkpoint selection is methodologically identical across
# every model being compared, not just the model itself.
CHECKPOINT_SMOOTHING_WINDOW = 3


def _configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("brainmri_nas.baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(output_dir / "training.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    return logger


def run_baseline_training(
    config: Config,
    *,
    model_name: str,
    split_indices_path: str | Path,
    output_dir: str | Path,
    image_size: int = 224,
    epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 7e-4,
    momentum: float = 0.9,
    label_smoothing: float = 0.1,
    grad_clip_norm: float = 5.0,
    checkpoint_metric: str = "macro_auc",
    seed: int = 1,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(output_dir)

    seed_everything(seed)
    device = resolve_device(config.training.device)
    use_amp = device.type == "cuda"

    logger.info(
        "Starting baseline training: model=%s device=%s amp=%s image_size=%d epochs=%d lr=%g",
        model_name,
        device,
        use_amp,
        image_size,
        epochs,
        learning_rate,
    )

    split_indices_path = Path(split_indices_path)
    if not split_indices_path.exists():
        raise FileNotFoundError(
            f"{split_indices_path} does not exist. This function never computes a fresh "
            "split -- point it at an existing split_indices.json (e.g. from the searched-"
            "architecture run) so the comparison is on the exact same patient split."
        )

    bundle = build_dataset_bundle(
        config.dataset.data_root,
        image_size=image_size,
        validation_fraction=config.dataset.validation_fraction,
        split_seed=config.dataset.split_seed,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        split_indices_path=split_indices_path,
        group_aware_split=config.dataset.group_aware_split,
    )
    logger.info(
        "Dataset ready: %d classes, %d train / %d val samples (split reused from %s)",
        bundle.num_classes,
        len(bundle.train_indices),
        len(bundle.val_indices),
        split_indices_path,
    )

    model = build_baseline_model(model_name, num_classes=bundle.num_classes)
    model.to(device)
    logger.info("Classes: %s", dict(bundle.class_to_idx))
    logger.info(
        "Regularization: weight_decay=%.1e label_smoothing=%.2f grad_clip_norm=%.1f",
        weight_decay,
        label_smoothing,
        grad_clip_norm,
    )

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs, momentum=momentum
    )
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    checkpoint_path = output_dir / "best_checkpoint.pt"
    best_score = float("-inf")
    best_epoch = -1
    history: list[dict] = []
    recent_scores: list[float] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            bundle.train_loader,
            optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            accumulation_steps=1,
            grad_clip_norm=grad_clip_norm,
            epoch=epoch,
            total_epochs=epochs,
            loss_cache=None,
            label_smoothing=label_smoothing,
            logger=logger,
        )
        scheduler.step()

        val_metrics = evaluate_model(model, bundle.val_loader, device=device, num_classes=bundle.num_classes)
        score = val_metrics[checkpoint_metric]
        recent_scores.append(score)

        window = [s for s in recent_scores[-CHECKPOINT_SMOOTHING_WINDOW:] if math.isfinite(s)]
        smoothed_score = sum(window) / len(window) if window else float("-inf")

        is_best = smoothed_score > best_score
        if is_best:
            best_score = smoothed_score
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model=model,
                model_name=model_name,
                class_to_idx=bundle.class_to_idx,
                image_size=image_size,
                epoch=epoch,
                validation_metrics=val_metrics,
            )
            dump_json(val_metrics, output_dir / "validation_metrics.json")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_macro_auc": val_metrics["macro_auc"],
                "checkpoint_score_smoothed": smoothed_score,
                "learning_rate": scheduler.get_last_lr()[0],
                "is_best": is_best,
            }
        )
        logger.info(
            "epoch %d/%d done: train_loss=%.4f val_loss=%.4f val_accuracy=%.4f "
            "val_macro_f1=%.4f val_macro_auc=%.4f (checkpoint_metric=%s)%s",
            epoch,
            epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            val_metrics["macro_auc"],
            checkpoint_metric,
            " [new best]" if is_best else "",
        )

    if best_epoch == -1:
        raise RuntimeError(
            f"No epoch produced a finite '{checkpoint_metric}' validation score -- "
            "no checkpoint was ever saved."
        )

    del model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info(
        "Training finished. Best epoch=%d, smoothed %s=%.4f (window=%d)",
        best_epoch,
        checkpoint_metric,
        best_score,
        CHECKPOINT_SMOOTHING_WINDOW,
    )

    best_payload = load_checkpoint(checkpoint_path)
    final_model = rebuild_model_from_checkpoint(best_payload)
    final_model.to(device)

    test_metrics = evaluate_model(final_model, bundle.test_loader, device=device, num_classes=bundle.num_classes)
    dump_json(test_metrics, output_dir / "test_metrics.json")
    logger.info(
        "Test set evaluated once: accuracy=%.4f macro_f1=%.4f macro_auc=%.4f",
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
        test_metrics["macro_auc"],
    )

    class_names = list(bundle.classes)
    save_per_class_metrics_csv(test_metrics, class_names, output_dir / "per_class_metrics.csv")
    save_confusion_matrix_plot(test_metrics, class_names, output_dir / "confusion_matrix.png")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    flops_params = profile_flops_and_params(
        final_model, input_shape=(bundle.input_channels, image_size, image_size), device=device
    )
    dump_json(flops_params, output_dir / "flops_params.json")
    logger.info(
        "Model size: %.2fM params, %.3f GFLOPs at %dx%d",
        flops_params["params_million"],
        flops_params["flops_billion"],
        image_size,
        image_size,
    )

    dump_json(
        {
            "model_name": model_name,
            "image_size": image_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "label_smoothing": label_smoothing,
            "grad_clip_norm": grad_clip_norm,
            "checkpoint_metric": checkpoint_metric,
            "seed": seed,
            "split_indices_path": str(split_indices_path),
        },
        output_dir / "baseline_config.json",
    )
    dump_json(get_run_manifest(), output_dir / "run_manifest.json")

    return {
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(checkpoint_path),
        "validation_metrics": best_payload["validation_metrics"],
        "test_metrics": test_metrics,
        "flops_params": flops_params,
        "history": history,
    }
