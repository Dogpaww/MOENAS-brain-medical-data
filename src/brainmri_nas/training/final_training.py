"""Final training orchestration (handoff §27, §33).

Flow: load the selected chromosome/genotype -> rebuild the model from
scratch (no weights carried over from SynFlow/ZiCO/NSGA-II/augmentation
trials) -> apply the selected augmentation policy -> train with
validation-only per-epoch evaluation -> track the best checkpoint via a
fresh CPU state-dict snapshot every time validation improves -> reload that
checkpoint into an independently rebuilt model -> evaluate the test set
exactly once. Every required output file (handoff §33) is written to
`output_dir`.
"""

from __future__ import annotations

import gc
import logging
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from brainmri_nas.augment.genotype import AugmentationPolicy
from brainmri_nas.augment.sample_adaptive_dataset import build_sample_adaptive_loader
from brainmri_nas.data.loader import build_dataset_bundle, compute_class_weights
from brainmri_nas.data.transforms import build_train_transform
from brainmri_nas.model.network import build_model
from brainmri_nas.proxies.profiling import profile_peak_memory
from brainmri_nas.search_space.genotype import NetworkGenotype
from brainmri_nas.training.checkpoint import load_checkpoint, rebuild_model_from_checkpoint, save_checkpoint
from brainmri_nas.training.engine import train_one_epoch
from brainmri_nas.training.evaluate import evaluate_model, save_confusion_matrix_plot, save_per_class_metrics_csv
from brainmri_nas.utils.config import Config, save_config
from brainmri_nas.utils.determinism import seed_everything
from brainmri_nas.utils.device import resolve_device
from brainmri_nas.utils.git_info import get_run_manifest
from brainmri_nas.utils.loss_cache import LossCache
from brainmri_nas.utils.optim import build_optimizer_and_scheduler
from brainmri_nas.utils.serialization import dump_json, load_json

# Matches trial_training.py's FITNESS_SMOOTHING_WINDOW: with 250 epochs to
# pick a "best" checkpoint from against a ~15-20% validation split, a single
# epoch's raw score is a high-variance estimator (real runs show the val
# metric swinging noticeably between adjacent epochs) -- checkpointing on a
# rolling average of the last few epochs instead makes the selected
# checkpoint reflect a genuine plateau rather than one lucky epoch.
CHECKPOINT_SMOOTHING_WINDOW = 3


def _configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("brainmri_nas.final_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(output_dir / "training.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    return logger


def _build_train_loader(
    train_dir: Path,
    train_indices: tuple[int, ...],
    *,
    image_size: int,
    batch_size: int,
    policy: AugmentationPolicy | None,
    loss_cache: LossCache | None,
    num_workers: int,
) -> DataLoader:
    if policy is not None:
        return build_sample_adaptive_loader(
            train_dir,
            train_indices,
            image_size=image_size,
            batch_size=batch_size,
            policy=policy,
            loss_cache=loss_cache,
            num_workers=num_workers,
        )
    dataset = datasets.ImageFolder(str(train_dir), transform=build_train_transform(image_size))
    return DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers)


def run_final_training(
    config: Config,
    *,
    selected_architecture_path: str | Path,
    split_indices_path: str | Path,
    output_dir: str | Path,
    selected_policy_path: str | Path | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(output_dir)

    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)

    use_amp = config.training.precision == "amp" and device.type == "cuda"
    if config.training.precision == "amp" and not use_amp:
        logger.info("precision=amp requested but device=%s is not CUDA -- training in FP32.", device)
    logger.info(
        "Starting final training on device=%s, amp=%s, physical_batch_size=%d, "
        "accumulation_steps=%d, effective_batch_size=%d",
        device,
        use_amp,
        config.training.physical_batch_size,
        config.training.gradient_accumulation_steps,
        config.training.physical_batch_size * config.training.gradient_accumulation_steps,
    )

    selected_architecture = load_json(selected_architecture_path)
    genotype = NetworkGenotype.from_dict(selected_architecture["genotype"])

    selected_policy_record = load_json(selected_policy_path) if selected_policy_path is not None else None
    policy = AugmentationPolicy.from_dict(selected_policy_record["policy"]) if selected_policy_record else None
    logger.info("Augmentation policy: %s", "from " + str(selected_policy_path) if policy else "none (identity)")

    bundle = build_dataset_bundle(
        config.dataset.data_root,
        image_size=config.dataset.image_size,
        validation_fraction=config.dataset.validation_fraction,
        split_seed=config.dataset.split_seed,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        split_indices_path=split_indices_path,  # must already exist -- reuse the search-time split (handoff §3)
    )

    model_config = dict(
        input_channels=config.dataset.input_channels,
        num_classes=bundle.num_classes,
        image_size=config.dataset.image_size,
        initial_channels=selected_architecture["initial_channels"],
        number_of_cells=selected_architecture["number_of_cells"],
        drop_path_probability=config.search_space.drop_path_probability,
        stem_type=config.search_space.stem_type,
        classifier_dropout_probability=config.training.classifier_dropout_probability,
    )

    model = build_model(genotype, **model_config)  # fresh weights, no reuse from search/proxies/augmentation trials
    model.to(device)

    class_weights = compute_class_weights(config.dataset.data_root, bundle.class_to_idx, bundle.train_indices)
    logger.info(
        "Class weights (balanced): %s",
        {cls: round(class_weights[idx].item(), 3) for cls, idx in bundle.class_to_idx.items()},
    )

    no_tumor_class_index = bundle.class_to_idx.get("no_tumor")
    if no_tumor_class_index is None:
        raise ValueError(f"Expected a 'no_tumor' class in class_to_idx, got {bundle.class_to_idx}.")
    logger.info(
        "Cancer->no_tumor false-negative penalty: %.3f (no_tumor class index=%d)",
        config.training.cancer_no_tumor_penalty,
        no_tumor_class_index,
    )

    # Only meaningful when a policy is actually being applied -- sized to the
    # full training split (not the physical batch size) since it tracks one
    # loss value per training sample, refreshed every ~1/40th of the run.
    loss_cache = (
        LossCache(num_samples=len(bundle.train_indices), total_epochs=config.training.final_epochs)
        if policy is not None
        else None
    )

    train_dir = Path(config.dataset.data_root) / "Training"
    train_loader = _build_train_loader(
        train_dir,
        bundle.train_indices,
        image_size=config.dataset.image_size,
        batch_size=config.training.physical_batch_size,
        policy=policy,
        loss_cache=loss_cache,
        num_workers=config.dataset.num_workers,
    )

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        epochs=config.training.final_epochs,
        momentum=config.training.momentum,
    )
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    checkpoint_path = output_dir / "best_checkpoint.pt"
    best_score = float("-inf")
    best_epoch = -1
    history: list[dict] = []
    recent_scores: list[float] = []

    for epoch in range(1, config.training.final_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            accumulation_steps=config.training.gradient_accumulation_steps,
            grad_clip_norm=config.training.grad_clip_norm,
            epoch=epoch,
            total_epochs=config.training.final_epochs,
            loss_cache=loss_cache,
            class_weights=class_weights,
            no_tumor_class_index=no_tumor_class_index,
            cancer_no_tumor_penalty=config.training.cancer_no_tumor_penalty,
            logger=logger,
        )
        scheduler.step()

        # Validation only during training -- test data is never touched here (handoff §29/§30 item 12).
        val_metrics = evaluate_model(model, bundle.val_loader, device=device, num_classes=bundle.num_classes)
        score = val_metrics[config.training.checkpoint_metric]
        recent_scores.append(score)

        # Rolling average over the last CHECKPOINT_SMOOTHING_WINDOW epochs
        # (partial window early on) decides "best", not this epoch's raw
        # score -- see CHECKPOINT_SMOOTHING_WINDOW's docstring above. The
        # checkpoint itself still saves this epoch's actual raw weights and
        # raw val_metrics; only the *decision* of whether to save is smoothed.
        window = [s for s in recent_scores[-CHECKPOINT_SMOOTHING_WINDOW:] if math.isfinite(s)]
        smoothed_score = sum(window) / len(window) if window else float("-inf")

        is_best = smoothed_score > best_score
        if is_best:
            best_score = smoothed_score
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model=model,
                genotype=genotype,
                model_config=model_config,
                class_to_idx=bundle.class_to_idx,
                image_size=config.dataset.image_size,
                epoch=epoch,
                validation_metrics=val_metrics,
                chromosome=selected_architecture.get("chromosome"),
                augmentation_policy=selected_policy_record["policy"] if selected_policy_record else None,
            )
            dump_json(val_metrics, output_dir / "validation_metrics.json")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
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
            config.training.final_epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            val_metrics["macro_auc"],
            config.training.checkpoint_metric,
            " [new best]" if is_best else "",
        )

    if best_epoch == -1:
        raise RuntimeError(
            f"No epoch produced a finite '{config.training.checkpoint_metric}' validation score -- "
            "no checkpoint was ever saved."
        )

    del model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info(
        "Training finished. Best epoch=%d, smoothed %s=%.4f (window=%d)",
        best_epoch,
        config.training.checkpoint_metric,
        best_score,
        CHECKPOINT_SMOOTHING_WINDOW,
    )

    # Reload the best checkpoint into an independently rebuilt model -- never
    # reuse the (already-past-its-best-epoch) training-loop model object.
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

    memory_profile = profile_peak_memory(
        final_model,
        input_shape=(config.dataset.input_channels, config.dataset.image_size, config.dataset.image_size),
        num_classes=bundle.num_classes,
        batch_size=config.training.physical_batch_size,
        device=device,
        precision=config.training.precision,
    )
    dump_json(memory_profile, output_dir / "memory_profile.json")

    save_config(config, output_dir / "training_config.yaml")
    dump_json(selected_architecture, output_dir / "selected_architecture.json")
    if selected_policy_record is not None:
        dump_json(selected_policy_record, output_dir / "selected_policy.json")
    dump_json(get_run_manifest(), output_dir / "run_manifest.json")

    return {
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(checkpoint_path),
        "validation_metrics": best_payload["validation_metrics"],
        "test_metrics": test_metrics,
        "history": history,
    }
