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
from brainmri_nas.augment.trial_training import build_policy_train_loader
from brainmri_nas.data.loader import build_dataset_bundle
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
from brainmri_nas.utils.optim import build_optimizer_and_scheduler
from brainmri_nas.utils.serialization import dump_json, load_json


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
    num_workers: int,
) -> DataLoader:
    if policy is not None:
        return build_policy_train_loader(
            train_dir,
            train_indices,
            image_size=image_size,
            batch_size=batch_size,
            policy=policy,
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
        initial_channels=config.search_space.initial_channels,
        number_of_cells=config.search_space.number_of_cells,
        drop_path_probability=config.search_space.drop_path_probability,
        stem_type=config.search_space.stem_type,
    )

    model = build_model(genotype, **model_config)  # fresh weights, no reuse from search/proxies/augmentation trials
    model.to(device)

    train_dir = Path(config.dataset.data_root) / "Training"
    train_loader = _build_train_loader(
        train_dir,
        bundle.train_indices,
        image_size=config.dataset.image_size,
        batch_size=config.training.physical_batch_size,
        policy=policy,
        num_workers=config.dataset.num_workers,
    )

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        epochs=config.training.final_epochs,
    )
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    checkpoint_path = output_dir / "best_checkpoint.pt"
    best_score = float("-inf")
    best_epoch = -1
    history: list[dict] = []

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
        )
        scheduler.step()

        # Validation only during training -- test data is never touched here (handoff §29/§30 item 12).
        val_metrics = evaluate_model(model, bundle.val_loader, device=device, num_classes=bundle.num_classes)
        score = val_metrics[config.training.checkpoint_metric]

        is_best = math.isfinite(score) and score > best_score
        if is_best:
            best_score = score
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
                "learning_rate": scheduler.get_last_lr()[0],
                "is_best": is_best,
            }
        )
        logger.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f val_%s=%.4f%s",
            epoch,
            config.training.final_epochs,
            train_loss,
            val_metrics["loss"],
            config.training.checkpoint_metric,
            score if math.isfinite(score) else float("nan"),
            " (best)" if is_best else "",
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

    logger.info("Training finished. Best epoch=%d, %s=%.4f", best_epoch, config.training.checkpoint_metric, best_score)

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
