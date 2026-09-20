"""Exhaustive search over the legacy fixed 2-op policy space.

The legacy GA search (`legacy_policy_search.py`) is the fixed-policy
baseline's own selection method, but its space is tiny: 21 unordered pairs of
the 7 operators. In the searched architecture's run it tried 10 of them and
spent 8 of its 25 trials on one pair, and that pair's 8 trials scored from
0.79 to 0.86 validation macro-AUC -- trial-to-trial noise as large as most
differences between pairs. So one trial per pair can't rank pairs reliably.

This evaluates every unordered pair `repeats` times with the unchanged legacy
trial protocol (same architecture, same shared initial weights, same
`train_trial_model`, same fitness) and selects the pair with the highest mean
validation macro-AUC. The result is written as `selected_legacy_policy.json`
in the legacy format, so it drops into any fixed-policy training run.

Details that are fixed on purpose:
  - Pairs are unordered and applied in menu order (`AUGMENTATION_OPS`).
    The legacy decoder distinguishes op1/op2 order; for these operators the
    order only changes which of two independent transforms runs first.
  - Every trial is reseeded with `config.augmentation.seed + repeat`, so all
    pairs within one repeat start from the same random state (same batch
    order): they are compared under matched conditions, and each repeat is
    an independent draw.
  - Repeats run in the outer loop, so an interruption leaves whole repeats.
  - The archive is saved after every trial and a rerun resumes from it; a
    rerun with different trial settings is refused.
  - The selection is written only once every trial has finished.
"""

from __future__ import annotations

import gc
import itertools
import logging
import statistics
from pathlib import Path

from brainmri_nas.augment.legacy_policy_search import _build_legacy_train_loader
from brainmri_nas.augment.search_space import AUGMENTATION_OPS
from brainmri_nas.augment.trial_training import train_trial_model
from brainmri_nas.data.loader import build_dataset_bundle
from brainmri_nas.model.network import build_model
from brainmri_nas.search_space.genotype import NetworkGenotype
from brainmri_nas.utils.config import Config, save_config
from brainmri_nas.utils.determinism import seed_everything
from brainmri_nas.utils.device import resolve_device
from brainmri_nas.utils.git_info import get_run_manifest
from brainmri_nas.utils.serialization import dump_json, load_json

ARCHIVE_FILENAME = "exhaustive_pair_archive.json"
SETTINGS_FILENAME = "exhaustive_pair_settings.json"
SUMMARY_FILENAME = "exhaustive_pair_summary.json"
SELECTED_FILENAME = "selected_legacy_policy.json"


def all_pairs() -> list[tuple[str, str]]:
    return list(itertools.combinations(AUGMENTATION_OPS, 2))


def summarize(archive: list[dict]) -> list[dict]:
    """Per pair: every score, mean and sample standard deviation, sorted by
    mean (best first; ties keep menu order)."""
    scores: dict[tuple[str, str], list[float]] = {}
    for record in archive:
        scores.setdefault(tuple(record["ops"]), []).append(record["val_macro_auc"])
    pair_order = {pair: i for i, pair in enumerate(all_pairs())}
    summary = [
        {
            "ops": list(pair),
            "num_evaluations": len(values),
            "val_macro_auc_mean": statistics.fmean(values),
            "val_macro_auc_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "val_macro_auc_values": values,
        }
        for pair, values in scores.items()
    ]
    return sorted(summary, key=lambda s: (-s["val_macro_auc_mean"], pair_order.get(tuple(s["ops"]), 0)))


def _configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("brainmri_nas.exhaustive_pair_search")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(output_dir / "exhaustive_pair_search.log", mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    return logger


def run_exhaustive_pair_search(
    config: Config,
    *,
    selected_architecture_path: str | Path,
    split_indices_path: str | Path,
    output_dir: str | Path,
    repeats: int,
    pairs: list[tuple[str, str]] | None = None,
) -> dict:
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}.")
    pairs = [tuple(p) for p in (pairs if pairs is not None else all_pairs())]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(output_dir)

    settings = {
        "pairs": [list(p) for p in pairs],
        "repeats": repeats,
        "base_seed": config.augmentation.seed,
        "trial_epochs": config.augmentation.trial_epochs,
        "learning_rate": config.augmentation.learning_rate,
        "weight_decay": config.augmentation.weight_decay,
        "momentum": config.augmentation.momentum,
        "image_size": config.dataset.image_size,
        "batch_size": config.dataset.batch_size,
        "selected_architecture_path": str(selected_architecture_path),
        "split_indices_path": str(split_indices_path),
    }
    settings_path = output_dir / SETTINGS_FILENAME
    archive_path = output_dir / ARCHIVE_FILENAME
    archive: list[dict] = []
    if archive_path.exists():
        previous = load_json(settings_path) if settings_path.exists() else None
        if previous is not None and {k: v for k, v in previous.items() if k != "repeats"} != {
            k: v for k, v in settings.items() if k != "repeats"
        }:
            raise ValueError(
                f"{output_dir} holds trials run with different settings ({settings_path}); "
                "use a fresh output directory."
            )
        archive = load_json(archive_path)
        logger.info("Resuming: %d trials already in %s", len(archive), archive_path)
    dump_json(settings, settings_path)
    done = {(tuple(r["ops"]), r["repeat"]) for r in archive}

    # Same sequence as the legacy search, so the shared initial weights match it.
    seed_everything(config.augmentation.seed)
    device = resolve_device(config.augmentation.device)
    selected_architecture = load_json(selected_architecture_path)
    genotype = NetworkGenotype.from_dict(selected_architecture["genotype"])
    bundle = build_dataset_bundle(
        config.dataset.data_root,
        image_size=config.dataset.image_size,
        validation_fraction=config.dataset.validation_fraction,
        split_seed=config.dataset.split_seed,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        split_indices_path=split_indices_path,
        group_aware_split=config.dataset.group_aware_split,
    )
    model_build_kwargs = dict(
        input_channels=config.dataset.input_channels,
        num_classes=bundle.num_classes,
        image_size=config.dataset.image_size,
        initial_channels=selected_architecture["initial_channels"],
        number_of_cells=selected_architecture["number_of_cells"],
        drop_path_probability=config.search_space.drop_path_probability,
        stem_type=config.search_space.stem_type,
    )
    base_model = build_model(genotype, **model_build_kwargs)
    initial_state = {name: value.detach().cpu().clone() for name, value in base_model.state_dict().items()}
    del base_model

    total = len(pairs) * repeats
    train_dir = Path(config.dataset.data_root) / "Training"
    logger.info("Exhaustive pair search on device=%s: %d pairs x %d repeats = %d trials", device, len(pairs), repeats, total)

    for repeat in range(repeats):
        seed = config.augmentation.seed + repeat
        for op1, op2 in pairs:
            if ((op1, op2), repeat) in done:
                continue
            seed_everything(seed)
            label = f"[{len(archive) + 1}/{total}] repeat {repeat + 1} (seed {seed}) {op1}+{op2}"
            logger.info("%s: starting", label)

            train_loader = _build_legacy_train_loader(
                train_dir,
                bundle.train_indices,
                op1=op1,
                op2=op2,
                image_size=config.dataset.image_size,
                batch_size=config.dataset.batch_size,
                num_workers=config.dataset.num_workers,
            )
            trial_model = build_model(genotype, **model_build_kwargs)
            trial_model.load_state_dict(initial_state)
            result = train_trial_model(
                trial_model,
                train_loader,
                bundle.val_loader,
                epochs=config.augmentation.trial_epochs,
                learning_rate=config.augmentation.learning_rate,
                weight_decay=config.augmentation.weight_decay,
                momentum=config.augmentation.momentum,
                device=device,
                num_classes=bundle.num_classes,
                loss_cache=None,
                logger=logger,
            )
            del trial_model, train_loader
            gc.collect()

            archive.append({"ops": [op1, op2], "repeat": repeat, "seed": seed, "val_macro_auc": result["val_macro_auc"]})
            dump_json(archive, archive_path)
            logger.info("%s: val_macro_auc=%.4f", label, result["val_macro_auc"])

    # Only this run's pairs and repeats count, even if the archive holds more.
    requested = [r for r in archive if tuple(r["ops"]) in set(pairs) and r["repeat"] < repeats]
    summary = summarize(requested)
    dump_json(summary, output_dir / SUMMARY_FILENAME)
    save_config(config, output_dir / "augmentation_config.yaml")
    dump_json(get_run_manifest(), output_dir / "run_manifest.json")

    best = summary[0]
    selected = {
        "ops": best["ops"],
        "val_macro_auc": best["val_macro_auc_mean"],
        "val_macro_auc_std": best["val_macro_auc_std"],
        "num_evaluations": best["num_evaluations"],
        "selection": f"highest mean validation macro-AUC over {repeats} repeats of every pair",
        "chromosome": None,
    }
    dump_json(selected, output_dir / SELECTED_FILENAME)
    logger.info("Selected %s: mean val_macro_auc=%.4f (std %.4f)", "+".join(best["ops"]), best["val_macro_auc_mean"], best["val_macro_auc_std"])
    for rank, s in enumerate(summary, start=1):
        logger.info("  %2d. %-40s mean=%.4f std=%.4f n=%d", rank, "+".join(s["ops"]), s["val_macro_auc_mean"], s["val_macro_auc_std"], s["num_evaluations"])

    return {"selected_policy": selected, "summary": summary, "archive": archive}
