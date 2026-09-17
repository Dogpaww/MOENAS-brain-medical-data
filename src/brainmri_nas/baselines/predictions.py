"""Recreate per-sample test predictions for a finished baseline run from its
saved checkpoint.

Runs trained before `test_predictions.csv` existed only saved aggregate
metrics, but they did keep `best_checkpoint.pt`. Rebuilding that checkpoint
and re-evaluating the test set repeats exactly what `train_baseline.py` did
at the end of training (FP32, eval mode, unshuffled loader, deterministic
eval transform), so the predictions can be recovered without retraining.

"Exactly" is checked, not assumed: the recomputed confusion matrix must equal
the saved one and macro-AUC must agree within `AUC_TOLERANCE`. Only then is
the CSV written. A mismatch writes nothing, because those predictions would
not be the ones behind the reported numbers.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from brainmri_nas.baselines.checkpoint import load_checkpoint, rebuild_model_from_checkpoint
from brainmri_nas.data.loader import build_dataset_bundle
from brainmri_nas.training.evaluate import evaluate_model, save_test_predictions_csv
from brainmri_nas.utils.config import Config

PREDICTIONS_FILENAME = "test_predictions.csv"
# GPU kernels are not bit-deterministic, so probabilities can differ in the
# last few decimals between two evaluations of the same weights. Anything
# larger than this means something other than kernel noise changed.
AUC_TOLERANCE = 1e-6


def backfill_run_predictions(run_dir: str | Path, config: Config, device: torch.device, *, force: bool = False) -> dict:
    run_dir = Path(run_dir)
    predictions_path = run_dir / PREDICTIONS_FILENAME
    result = {"run_dir": str(run_dir)}

    if predictions_path.exists() and not force:
        return {**result, "status": "skipped", "reason": f"{PREDICTIONS_FILENAME} already exists"}
    missing = [
        name
        for name in ("best_checkpoint.pt", "test_metrics.json", "baseline_config.json", "split_indices.json")
        if not (run_dir / name).exists()
    ]
    if missing:
        return {**result, "status": "skipped", "reason": f"missing {', '.join(missing)}"}

    baseline_config = json.loads((run_dir / "baseline_config.json").read_text())
    saved_metrics = json.loads((run_dir / "test_metrics.json").read_text())

    bundle = build_dataset_bundle(
        config.dataset.data_root,
        image_size=baseline_config["image_size"],
        validation_fraction=config.dataset.validation_fraction,
        split_seed=config.dataset.split_seed,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        split_indices_path=run_dir / "split_indices.json",
        group_aware_split=config.dataset.group_aware_split,
    )

    payload = load_checkpoint(run_dir / "best_checkpoint.pt")
    if dict(payload["class_to_idx"]) != dict(bundle.class_to_idx):
        return {**result, "status": "mismatch", "reason": "checkpoint class_to_idx differs from the dataset's"}
    model = rebuild_model_from_checkpoint(payload).to(device)

    metrics, predictions = evaluate_model(
        model, bundle.test_loader, device=device, num_classes=bundle.num_classes, return_predictions=True
    )
    del model, payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    auc_difference = abs(metrics["macro_auc"] - saved_metrics["macro_auc"])
    result.update(
        saved_accuracy=saved_metrics["accuracy"],
        recomputed_accuracy=metrics["accuracy"],
        auc_difference=auc_difference,
    )
    if metrics["confusion_matrix"] != saved_metrics["confusion_matrix"]:
        return {
            **result,
            "status": "mismatch",
            "reason": f"confusion matrix {metrics['confusion_matrix']} != saved {saved_metrics['confusion_matrix']}",
        }
    if auc_difference > AUC_TOLERANCE:
        return {**result, "status": "mismatch", "reason": f"macro-AUC differs by {auc_difference:.2e}"}

    save_test_predictions_csv(
        predictions, bundle.test_loader.dataset.samples, list(bundle.classes), config.dataset.data_root, predictions_path
    )
    return {**result, "status": "written"}
