"""Full evaluation metric suite (handoff §29).

Because the classes are imbalanced, accuracy alone is never reported --
macro precision/recall/F1/AUC, per-class breakdowns, and a confusion matrix
always come with it. Prediction tensors are moved to CPU immediately after
each forward pass and concatenated exactly once after the loop (handoff
§30 items 14-15: never accumulate on GPU, never repeatedly concatenate a
growing tensor inside the loop).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    loss_fn: nn.Module | None = None,
    return_predictions: bool = False,
) -> dict | tuple[dict, dict]:
    """Aggregate metrics over `loader`.

    With `return_predictions=True`, returns `(metrics, predictions)` from the
    same single pass, where `predictions` holds one entry per sample in loader
    order: `targets`, `predicted`, and softmax `probabilities`. Paired tests
    such as McNemar's need these; the aggregates can't say which images two
    models disagree on. Returned separately so they never end up in
    test_metrics.json by accident.
    """
    was_training = model.training
    model.eval()
    loss_fn = loss_fn or nn.CrossEntropyLoss()

    per_batch_logits: list[torch.Tensor] = []
    per_batch_targets: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x_device, y_device = x.to(device), y.to(device)
        logits = model(x_device)
        loss = loss_fn(logits, y_device)

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
        per_batch_logits.append(logits.detach().to("cpu"))
        per_batch_targets.append(y)  # loader tensors are already CPU-resident

    model.train(was_training)

    logits = torch.cat(per_batch_logits, dim=0)
    targets = torch.cat(per_batch_targets, dim=0)

    probs = torch.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()
    targets_np = targets.numpy()
    labels = list(range(num_classes))

    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        targets_np, preds, average=None, labels=labels, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets_np, preds, average="macro", labels=labels, zero_division=0
    )

    try:
        macro_auc = float(roc_auc_score(targets_np, probs, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        # e.g. a class absent from this split -- can't be scored, not a crash.
        macro_auc = float("nan")

    cm = confusion_matrix(targets_np, preds, labels=labels)

    metrics = {
        "num_samples": total_samples,
        "loss": total_loss / max(total_samples, 1),
        "accuracy": float((preds == targets_np).mean()),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "macro_auc": macro_auc,
        "per_class": {
            "precision": per_class_precision.tolist(),
            "recall": per_class_recall.tolist(),
            "f1": per_class_f1.tolist(),
            "support": per_class_support.tolist(),
        },
        "confusion_matrix": cm.tolist(),
    }

    if not return_predictions:
        return metrics

    predictions = {
        "targets": targets_np.tolist(),
        "predicted": preds.tolist(),
        "probabilities": probs.tolist(),
    }
    return metrics, predictions


def save_test_predictions_csv(
    predictions: dict,
    samples: list[tuple[str, int]],
    class_names: list[str],
    data_root: str | Path,
    path: str | Path,
) -> None:
    """One row per test image: path relative to `data_root` (so the file is
    the same on any machine), true and predicted class, whether it was
    correct, and the per-class probabilities.

    `samples` is the evaluated `ImageFolder.samples`. It lines up with
    `predictions` only because the test loader is never shuffled, so that is
    checked row by row against the loader's own targets rather than assumed.
    """
    import os

    import pandas as pd

    targets = predictions["targets"]
    if len(samples) != len(targets):
        raise ValueError(f"{len(samples)} samples but {len(targets)} predictions; loader and dataset disagree.")
    for (sample_path, sample_label), target in zip(samples, targets):
        if sample_label != target:
            raise ValueError(
                f"{sample_path} has label {sample_label} but the loader yielded {target} at that position; "
                "predictions are not in dataset order (was the loader shuffled?)."
            )

    rows = []
    for (sample_path, _), target, predicted, probabilities in zip(
        samples, targets, predictions["predicted"], predictions["probabilities"]
    ):
        row = {
            "image": os.path.relpath(sample_path, data_root),
            "true_label": class_names[target],
            "predicted_label": class_names[predicted],
            "correct": int(target == predicted),
        }
        row.update({f"prob_{name}": p for name, p in zip(class_names, probabilities)})
        rows.append(row)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_per_class_metrics_csv(metrics: dict, class_names: list[str], path: str | Path) -> None:
    import pandas as pd

    per_class = metrics["per_class"]
    df = pd.DataFrame(
        {
            "class": class_names,
            "precision": per_class["precision"],
            "recall": per_class["recall"],
            "f1": per_class["f1"],
            "support": per_class["support"],
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_confusion_matrix_plot(metrics: dict, class_names: list[str], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
