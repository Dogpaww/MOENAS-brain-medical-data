"""Equal-conditions trial training for one augmentation policy (handoff §25).

Every policy trial must start from the *same* initial weights, see the same
data split, and use the same optimizer/scheduler/epoch budget -- the only
thing that varies between trials is the train transform. Callers are
responsible for building a fresh model, loading the shared initial state
dict into it, and deleting it after the trial (see `policy_search.py`); this
module only knows how to run one trial's train loop and evaluate it.

Predictions are moved to CPU per batch and concatenated once after the eval
loop (handoff §29/§30 items 14-15), even though this is a smaller-scale
trial loop than Stage 4's final training.

Validation macro AUC is (re-)evaluated at the end of *every* trial epoch,
not just once after the last one -- purely for visibility into how a
policy's score evolves during its trial (the handoff only prohibits
evaluating *test* every epoch; validation every epoch is fine, same as
final training already does). The value returned is still just the last
epoch's score, so this doesn't change which policy ends up selected.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from brainmri_nas.augment.genotype import AugmentationPolicy
from brainmri_nas.augment.transform_builder import build_augmented_train_transform
from brainmri_nas.utils.optim import build_optimizer_and_scheduler
from brainmri_nas.utils.progress import batch_log_interval, maybe_log_batch_progress

DEFAULT_LOGGER_NAME = "brainmri_nas.augmentation_search"


def build_policy_train_loader(
    train_dir: str | Path,
    train_indices: tuple[int, ...],
    *,
    image_size: int,
    batch_size: int,
    policy: AugmentationPolicy,
    num_workers: int = 0,
) -> DataLoader:
    transform = build_augmented_train_transform(policy, image_size)
    dataset = datasets.ImageFolder(str(train_dir), transform=transform)
    return DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers)


@torch.no_grad()
def evaluate_macro_auc(model: nn.Module, loader: DataLoader, *, device: torch.device, num_classes: int) -> float:
    was_training = model.training
    model.eval()

    all_probs = []
    all_targets = []
    for x, y in loader:
        x = x.to(device)
        probs = torch.softmax(model(x), dim=1).to("cpu")
        all_probs.append(probs)
        all_targets.append(y)

    model.train(was_training)

    probs = torch.cat(all_probs, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    try:
        return float(roc_auc_score(targets, probs, multi_class="ovr", average="macro", labels=list(range(num_classes))))
    except ValueError:
        # e.g. a class absent from a tiny validation split -- can't be scored, not a crash.
        return 0.0


def train_trial_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    num_classes: int,
    logger: logging.Logger | None = None,
) -> dict:
    logger = logger or logging.getLogger(DEFAULT_LOGGER_NAME)

    model.to(device)
    model.train()

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs
    )
    loss_fn = nn.CrossEntropyLoss()

    num_batches = len(train_loader)
    log_interval = batch_log_interval(num_batches)

    val_macro_auc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            maybe_log_batch_progress(
                logger,
                prefix="trial",
                epoch=epoch,
                total_epochs=epochs,
                batch_idx=step,
                total_batches=num_batches,
                running_loss=total_loss / total_samples,
                interval=log_interval,
            )

        scheduler.step()

        val_macro_auc = evaluate_macro_auc(model, val_loader, device=device, num_classes=num_classes)
        logger.info(
            "trial epoch %d/%d done: train_loss=%.4f val_macro_auc=%.4f",
            epoch,
            epochs,
            total_loss / max(total_samples, 1),
            val_macro_auc,
        )

    return {"val_macro_auc": val_macro_auc}
