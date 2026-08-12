"""One saved stratified train/validation split (handoff §3).

Every stage (candidate search, augmentation search, final training) that
needs a train/val split must reuse the exact indices saved here rather than
resplitting -- `build_dataset_bundle` in `loader.py` handles the
save-once/reuse-after logic; this module only knows how to compute and
serialize the split itself.

Two split rules live here:

- `stratified_split` treats every image as an independent sample.
- `grouped_stratified_split` additionally keeps all images sharing a group
  key (a patient / study) on the same side of the split.

The grouped rule exists because medical imaging datasets contain many
slices per patient, and those slices are *not* independent samples: they
share skull shape, ventricle geometry, scanner and protocol, and -- fatally
-- a label. A model can score on validation by recognising the patient
rather than the pathology. Measured on the Kaggle/SARTAJ data, 2870 images
were only ~1159 independent studies, 73% of validation images shared a
near-duplicate group with a training image, and removing that overlap cost
12 points of validation accuracy (22 points of validation glioma recall).
Published results put the same effect at 29-55% inflation for brain MRI,
with 96% "accuracy" reachable on randomly-labelled data under slice-level
splitting (Scientific Reports 11, 22544).

`stratified_split` is kept rather than deleted so the leaky-vs-clean
comparison stays runnable as an ablation (`DatasetConfig.group_aware_split`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from brainmri_nas.utils.serialization import dump_json, load_json


def stratified_split(
    targets: Sequence[int],
    validation_fraction: float,
    split_seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    all_indices = list(range(len(targets)))
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=validation_fraction,
        random_state=split_seed,
        stratify=targets,
    )
    return tuple(sorted(train_indices)), tuple(sorted(val_indices))


def grouped_stratified_split(
    targets: Sequence[int],
    groups: Sequence[str],
    validation_fraction: float,
    split_seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Stratify by class while keeping each group wholly on one side.

    Implemented with `StratifiedGroupKFold` (a maintained sklearn
    implementation) rather than a hand-rolled greedy pass: the fraction is
    realised as one fold of `round(1 / validation_fraction)`, so the achieved
    validation fraction is approximate. It cannot be exact anyway -- groups
    are indivisible, and one group here can hold hundreds of images.
    """
    if len(targets) != len(groups):
        raise ValueError(f"targets has {len(targets)} entries but groups has {len(groups)}.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}.")

    num_splits = max(2, round(1.0 / validation_fraction))
    distinct_groups = len(set(groups))
    if distinct_groups < num_splits:
        raise ValueError(
            f"Group-aware splitting needs at least {num_splits} distinct groups for a "
            f"{validation_fraction:.0%} validation split, but only {distinct_groups} were found. "
            "Use a larger validation_fraction, or disable group_aware_split."
        )

    splitter = StratifiedGroupKFold(n_splits=num_splits, shuffle=True, random_state=split_seed)
    train_indices, val_indices = next(
        iter(splitter.split(np.zeros(len(targets)), np.asarray(targets), groups=np.asarray(groups)))
    )
    return tuple(sorted(int(i) for i in train_indices)), tuple(sorted(int(i) for i in val_indices))


def save_split_indices(
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    path: str | Path,
) -> None:
    dump_json({"train_indices": list(train_indices), "val_indices": list(val_indices)}, path)


def load_split_indices(path: str | Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    data = load_json(path)
    return tuple(data["train_indices"]), tuple(data["val_indices"])
