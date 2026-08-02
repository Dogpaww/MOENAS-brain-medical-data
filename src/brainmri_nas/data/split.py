"""One saved stratified train/validation split (handoff §3).

Every stage (candidate search, augmentation search, final training) that
needs a train/val split must reuse the exact indices saved here rather than
resplitting -- `build_dataset_bundle` in `loader.py` handles the
save-once/reuse-after logic; this module only knows how to compute and
serialize the split itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from sklearn.model_selection import train_test_split

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


def save_split_indices(
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    path: str | Path,
) -> None:
    dump_json({"train_indices": list(train_indices), "val_indices": list(val_indices)}, path)


def load_split_indices(path: str | Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    data = load_json(path)
    return tuple(data["train_indices"]), tuple(data["val_indices"])
