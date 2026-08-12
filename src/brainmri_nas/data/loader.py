"""Dataset validation and loading (handoff §2-4).

`build_dataset_bundle` is the single entry point: it validates the expected
`Training/`+`Testing/` folder structure, verifies `Training` and `Testing`
resolve to the *same* `class_to_idx` mapping via `ImageFolder` itself
(never hardcoded), builds one saved stratified split of `Training` alone
(handoff §3: `Testing` is only ever used as the final, held-out test set),
and returns a `DatasetBundle`.

Train and train-eval use two separate `ImageFolder` instances (one per
transform) over the same directory, so the training split's augmenting
transform is never accidentally handed to a validation/eval view of that
same data.

If a `patient_ids.json` sidecar exists in `data_root`, the train/val split
becomes group-aware: every image sharing a patient/study key stays wholly on
one side, so validation measures generalisation to unseen patients rather
than to unseen slices of patients already memorised (see `split.py` for the
measured size of that effect). The file maps a `data_root`-relative POSIX
path to a group key:

    {"Training/glioma_tumor/000001.png": "patient_042", ...}

Datasets without the sidecar keep the plain per-image stratified split, and
`group_aware_split=False` forces it even when the sidecar is present -- that
combination is the leaky-vs-clean ablation.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from brainmri_nas.data.bundle import DatasetBundle
from brainmri_nas.data.split import (
    grouped_stratified_split,
    load_split_indices,
    save_split_indices,
    stratified_split,
)
from brainmri_nas.data.transforms import build_eval_transform, build_train_transform
from brainmri_nas.utils.serialization import load_json

TRAIN_SPLIT_DIR_NAME = "Training"
TEST_SPLIT_DIR_NAME = "Testing"
PATIENT_IDS_FILE_NAME = "patient_ids.json"


class DatasetValidationError(Exception):
    pass


def _class_subfolders(directory: Path) -> set[str]:
    return {p.name for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")}


def _count_images(directory: Path) -> int:
    return sum(1 for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))


def _validate_dataset_root(data_root: Path) -> tuple[Path, Path]:
    train_dir = data_root / TRAIN_SPLIT_DIR_NAME
    test_dir = data_root / TEST_SPLIT_DIR_NAME

    for split_name, split_dir in ((TRAIN_SPLIT_DIR_NAME, train_dir), (TEST_SPLIT_DIR_NAME, test_dir)):
        if not split_dir.is_dir():
            raise DatasetValidationError(f"Expected a '{split_name}' folder at {split_dir}, but it does not exist.")

    train_classes = _class_subfolders(train_dir)
    test_classes = _class_subfolders(test_dir)

    if not train_classes:
        raise DatasetValidationError(f"No class subfolders found under {train_dir}.")
    if train_classes != test_classes:
        raise DatasetValidationError(
            f"Training classes {sorted(train_classes)} do not match Testing classes {sorted(test_classes)}."
        )

    for split_name, split_dir, classes in (
        (TRAIN_SPLIT_DIR_NAME, train_dir, train_classes),
        (TEST_SPLIT_DIR_NAME, test_dir, test_classes),
    ):
        for cls in classes:
            if _count_images(split_dir / cls) == 0:
                raise DatasetValidationError(f"Class '{cls}' in {split_name} has no images.")

    return train_dir, test_dir


def load_patient_ids(data_root: str | Path) -> dict[str, str] | None:
    """The `patient_ids.json` sidecar, or None when the dataset ships without one."""
    path = Path(data_root) / PATIENT_IDS_FILE_NAME
    if not path.exists():
        return None
    mapping = load_json(path)
    if not isinstance(mapping, dict) or not mapping:
        raise DatasetValidationError(f"{path} must be a non-empty object mapping relative paths to group ids.")
    return {str(k): str(v) for k, v in mapping.items()}


def _group_keys_for_samples(
    samples: list[tuple[str, int]], data_root: Path, patient_ids: dict[str, str]
) -> list[str]:
    """Group key per ImageFolder sample, in sample order.

    Missing entries raise rather than silently falling back to a per-image
    key: a partially-populated sidecar would leak exactly the patients it
    failed to cover, which is the failure this whole mechanism exists to stop.
    """
    keys, missing = [], []
    for path, _ in samples:
        relative = Path(path).resolve().relative_to(data_root.resolve()).as_posix()
        if relative not in patient_ids:
            missing.append(relative)
        else:
            keys.append(patient_ids[relative])
    if missing:
        raise DatasetValidationError(
            f"{len(missing)} image(s) are missing from {PATIENT_IDS_FILE_NAME}, "
            f"e.g. {missing[:3]}. Every training image needs a group id, or the "
            "split silently leaks the uncovered ones."
        )
    return keys


def describe_dataset(data_root: str | Path) -> dict:
    """Per-class image counts for Training/Testing (handoff §4 item 2, §33 dataset_report.json)."""
    data_root = Path(data_root)
    train_dir, test_dir = _validate_dataset_root(data_root)
    report: dict = {"data_root": str(data_root), "splits": {}}
    for split_name, split_dir in ((TRAIN_SPLIT_DIR_NAME, train_dir), (TEST_SPLIT_DIR_NAME, test_dir)):
        classes = sorted(_class_subfolders(split_dir))
        report["splits"][split_name] = {cls: _count_images(split_dir / cls) for cls in classes}
    return report


def build_dataset_bundle(
    data_root: str | Path,
    *,
    image_size: int,
    validation_fraction: float,
    split_seed: int,
    batch_size: int,
    num_workers: int = 0,
    split_indices_path: str | Path | None = None,
    group_aware_split: bool = True,
) -> DatasetBundle:
    data_root = Path(data_root)
    train_dir, test_dir = _validate_dataset_root(data_root)

    train_transform = build_train_transform(image_size)
    eval_transform = build_eval_transform(image_size)

    train_dataset_for_train = datasets.ImageFolder(str(train_dir), transform=train_transform)
    train_dataset_for_eval = datasets.ImageFolder(str(train_dir), transform=eval_transform)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=eval_transform)

    if train_dataset_for_train.class_to_idx != test_dataset.class_to_idx:
        raise DatasetValidationError(
            f"Training class_to_idx {train_dataset_for_train.class_to_idx} does not match "
            f"Testing class_to_idx {test_dataset.class_to_idx}. Re-run with matching class folder names."
        )

    class_to_idx = train_dataset_for_train.class_to_idx
    classes = tuple(sorted(class_to_idx, key=class_to_idx.get))

    targets = [label for _, label in train_dataset_for_train.samples]

    patient_ids = load_patient_ids(data_root) if group_aware_split else None

    if split_indices_path is not None and Path(split_indices_path).exists():
        train_indices, val_indices = load_split_indices(split_indices_path)
    else:
        if patient_ids is not None:
            group_keys = _group_keys_for_samples(train_dataset_for_train.samples, data_root, patient_ids)
            train_indices, val_indices = grouped_stratified_split(
                targets, group_keys, validation_fraction, split_seed
            )
        else:
            train_indices, val_indices = stratified_split(targets, validation_fraction, split_seed)
        if split_indices_path is not None:
            save_split_indices(train_indices, val_indices, split_indices_path)

    overlap = set(train_indices) & set(val_indices)
    if overlap:
        raise DatasetValidationError(f"Loaded split has {len(overlap)} overlapping train/val indices.")

    # Guard the invariant itself, not just the code path that usually provides
    # it -- a split loaded from disk may predate group-aware splitting, and
    # silently training on a leaky saved split is the exact failure mode this
    # is here to prevent.
    if patient_ids is not None:
        group_keys = _group_keys_for_samples(train_dataset_for_train.samples, data_root, patient_ids)
        shared = {group_keys[i] for i in train_indices} & {group_keys[i] for i in val_indices}
        if shared:
            raise DatasetValidationError(
                f"{len(shared)} group(s) appear in BOTH train and validation "
                f"(e.g. {sorted(shared)[:3]}). If this split came from "
                f"{split_indices_path}, it predates group-aware splitting -- delete it and re-split."
            )

    train_loader = DataLoader(
        Subset(train_dataset_for_train, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    train_eval_loader = DataLoader(
        Subset(train_dataset_for_eval, train_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_loader = DataLoader(
        Subset(train_dataset_for_eval, val_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return DatasetBundle(
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=classes,
        class_to_idx=dict(class_to_idx),
        train_indices=tuple(train_indices),
        val_indices=tuple(val_indices),
        input_channels=3,
        num_classes=len(classes),
    )
