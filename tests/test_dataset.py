from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from torchvision import datasets

from brainmri_nas.data.loader import DatasetValidationError, build_dataset_bundle, describe_dataset
from brainmri_nas.data.split import stratified_split
from brainmri_nas.data.transforms import build_eval_transform, build_train_transform
from conftest import CLASSES, TEST_COUNTS, TRAIN_COUNTS


def test_expected_classes_present(synthetic_dataset_root: Path):
    report = describe_dataset(synthetic_dataset_root)
    assert set(report["splits"]["Training"]) == set(CLASSES)
    assert set(report["splits"]["Testing"]) == set(CLASSES)
    assert report["splits"]["Training"] == TRAIN_COUNTS
    assert report["splits"]["Testing"] == TEST_COUNTS


def test_missing_testing_folder_raises(synthetic_dataset_root: Path):
    shutil.rmtree(synthetic_dataset_root / "Testing")
    with pytest.raises(DatasetValidationError, match="Testing"):
        describe_dataset(synthetic_dataset_root)


def test_empty_class_folder_raises(synthetic_dataset_root: Path):
    empty_cls_dir = synthetic_dataset_root / "Training" / "glioma_tumor"
    for f in empty_cls_dir.iterdir():
        f.unlink()
    with pytest.raises(DatasetValidationError, match="glioma_tumor"):
        describe_dataset(synthetic_dataset_root)


def test_mismatched_classes_between_splits_raises(synthetic_dataset_root: Path):
    (synthetic_dataset_root / "Testing" / "extra_class").mkdir()
    (synthetic_dataset_root / "Testing" / "extra_class" / "img.png").write_bytes(b"not a real image but nonempty")
    with pytest.raises(DatasetValidationError):
        describe_dataset(synthetic_dataset_root)


def test_split_is_stratified_and_deterministic(synthetic_dataset_root: Path):
    dataset = datasets.ImageFolder(str(synthetic_dataset_root / "Training"))
    targets = [label for _, label in dataset.samples]

    train_a, val_a = stratified_split(targets, validation_fraction=0.2, split_seed=7)
    train_b, val_b = stratified_split(targets, validation_fraction=0.2, split_seed=7)
    assert train_a == train_b
    assert val_a == val_b

    train_c, _ = stratified_split(targets, validation_fraction=0.2, split_seed=999)
    assert train_a != train_c  # different seed should (almost certainly) differ

    # Stratification: each class's val fraction should be close to the requested fraction.
    num_classes = len(dataset.classes)
    val_targets = [targets[i] for i in val_a]
    for class_idx in range(num_classes):
        class_total = sum(1 for t in targets if t == class_idx)
        class_val = sum(1 for t in val_targets if t == class_idx)
        assert abs(class_val / class_total - 0.2) < 0.15


def test_train_val_indices_never_overlap(synthetic_dataset_root: Path):
    bundle = build_dataset_bundle(
        synthetic_dataset_root,
        image_size=32,
        validation_fraction=0.2,
        split_seed=1,
        batch_size=4,
    )
    assert set(bundle.train_indices).isdisjoint(bundle.val_indices)
    assert len(bundle.train_indices) + len(bundle.val_indices) == sum(TRAIN_COUNTS.values())


def test_class_to_idx_matches_imagefolder(synthetic_dataset_root: Path):
    bundle = build_dataset_bundle(
        synthetic_dataset_root,
        image_size=32,
        validation_fraction=0.2,
        split_seed=1,
        batch_size=4,
    )
    reference = datasets.ImageFolder(str(synthetic_dataset_root / "Training"))
    assert bundle.class_to_idx == reference.class_to_idx
    assert bundle.classes == tuple(reference.classes)
    assert bundle.num_classes == len(CLASSES)


def test_split_indices_saved_and_reused(synthetic_dataset_root: Path, tmp_path: Path):
    split_path = tmp_path / "split_indices.json"
    bundle_1 = build_dataset_bundle(
        synthetic_dataset_root,
        image_size=32,
        validation_fraction=0.2,
        split_seed=1,
        batch_size=4,
        split_indices_path=split_path,
    )
    assert split_path.exists()

    # Even with a different seed, loading from the saved file must win.
    bundle_2 = build_dataset_bundle(
        synthetic_dataset_root,
        image_size=32,
        validation_fraction=0.2,
        split_seed=999,
        batch_size=4,
        split_indices_path=split_path,
    )
    assert bundle_1.train_indices == bundle_2.train_indices
    assert bundle_1.val_indices == bundle_2.val_indices


@pytest.mark.parametrize("image_size", [32, 64])
def test_transform_output_shape(synthetic_dataset_root: Path, image_size: int):
    img_path = next((synthetic_dataset_root / "Training" / "glioma_tumor").iterdir())
    from PIL import Image

    img = Image.open(img_path)

    eval_tensor = build_eval_transform(image_size)(img)
    assert eval_tensor.shape == torch.Size([3, image_size, image_size])

    train_tensor = build_train_transform(image_size)(img)
    assert train_tensor.shape == torch.Size([3, image_size, image_size])


def test_train_and_eval_transforms_are_distinct_instances(synthetic_dataset_root: Path):
    bundle = build_dataset_bundle(
        synthetic_dataset_root,
        image_size=32,
        validation_fraction=0.2,
        split_seed=1,
        batch_size=4,
    )
    train_transform = bundle.train_loader.dataset.dataset.transform
    val_transform = bundle.val_loader.dataset.dataset.transform
    train_eval_transform = bundle.train_eval_loader.dataset.dataset.transform
    # The (potentially randomized) train transform must never be the same object
    # handed to a validation/eval view.
    assert train_transform is not val_transform
    assert train_transform is not train_eval_transform
    # val_loader and train_eval_loader are both deterministic eval views and may
    # legitimately share one transform instance -- that's not the bug being guarded against.
