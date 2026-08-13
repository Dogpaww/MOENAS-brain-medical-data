from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from torchvision import datasets

from PIL import Image

from brainmri_nas.data.loader import DatasetValidationError, build_dataset_bundle, describe_dataset
from brainmri_nas.data.split import grouped_stratified_split, save_split_indices, stratified_split
from brainmri_nas.data.transforms import (
    PerImageNormalize,
    ResizeLongerSideAndPad,
    build_eval_transform,
    build_train_transform,
)
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


def _write_patient_ids(root: Path, images_per_group: int = 3) -> dict[str, str]:
    """Assign consecutive images within each Training/ class to shared groups,
    imitating multiple slices per patient."""
    mapping = {}
    for class_dir in sorted((root / "Training").iterdir()):
        if not class_dir.is_dir():
            continue
        for i, img in enumerate(sorted(class_dir.iterdir())):
            key = img.relative_to(root).as_posix()
            mapping[key] = f"{class_dir.name}_patient_{i // images_per_group}"
    (root / "patient_ids.json").write_text(json.dumps(mapping))
    return mapping


def test_macos_appledouble_sidecars_are_not_loaded_as_images(synthetic_dataset_root: Path):
    """A dataset tarred on macOS and extracted on Linux gains a `._name.png`
    sidecar next to every file. Those are metadata, and loading them as images
    both doubles the dataset and breaks index alignment against patient_ids."""
    for class_dir in (synthetic_dataset_root / "Training").iterdir():
        if class_dir.is_dir():
            for img in list(class_dir.glob("*.png")):
                (class_dir / f"._{img.name}").write_bytes(b"\x00\x05\x16\x07not an image")

    bundle = build_dataset_bundle(
        synthetic_dataset_root, image_size=32, validation_fraction=0.2, split_seed=1, batch_size=4
    )
    loaded = [Path(p).name for p, _ in bundle.train_loader.dataset.dataset.samples]
    assert loaded, "no images loaded at all"
    assert not any(n.startswith("._") for n in loaded)
    assert len(bundle.train_indices) + len(bundle.val_indices) == sum(TRAIN_COUNTS.values())


def test_appledouble_sidecars_do_not_break_group_aware_splitting(synthetic_dataset_root: Path):
    """The real failure: patient_ids.json covers real files only, so sidecars
    loaded as images have no group id and the completeness check aborts."""
    _write_patient_ids(synthetic_dataset_root)
    for class_dir in (synthetic_dataset_root / "Training").iterdir():
        if class_dir.is_dir():
            for img in list(class_dir.glob("[!.]*.png")):
                (class_dir / f"._{img.name}").write_bytes(b"\x00\x05\x16\x07")

    bundle = build_dataset_bundle(
        synthetic_dataset_root, image_size=32, validation_fraction=0.25, split_seed=1, batch_size=4
    )
    assert len(bundle.train_indices) + len(bundle.val_indices) == sum(TRAIN_COUNTS.values())


def test_grouped_split_never_puts_one_group_on_both_sides():
    targets, groups = [], []
    for cls in range(3):
        for g in range(20):
            n = 1 + (g % 5)  # uneven group sizes, like real per-patient slice counts
            targets += [cls] * n
            groups += [f"c{cls}_g{g}"] * n

    train_idx, val_idx = grouped_stratified_split(targets, groups, validation_fraction=0.25, split_seed=0)

    assert set(train_idx).isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == len(targets)
    train_groups = {groups[i] for i in train_idx}
    val_groups = {groups[i] for i in val_idx}
    assert train_groups.isdisjoint(val_groups)  # the whole point
    assert set(targets[i] for i in val_idx) == {0, 1, 2}  # still stratified


def test_grouped_split_is_deterministic_for_a_given_seed():
    targets = [i % 2 for i in range(60)]
    groups = [f"g{i // 3}" for i in range(60)]
    a = grouped_stratified_split(targets, groups, 0.25, split_seed=7)
    b = grouped_stratified_split(targets, groups, 0.25, split_seed=7)
    assert a == b
    assert grouped_stratified_split(targets, groups, 0.25, split_seed=8) != a


def test_grouped_split_rejects_too_few_groups():
    # 3 groups cannot yield a 20% (1-in-5) grouped split.
    with pytest.raises(ValueError, match="distinct groups"):
        grouped_stratified_split([0] * 9, ["a", "a", "a", "b", "b", "b", "c", "c", "c"], 0.20, 0)


def test_bundle_uses_group_aware_split_when_patient_ids_present(synthetic_dataset_root: Path):
    mapping = _write_patient_ids(synthetic_dataset_root)
    bundle = build_dataset_bundle(
        synthetic_dataset_root, image_size=32, validation_fraction=0.25, split_seed=1, batch_size=4
    )
    samples = bundle.train_loader.dataset.dataset.samples
    keys = [mapping[Path(p).relative_to(synthetic_dataset_root).as_posix()] for p, _ in samples]
    train_groups = {keys[i] for i in bundle.train_indices}
    val_groups = {keys[i] for i in bundle.val_indices}
    assert train_groups and val_groups
    assert train_groups.isdisjoint(val_groups)


def test_group_aware_split_can_be_disabled_for_the_leaky_ablation(synthetic_dataset_root: Path):
    """Turning the flag off must reproduce the plain per-image split, which is
    how the 'how much of validation is memorisation' comparison is run."""
    _write_patient_ids(synthetic_dataset_root)
    bundle = build_dataset_bundle(
        synthetic_dataset_root,
        image_size=32,
        validation_fraction=0.25,
        split_seed=1,
        batch_size=4,
        group_aware_split=False,
    )
    targets = [label for _, label in bundle.train_loader.dataset.dataset.samples]
    expected_train, expected_val = stratified_split(targets, 0.25, 1)
    assert bundle.train_indices == expected_train
    assert bundle.val_indices == expected_val


def test_incomplete_patient_ids_file_raises_rather_than_leaking(synthetic_dataset_root: Path):
    mapping = _write_patient_ids(synthetic_dataset_root)
    del mapping[next(iter(mapping))]  # drop one image
    (synthetic_dataset_root / "patient_ids.json").write_text(json.dumps(mapping))
    with pytest.raises(DatasetValidationError, match="missing from patient_ids.json"):
        build_dataset_bundle(
            synthetic_dataset_root, image_size=32, validation_fraction=0.25, split_seed=1, batch_size=4
        )


def test_saved_split_that_violates_groups_is_rejected(synthetic_dataset_root: Path, tmp_path: Path):
    """A split file written before group-aware splitting existed must not be
    silently reused -- that would train on a leaky split without any signal."""
    _write_patient_ids(synthetic_dataset_root)
    split_path = tmp_path / "split_indices.json"
    targets = [label for _, label in datasets.ImageFolder(str(synthetic_dataset_root / "Training")).samples]
    leaky_train, leaky_val = stratified_split(targets, 0.25, 1)
    save_split_indices(leaky_train, leaky_val, split_path)

    with pytest.raises(DatasetValidationError, match="BOTH train and validation"):
        build_dataset_bundle(
            synthetic_dataset_root,
            image_size=32,
            validation_fraction=0.25,
            split_seed=1,
            batch_size=4,
            split_indices_path=split_path,
        )


def test_resize_longer_side_and_pad_produces_exact_target_size():
    transform = ResizeLongerSideAndPad(size=64)
    for original_size in [(200, 100), (100, 200), (50, 50), (174, 1375)]:
        img = Image.new("RGB", original_size, color=(10, 20, 30))
        out = transform(img)
        assert out.size == (64, 64)


def test_resize_longer_side_and_pad_preserves_aspect_ratio():
    # 200x100 (2:1) -> longer side (width) becomes 64, height scales to 32,
    # then 16px of padding is added above and below to reach 64x64.
    transform = ResizeLongerSideAndPad(size=64, fill=0)
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    out = transform(img)

    top_row = out.crop((0, 0, 64, 1)).getpixel((0, 0))
    middle_row = out.crop((0, 32, 64, 33)).getpixel((0, 0))
    assert top_row == (0, 0, 0)  # padding: black
    assert middle_row == (255, 255, 255)  # real (resized) content: white


def test_resize_longer_side_and_pad_no_padding_needed_for_square_input():
    transform = ResizeLongerSideAndPad(size=64)
    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    out = transform(img)
    assert out.size == (64, 64)
    # No black border should appear anywhere for an already-square input.
    corners = [out.getpixel(p) for p in [(0, 0), (63, 0), (0, 63), (63, 63)]]
    assert all(c != (0, 0, 0) for c in corners)


def test_per_image_normalize_output_has_zero_mean_unit_std():
    normalize = PerImageNormalize()
    tensor = torch.rand(3, 16, 16)
    out = normalize(tensor)
    assert out.mean().abs() < 1e-5
    assert abs(out.std().item() - 1.0) < 1e-3


def test_per_image_normalize_removes_uniform_brightness_shift():
    # Same underlying "content" (relative pattern), two different absolute
    # brightness levels -- exactly the train/test gap measured in the real
    # dataset. Per-image normalization must make these identical.
    torch.manual_seed(0)
    dark = torch.rand(1, 8, 8) * 0.3 + 0.1  # roughly in [0.1, 0.4]
    bright = dark + 0.3  # same pattern, uniformly brighter

    normalize = PerImageNormalize()
    assert torch.allclose(normalize(dark), normalize(bright), atol=1e-5)


def test_per_image_normalize_handles_constant_image_without_nan():
    normalize = PerImageNormalize()
    constant = torch.full((3, 8, 8), 0.5)
    out = normalize(constant)
    assert torch.isfinite(out).all()
