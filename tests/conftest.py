"""Shared pytest fixtures: a tiny synthetic ImageFolder-style dataset so
Stage 1 tests don't depend on the (not-yet-downloaded) Kaggle dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

CLASSES = ("glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor")
# Deliberately imbalanced per class so the stratified split actually has
# something nontrivial to balance.
TRAIN_COUNTS = {"glioma_tumor": 12, "meningioma_tumor": 9, "no_tumor": 15, "pituitary_tumor": 6}
TEST_COUNTS = {cls: 4 for cls in CLASSES}

_CLASS_COLORS = {
    "glioma_tumor": (200, 50, 50),
    "meningioma_tumor": (50, 200, 50),
    "no_tumor": (50, 50, 200),
    "pituitary_tumor": (200, 200, 50),
}


def _write_fake_images(split_dir: Path, counts: dict[str, int]) -> None:
    for cls, count in counts.items():
        cls_dir = split_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        color = _CLASS_COLORS[cls]
        for i in range(count):
            img = Image.new("RGB", (48, 48), color=color)
            img.save(cls_dir / f"{cls}_{i:03d}.png")


@pytest.fixture()
def synthetic_dataset_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "brain_tumor_mri"
    _write_fake_images(data_root / "Training", TRAIN_COUNTS)
    _write_fake_images(data_root / "Testing", TEST_COUNTS)
    return data_root
