"""Common dataset bundle (handoff §4)."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass(frozen=True)
class DatasetBundle:
    train_loader: DataLoader
    train_eval_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    classes: tuple[str, ...]
    class_to_idx: dict[str, int]
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    input_channels: int
    num_classes: int
