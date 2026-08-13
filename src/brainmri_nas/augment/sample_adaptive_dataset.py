"""Sample-adaptive training dataset (SapAugment, Hu et al. 2021).

Wraps a raw (untransformed) `Subset(ImageFolder)` so each read builds its
own transform on the fly from that sample's current cached loss rank,
rather than sharing one fixed transform across every sample -- the whole
point of sample-adaptive augmentation is that the same image gets a
different transform depending on how the model is currently doing on it.

`__getitem__` also returns the sample's local index (its position within
`train_indices`, matching how `LossCache` is sized and indexed), which the
training loop needs to record that sample's loss back into the same cache
after the forward pass. This is why the underlying `ImageFolder` must be
built with `transform=None` -- the transform can't be baked into the
dataset ahead of time here, unlike the loaders elsewhere in this project.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from brainmri_nas.augment.genotype import AugmentationPolicy
from brainmri_nas.augment.transform_builder import build_sample_adaptive_transform
from brainmri_nas.data.loader import is_real_image_file
from brainmri_nas.utils.loss_cache import LossCache


class SampleAdaptiveDataset(Dataset):
    def __init__(
        self,
        base_dataset: Subset,
        *,
        image_size: int,
        policy: AugmentationPolicy,
        loss_cache: LossCache,
    ):
        self.base_dataset = base_dataset
        self.image_size = image_size
        self.policy = policy
        self.loss_cache = loss_cache

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        img, label = self.base_dataset[index]
        loss_rank = float(self.loss_cache.get_loss_ranks()[index])
        transform = build_sample_adaptive_transform(self.policy, self.image_size, loss_rank)
        return transform(img), label, index


def build_sample_adaptive_loader(
    train_dir: str | Path,
    train_indices: tuple[int, ...],
    *,
    image_size: int,
    batch_size: int,
    policy: AugmentationPolicy,
    loss_cache: LossCache,
    num_workers: int = 0,
) -> DataLoader:
    raw_dataset = datasets.ImageFolder(str(train_dir), transform=None, is_valid_file=is_real_image_file)
    subset = Subset(raw_dataset, train_indices)
    adaptive_dataset = SampleAdaptiveDataset(subset, image_size=image_size, policy=policy, loss_cache=loss_cache)
    return DataLoader(adaptive_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
