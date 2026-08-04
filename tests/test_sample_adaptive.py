"""Integration tests for the sample-adaptive augmentation wiring itself --
not just that the individual pieces (LossCache, resolve_step) work in
isolation, but that the dataset/loader/training-loop plumbing connecting
them actually behaves as intended.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from brainmri_nas.augment.sample_adaptive_dataset import SampleAdaptiveDataset, build_sample_adaptive_loader
from brainmri_nas.augment.search_space import chromosome_length, decode_chromosome
from brainmri_nas.augment.trial_training import train_trial_model
from brainmri_nas.data.loader import build_dataset_bundle
from brainmri_nas.model.network import build_model
from brainmri_nas.search_space.chromosome import chromosome_length as arch_chromosome_length
from brainmri_nas.search_space.chromosome import decode_chromosome as decode_arch_chromosome
from brainmri_nas.training.engine import train_one_epoch
from brainmri_nas.utils.loss_cache import LossCache


def _policy():
    length = chromosome_length()
    chromosome = [((i * 53) % 101) / 101.0 for i in range(length)]
    return decode_chromosome(chromosome)


def test_dataset_getitem_uses_that_specific_samples_cached_rank(monkeypatch, synthetic_dataset_root: Path):
    calls: list[float] = []

    def fake_build_transform(policy, image_size, loss_rank, class_scale=1.0):
        calls.append(loss_rank)
        return lambda img: torch.zeros(3, image_size, image_size)

    monkeypatch.setattr(
        "brainmri_nas.augment.sample_adaptive_dataset.build_sample_adaptive_transform", fake_build_transform
    )

    from torchvision import datasets

    from torch.utils.data import Subset

    raw = datasets.ImageFolder(str(synthetic_dataset_root / "Training"), transform=None)
    indices = tuple(range(len(raw)))
    subset = Subset(raw, indices)

    cache = LossCache(num_samples=len(indices), total_epochs=1)  # interval=1
    dataset = SampleAdaptiveDataset(subset, image_size=16, policy=_policy(), loss_cache=cache)

    # Cold start: every read should see the neutral rank.
    dataset[0]
    assert calls[-1] == 0.5

    # After a refresh where sample 0 is clearly the hardest, its read must
    # reflect that -- not some other sample's rank, not a stale value.
    cache.record_batch(list(indices), [0.0] * (len(indices) - 1) + [100.0])
    cache.end_epoch()
    dataset[len(indices) - 1]
    assert calls[-1] == 1.0  # hardest sample -> rank 1.0


def test_sample_adaptive_loader_yields_index_triples(synthetic_dataset_root: Path):
    bundle = build_dataset_bundle(
        synthetic_dataset_root, image_size=16, validation_fraction=0.2, split_seed=1, batch_size=4
    )
    cache = LossCache(num_samples=len(bundle.train_indices), total_epochs=1)
    loader = build_sample_adaptive_loader(
        synthetic_dataset_root / "Training",
        bundle.train_indices,
        image_size=16,
        batch_size=4,
        policy=_policy(),
        loss_cache=cache,
        num_workers=0,
    )

    x, y, indices = next(iter(loader))
    assert x.shape[1:] == (3, 16, 16)
    assert indices.shape == y.shape
    assert torch.all(indices >= 0) and torch.all(indices < len(bundle.train_indices))


def _tiny_model():
    genotype = decode_arch_chromosome(
        [((i * 37) % 97) / 97.0 for i in range(arch_chromosome_length())]
    )
    return build_model(
        genotype,
        input_channels=3,
        num_classes=4,
        image_size=16,
        initial_channels=4,
        number_of_cells=3,
        drop_path_probability=0.0,
        stem_type="cifar",
    )


def test_train_one_epoch_populates_the_loss_cache(synthetic_dataset_root: Path):
    bundle = build_dataset_bundle(
        synthetic_dataset_root, image_size=16, validation_fraction=0.2, split_seed=1, batch_size=4
    )
    cache = LossCache(num_samples=len(bundle.train_indices), total_epochs=1)  # interval=1, refreshes every epoch
    loader = build_sample_adaptive_loader(
        synthetic_dataset_root / "Training",
        bundle.train_indices,
        image_size=16,
        batch_size=4,
        policy=_policy(),
        loss_cache=cache,
        num_workers=0,
    )

    model = _tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    assert cache.has_data is False
    train_one_epoch(
        model,
        loader,
        optimizer,
        device=torch.device("cpu"),
        use_amp=False,
        scaler=None,
        accumulation_steps=1,
        grad_clip_norm=5.0,
        loss_cache=cache,
    )
    assert cache.has_data is True
    ranks = cache.get_loss_ranks()
    assert ranks.shape == (len(bundle.train_indices),)
    assert not (ranks == 0.5).all()  # a real spread, not everyone still stuck at neutral


def test_train_trial_model_populates_the_loss_cache_via_full_wiring(synthetic_dataset_root: Path):
    bundle = build_dataset_bundle(
        synthetic_dataset_root, image_size=16, validation_fraction=0.2, split_seed=1, batch_size=4
    )
    cache = LossCache(num_samples=len(bundle.train_indices), total_epochs=1)
    train_loader = build_sample_adaptive_loader(
        synthetic_dataset_root / "Training",
        bundle.train_indices,
        image_size=16,
        batch_size=4,
        policy=_policy(),
        loss_cache=cache,
        num_workers=0,
    )

    model = _tiny_model()
    train_trial_model(
        model,
        train_loader,
        bundle.val_loader,
        epochs=1,
        learning_rate=0.025,
        weight_decay=3e-4,
        device=torch.device("cpu"),
        num_classes=4,
        loss_cache=cache,
    )

    assert cache.has_data is True
