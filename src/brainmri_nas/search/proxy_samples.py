"""Fixed proxy sample selection (handoff §19).

ZiCO needs real data, and every candidate in a search run must see the
*exact same* handful of training samples -- otherwise score differences
could come from which images a candidate happened to see rather than the
architecture itself. This module selects that fixed sample set once (from
the training split only, using the deterministic eval transform, never the
randomly-augmenting train transform), saves it, and reuses it thereafter.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from brainmri_nas.data.bundle import DatasetBundle
from brainmri_nas.utils.serialization import dump_json, load_json


def select_fixed_local_indices(train_split_size: int, num_samples: int, seed: int) -> tuple[int, ...]:
    if num_samples > train_split_size:
        raise ValueError(
            f"Requested {num_samples} proxy samples but the training split only has {train_split_size}."
        )
    rng = random.Random(seed)
    pool = list(range(train_split_size))
    rng.shuffle(pool)
    return tuple(sorted(pool[:num_samples]))


def save_proxy_sample_indices(bundle: DatasetBundle, local_indices: tuple[int, ...], path: str | Path) -> None:
    absolute_indices = [bundle.train_indices[i] for i in local_indices]
    dump_json(
        {"local_indices": list(local_indices), "absolute_dataset_indices": absolute_indices},
        path,
    )


def load_proxy_sample_indices(path: str | Path) -> tuple[int, ...]:
    return tuple(load_json(path)["local_indices"])


def build_fixed_proxy_batches(
    bundle: DatasetBundle,
    *,
    num_batches: int,
    batch_size: int,
    seed: int,
    proxy_indices_path: str | Path | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    # `train_eval_loader`'s dataset is a Subset(ImageFolder, train_indices) built
    # with the deterministic eval transform -- exactly what fixed proxy samples need.
    train_eval_dataset = bundle.train_eval_loader.dataset
    num_samples = num_batches * batch_size

    if proxy_indices_path is not None and Path(proxy_indices_path).exists():
        local_indices = load_proxy_sample_indices(proxy_indices_path)
        if len(local_indices) != num_samples:
            raise ValueError(
                f"Saved proxy sample indices at {proxy_indices_path} have {len(local_indices)} entries, "
                f"but num_batches * batch_size = {num_samples}."
            )
    else:
        local_indices = select_fixed_local_indices(len(train_eval_dataset), num_samples, seed)
        if proxy_indices_path is not None:
            save_proxy_sample_indices(bundle, local_indices, proxy_indices_path)

    fixed_subset = Subset(train_eval_dataset, local_indices)
    loader = DataLoader(fixed_subset, batch_size=batch_size, shuffle=False)
    return [(x, y) for x, y in loader]
