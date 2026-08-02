"""YAML <-> dataclass configuration.

Every run (search, augmentation search, final training) loads one YAML file
into a `Config` and saves that same `Config` back out alongside its outputs,
so a run's exact settings are always reproducible from disk (handoff §33).

Only `dataset` and `search_space` are wired up in Stage 1. Later stages add
sibling sections (proxies, nsga2, augmentation, training) as new dataclass
fields on `Config` -- existing YAML files keep working since new fields carry
defaults.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass
class DatasetConfig:
    data_root: str
    image_size: int = 64
    input_channels: int = 3
    num_classes: int = 4
    validation_fraction: float = 0.15
    split_seed: int = 42
    batch_size: int = 32
    num_workers: int = 4


@dataclasses.dataclass
class SearchSpaceConfig:
    num_intermediate_nodes: int = 4
    edges_per_node: int = 2
    initial_channels: int = 16
    number_of_cells: int = 7
    drop_path_probability: float = 0.2
    stem_type: str = "cifar"


@dataclasses.dataclass
class Config:
    dataset: DatasetConfig
    search_space: SearchSpaceConfig = dataclasses.field(default_factory=SearchSpaceConfig)


def _dataclass_from_dict(cls, data: dict):
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    dataset_raw = raw.get("dataset")
    if dataset_raw is None:
        raise ValueError("Config is missing required 'dataset' section.")
    dataset = _dataclass_from_dict(DatasetConfig, dataset_raw)

    search_space = _dataclass_from_dict(SearchSpaceConfig, raw.get("search_space", {}) or {})

    return Config(dataset=dataset, search_space=search_space)


def save_config(config: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "dataset": dataclasses.asdict(config.dataset),
        "search_space": dataclasses.asdict(config.search_space),
    }
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
