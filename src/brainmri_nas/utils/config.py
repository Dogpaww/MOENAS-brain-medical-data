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
    # Depth/width are chromosome-encoded (searched per-candidate), not fixed
    # -- these are the bounds NSGA-II picks within, not single values.
    number_of_cells_min: int = 3
    number_of_cells_max: int = 11
    initial_channels_min: int = 8
    initial_channels_max: int = 48
    drop_path_probability: float = 0.2
    stem_type: str = "cifar"


@dataclasses.dataclass
class ProxyConfig:
    """Zero-cost proxy settings (handoff §17-19)."""

    zico_batch_size: int = 4
    zico_num_batches: int = 4
    proxy_sample_seed: int = 123


@dataclasses.dataclass
class NSGA2Config:
    """NSGA-II search settings (handoff §22, §24, §34)."""

    population_size: int = 20
    num_generations: int = 10
    seed: int = 1
    crossover_prob: float = 0.9
    # None -> 1 / n_var, pymoo's usual default for polynomial mutation.
    mutation_prob: float | None = None
    device: str = "cpu"
    # (log_synflow, zico, flops_billion); log_synflow/zico are benefit criteria, flops is a cost criterion.
    topsis_weights: tuple[float, float, float] = (0.4, 0.4, 0.2)


@dataclasses.dataclass
class AugmentationConfig:
    """Augmentation policy search settings (handoff §25, §34)."""

    population_size: int = 5
    num_generations: int = 5
    trial_epochs: int = 10
    learning_rate: float = 0.025
    weight_decay: float = 3e-4
    seed: int = 1
    mutation_prob: float | None = None
    device: str = "cpu"


@dataclasses.dataclass
class TrainingConfig:
    """Final training settings (handoff §26-29, §34).

    `device` defaults to "auto" here (unlike the search stages' "cpu"
    default) because a real multi-epoch training run should actually use
    available accelerator hardware; `precision="amp"` only actually engages
    autocast/GradScaler when the resolved device is CUDA (handoff §27, §30
    item 17 -- never claim AMP is on while quietly training FP32).

    `learning_rate`/`weight_decay` and the optimizer/scheduler built from
    them (see `utils/optim.py`) match the legacy repo's `evaluate.py::train()`
    exactly (Adam, MultiStepLR at 50%/75% of the epoch budget) -- there's no
    `momentum` field because Adam doesn't take one.
    """

    physical_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    final_epochs: int = 250
    learning_rate: float = 0.025
    weight_decay: float = 3e-4
    grad_clip_norm: float = 5.0
    precision: str = "amp"  # "amp" or "fp32"
    checkpoint_metric: str = "macro_auc"  # key into evaluate_model()'s returned dict
    seed: int = 1
    device: str = "auto"


@dataclasses.dataclass
class Config:
    dataset: DatasetConfig
    search_space: SearchSpaceConfig = dataclasses.field(default_factory=SearchSpaceConfig)
    proxies: ProxyConfig = dataclasses.field(default_factory=ProxyConfig)
    nsga2: NSGA2Config = dataclasses.field(default_factory=NSGA2Config)
    augmentation: AugmentationConfig = dataclasses.field(default_factory=AugmentationConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)


def _dataclass_from_dict(cls, data: dict):
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")

    # YAML has no tuple type -- coerce list -> tuple for any field declared as one,
    # so e.g. NSGA2Config.topsis_weights round-trips as a tuple either way.
    coerced = dict(data)
    for f in dataclasses.fields(cls):
        if f.name in coerced and isinstance(coerced[f.name], list) and isinstance(f.type, str) and f.type.startswith("tuple"):
            coerced[f.name] = tuple(coerced[f.name])

    return cls(**coerced)


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    dataset_raw = raw.get("dataset")
    if dataset_raw is None:
        raise ValueError("Config is missing required 'dataset' section.")
    dataset = _dataclass_from_dict(DatasetConfig, dataset_raw)

    search_space = _dataclass_from_dict(SearchSpaceConfig, raw.get("search_space", {}) or {})
    proxies = _dataclass_from_dict(ProxyConfig, raw.get("proxies", {}) or {})
    nsga2 = _dataclass_from_dict(NSGA2Config, raw.get("nsga2", {}) or {})
    augmentation = _dataclass_from_dict(AugmentationConfig, raw.get("augmentation", {}) or {})
    training = _dataclass_from_dict(TrainingConfig, raw.get("training", {}) or {})

    return Config(
        dataset=dataset,
        search_space=search_space,
        proxies=proxies,
        nsga2=nsga2,
        augmentation=augmentation,
        training=training,
    )


def save_config(config: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "dataset": dataclasses.asdict(config.dataset),
        "search_space": dataclasses.asdict(config.search_space),
        "proxies": dataclasses.asdict(config.proxies),
        "nsga2": dataclasses.asdict(config.nsga2),
        "augmentation": dataclasses.asdict(config.augmentation),
        "training": dataclasses.asdict(config.training),
    }
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
