"""Legacy-style augmentation search orchestration (branch: fixed_da).

Structurally identical to `policy_search.py` -- same equal-conditions
protocol (one architecture built once, its initial weights snapshotted and
reloaded fresh for every trial, so a trial's score reflects the policy under
test and nothing else), same pymoo single-objective `GA`, same fitness
(validation macro AUC via `train_trial_model`). The only thing that changes
is what a "policy" is and how it is turned into a transform: a fixed pick of
2 ops from the menu (`legacy_search_space.py`), applied to every image with
no per-sample adaptivity, rather than 7 sample-adaptive SapAugment curves.

Because there is no `LossCache` here at all, `train_loader` yields plain
`(x, y)` batches -- `train_trial_model(..., loss_cache=None)` already
supports exactly this (see its docstring).
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from brainmri_nas.augment.legacy_search_space import chromosome_length, decode_legacy_chromosome
from brainmri_nas.augment.legacy_transform_builder import build_legacy_transform
from brainmri_nas.augment.trial_training import train_trial_model
from brainmri_nas.data.loader import build_dataset_bundle, is_real_image_file
from brainmri_nas.model.network import build_model
from brainmri_nas.search_space.genotype import NetworkGenotype
from brainmri_nas.utils.config import Config, save_config
from brainmri_nas.utils.determinism import seed_everything
from brainmri_nas.utils.device import resolve_device
from brainmri_nas.utils.git_info import get_run_manifest
from brainmri_nas.utils.serialization import dump_json, load_json

CHROMOSOME_UPPER_BOUND = 1.0 - 1e-9


def _configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("brainmri_nas.legacy_augmentation_search")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(output_dir / "legacy_augmentation_search.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    return logger


def _build_legacy_train_loader(
    train_dir: Path,
    train_indices: tuple[int, ...],
    *,
    op1: str,
    op2: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    transform = build_legacy_transform(op1, op2, image_size)
    dataset = datasets.ImageFolder(str(train_dir), transform=transform, is_valid_file=is_real_image_file)
    return DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers)


class LegacyPolicySearchProblem(ElementwiseProblem):
    def __init__(
        self,
        *,
        n_var: int,
        archive: list[dict],
        genotype: NetworkGenotype,
        model_build_kwargs: dict,
        initial_state: dict[str, torch.Tensor],
        train_dir: str | Path,
        train_indices: tuple[int, ...],
        val_loader,
        image_size: int,
        batch_size: int,
        num_workers: int,
        trial_epochs: int,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        num_classes: int,
        device: torch.device,
        total_trials: int,
        logger: logging.Logger,
    ):
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=0, xl=np.zeros(n_var), xu=np.full(n_var, CHROMOSOME_UPPER_BOUND))
        self.archive = archive
        self.genotype = genotype
        self.model_build_kwargs = model_build_kwargs
        self.initial_state = initial_state
        self.train_dir = train_dir
        self.train_indices = train_indices
        self.val_loader = val_loader
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trial_epochs = trial_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.num_classes = num_classes
        self.device = device
        self.total_trials = total_trials
        self.logger = logger
        self.trial_count = 0

    def _evaluate(self, x, out, *args, **kwargs):
        self.trial_count += 1
        trial_label = f"Trial {self.trial_count}/{self.total_trials}"

        op1, op2 = decode_legacy_chromosome(x)
        self.logger.info("%s: starting (ops=%s+%s, %d trial epochs)", trial_label, op1, op2, self.trial_epochs)

        train_loader = _build_legacy_train_loader(
            self.train_dir,
            self.train_indices,
            op1=op1,
            op2=op2,
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        trial_model = build_model(self.genotype, **self.model_build_kwargs)
        trial_model.load_state_dict(self.initial_state)

        result = train_trial_model(
            trial_model,
            train_loader,
            self.val_loader,
            epochs=self.trial_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            device=self.device,
            num_classes=self.num_classes,
            loss_cache=None,  # no per-sample adaptivity in the legacy mechanism
            logger=self.logger,
        )

        del trial_model, train_loader
        gc.collect()

        best_so_far = max((r["val_macro_auc"] for r in self.archive), default=result["val_macro_auc"])
        best_so_far = max(best_so_far, result["val_macro_auc"])
        self.logger.info(
            "%s: finished val_macro_auc=%.4f (best so far: %.4f)",
            trial_label,
            result["val_macro_auc"],
            best_so_far,
        )

        record = {
            "chromosome": [float(v) for v in x],
            "ops": [op1, op2],
            "val_macro_auc": result["val_macro_auc"],
        }
        self.archive.append(record)

        out["F"] = np.array([-result["val_macro_auc"]], dtype=float)


def run_legacy_augmentation_search(
    config: Config,
    *,
    selected_architecture_path: str | Path,
    split_indices_path: str | Path,
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(output_dir)

    seed_everything(config.augmentation.seed)
    device = resolve_device(config.augmentation.device)
    logger.info("Starting legacy (fixed 2-op) augmentation search on device=%s", device)

    selected_architecture = load_json(selected_architecture_path)
    genotype = NetworkGenotype.from_dict(selected_architecture["genotype"])

    bundle = build_dataset_bundle(
        config.dataset.data_root,
        image_size=config.dataset.image_size,
        validation_fraction=config.dataset.validation_fraction,
        split_seed=config.dataset.split_seed,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        split_indices_path=split_indices_path,  # must already exist -- reuses the architecture-search split
        group_aware_split=config.dataset.group_aware_split,
    )
    logger.info(
        "Reused split from %s: %d train / %d val samples", split_indices_path, len(bundle.train_indices), len(bundle.val_indices)
    )

    model_build_kwargs = dict(
        input_channels=config.dataset.input_channels,
        num_classes=bundle.num_classes,
        image_size=config.dataset.image_size,
        initial_channels=selected_architecture["initial_channels"],
        number_of_cells=selected_architecture["number_of_cells"],
        drop_path_probability=config.search_space.drop_path_probability,
        stem_type=config.search_space.stem_type,
    )

    base_model = build_model(genotype, **model_build_kwargs)
    initial_state = {name: value.detach().cpu().clone() for name, value in base_model.state_dict().items()}
    del base_model
    gc.collect()
    logger.info("Captured shared initial weights for policy trials")

    train_dir = Path(config.dataset.data_root) / "Training"
    n_var = chromosome_length()
    archive: list[dict] = []

    problem = LegacyPolicySearchProblem(
        n_var=n_var,
        archive=archive,
        genotype=genotype,
        model_build_kwargs=model_build_kwargs,
        initial_state=initial_state,
        train_dir=train_dir,
        train_indices=bundle.train_indices,
        val_loader=bundle.val_loader,
        image_size=config.dataset.image_size,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        trial_epochs=config.augmentation.trial_epochs,
        learning_rate=config.augmentation.learning_rate,
        weight_decay=config.augmentation.weight_decay,
        momentum=config.augmentation.momentum,
        num_classes=bundle.num_classes,
        device=device,
        total_trials=config.augmentation.population_size * config.augmentation.num_generations,
        logger=logger,
    )

    algorithm = GA(
        pop_size=config.augmentation.population_size,
        sampling=FloatRandomSampling(),
        crossover=TwoPointCrossover(prob=0.9),
        mutation=PolynomialMutation(prob=config.augmentation.mutation_prob or (1.0 / n_var)),
        eliminate_duplicates=True,
    )

    minimize(
        problem,
        algorithm,
        termination=("n_gen", config.augmentation.num_generations),
        seed=config.augmentation.seed,
        save_history=False,
        verbose=True,
    )

    logger.info("Legacy augmentation search finished: %d trials evaluated", len(archive))

    dump_json(archive, output_dir / "legacy_policy_archive.json")

    selected_policy = max(archive, key=lambda r: r["val_macro_auc"])
    dump_json(selected_policy, output_dir / "selected_legacy_policy.json")
    logger.info("Selected ops %s val_macro_auc=%.4f", selected_policy["ops"], selected_policy["val_macro_auc"])

    save_config(config, output_dir / "augmentation_config.yaml")
    dump_json(get_run_manifest(), output_dir / "run_manifest.json")

    return {"selected_policy": selected_policy, "archive": archive}
