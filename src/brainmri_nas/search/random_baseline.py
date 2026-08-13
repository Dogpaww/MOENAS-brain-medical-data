"""Random-architecture baseline (branch: random_architecture).

`run_random_baseline` replaces NSGA-II's stage 1 with a single uniform draw
from the *same* chromosome space, decoded and scored through the exact same
`evaluate_candidate` path every NSGA-II candidate goes through -- so the
zero-cost proxy scores it writes are directly comparable to the searched
architecture's, even though nothing here is selecting on them.

This exists to answer the question the rest of the pipeline can't answer on
its own: does NSGA-II-over-proxies actually find a better architecture than
chance? Everything downstream of architecture selection (augmentation
search, final training) is untouched -- it only ever reads
`selected_architecture.json`, agnostic to how it was produced, so stages 2
and 3 run completely unmodified against this file.

Writes the same required keys `run_search` does (genotype, number_of_cells,
initial_channels, candidate_hash, chromosome) plus the same proxy fields,
skipping only the NSGA-II/TOPSIS-specific artifacts (Pareto front, TOPSIS
ranking) that don't apply to a single draw.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from brainmri_nas.data.loader import build_dataset_bundle, describe_dataset, load_patient_ids
from brainmri_nas.search.candidate import CandidateCache, evaluate_candidate
from brainmri_nas.search.proxy_samples import build_fixed_proxy_batches
from brainmri_nas.search_space.chromosome import total_chromosome_length
from brainmri_nas.utils.config import Config, save_config
from brainmri_nas.utils.determinism import seed_everything
from brainmri_nas.utils.device import resolve_device
from brainmri_nas.utils.git_info import get_run_manifest
from brainmri_nas.utils.serialization import dump_json

# Matches CHROMOSOME_UPPER_BOUND in nsga2_problem.py / policy_search.py --
# pymoo's real xu is exclusive-in-effect via this same 1 - eps convention,
# kept identical here so a random gene can't land exactly on 1.0 and decode
# out of range the way a gene of precisely 1.0 would.
CHROMOSOME_UPPER_BOUND = 1.0 - 1e-9


def _configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("brainmri_nas.random_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(output_dir / "search.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    return logger


def run_random_baseline(config: Config, output_dir: str | Path, *, seed: int | None = None) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(output_dir)

    # Falls back to config.nsga2.seed rather than a hardcoded default so a
    # bare rerun with no --random-seed is still reproducible from the config
    # alone, same as every other seeded stage in this pipeline.
    resolved_seed = config.nsga2.seed if seed is None else seed
    seed_everything(resolved_seed)
    device = resolve_device(config.nsga2.device)
    logger.info("Starting random-architecture baseline on device=%s (seed=%d)", device, resolved_seed)

    dump_json(describe_dataset(config.dataset.data_root), output_dir / "dataset_report.json")

    # Same split-generation call as run_search, at the same path -- stages 2
    # and 3 require search_dir/split_indices.json to already exist.
    bundle = build_dataset_bundle(
        config.dataset.data_root,
        image_size=config.dataset.image_size,
        validation_fraction=config.dataset.validation_fraction,
        split_seed=config.dataset.split_seed,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        split_indices_path=output_dir / "split_indices.json",
        group_aware_split=config.dataset.group_aware_split,
    )
    dump_json(bundle.class_to_idx, output_dir / "class_mapping.json")
    patient_ids = load_patient_ids(config.dataset.data_root)
    if patient_ids is None:
        split_mode = "per-image (no patient_ids.json -- validation may share patients with training)"
    elif config.dataset.group_aware_split:
        split_mode = f"group-aware over {len(set(patient_ids.values()))} groups"
    else:
        split_mode = "per-image (group_aware_split=False, ablation -- LEAKY on purpose)"
    logger.info(
        "Dataset ready: %d classes, %d train / %d val samples; split=%s",
        bundle.num_classes,
        len(bundle.train_indices),
        len(bundle.val_indices),
        split_mode,
    )

    proxy_batches = build_fixed_proxy_batches(
        bundle,
        num_batches=config.proxies.zico_num_batches,
        batch_size=config.proxies.zico_batch_size,
        seed=config.proxies.proxy_sample_seed,
        proxy_indices_path=output_dir / "proxy_sample_indices.json",
    )

    n_var = total_chromosome_length(config.search_space.num_intermediate_nodes, config.search_space.edges_per_node)
    rng = np.random.default_rng(resolved_seed)
    chromosome = rng.uniform(0.0, CHROMOSOME_UPPER_BOUND, size=n_var)
    logger.info("Drew one uniform-random chromosome (%d genes, seed=%d)", n_var, resolved_seed)

    cache = CandidateCache()
    selected_architecture = evaluate_candidate(
        chromosome,
        cache=cache,
        num_intermediate_nodes=config.search_space.num_intermediate_nodes,
        edges_per_node=config.search_space.edges_per_node,
        input_channels=config.dataset.input_channels,
        num_classes=bundle.num_classes,
        image_size=config.dataset.image_size,
        stem_type=config.search_space.stem_type,
        proxy_batches=proxy_batches,
        device=device,
        number_of_cells_range=(config.search_space.number_of_cells_min, config.search_space.number_of_cells_max),
        initial_channels_range=(config.search_space.initial_channels_min, config.search_space.initial_channels_max),
    )
    if not selected_architecture["valid"]:
        raise RuntimeError(f"Random architecture draw was invalid: {selected_architecture['error']}")

    dump_json(selected_architecture, output_dir / "selected_architecture.json")
    logger.info(
        "Random architecture %s: cells=%d channels=%d log_synflow=%.3f zico=%.3f flops_billion=%.4f",
        selected_architecture["candidate_hash"][:12],
        selected_architecture["number_of_cells"],
        selected_architecture["initial_channels"],
        selected_architecture["log_synflow"],
        selected_architecture["zico"],
        selected_architecture["flops_billion"],
    )

    save_config(config, output_dir / "config.yaml")
    dump_json(get_run_manifest(), output_dir / "run_manifest.json")

    return {"selected_architecture": selected_architecture}
