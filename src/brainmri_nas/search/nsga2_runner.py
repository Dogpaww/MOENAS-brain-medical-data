"""NSGA-II search orchestration (handoff §16, §22-24, §33).

`run_search` is the single entry point: build the dataset bundle and fixed
proxy batches, run NSGA-II with `NASSearchProblem` (zero-cost proxies only,
no candidate is ever trained), then extract the Pareto front and select one
architecture via TOPSIS. Every required search-run output file (handoff §33)
is written to `output_dir`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from brainmri_nas.data.loader import build_dataset_bundle, describe_dataset
from brainmri_nas.search.candidate import CandidateCache
from brainmri_nas.search.nsga2_problem import NASSearchProblem
from brainmri_nas.search.proxy_samples import build_fixed_proxy_batches
from brainmri_nas.search.topsis import get_pareto_front, rank_topsis
from brainmri_nas.search.visualization import save_pareto_front_2d_plot, save_pareto_front_3d_plot
from brainmri_nas.search_space.chromosome import total_chromosome_length
from brainmri_nas.utils.config import Config, save_config
from brainmri_nas.utils.determinism import seed_everything
from brainmri_nas.utils.device import resolve_device
from brainmri_nas.utils.git_info import get_run_manifest
from brainmri_nas.utils.serialization import dump_json


def _configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("brainmri_nas.search")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(output_dir / "search.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    return logger


def run_search(config: Config, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(output_dir)

    seed_everything(config.nsga2.seed)
    device = resolve_device(config.nsga2.device)
    logger.info("Starting NSGA-II search on device=%s", device)

    dump_json(describe_dataset(config.dataset.data_root), output_dir / "dataset_report.json")

    bundle = build_dataset_bundle(
        config.dataset.data_root,
        image_size=config.dataset.image_size,
        validation_fraction=config.dataset.validation_fraction,
        split_seed=config.dataset.split_seed,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        split_indices_path=output_dir / "split_indices.json",
    )
    dump_json(bundle.class_to_idx, output_dir / "class_mapping.json")
    logger.info(
        "Dataset ready: %d classes, %d train / %d val samples",
        bundle.num_classes,
        len(bundle.train_indices),
        len(bundle.val_indices),
    )

    proxy_batches = build_fixed_proxy_batches(
        bundle,
        num_batches=config.proxies.zico_num_batches,
        batch_size=config.proxies.zico_batch_size,
        seed=config.proxies.proxy_sample_seed,
        proxy_indices_path=output_dir / "proxy_sample_indices.json",
    )
    logger.info(
        "Fixed %d proxy batches (%d samples) selected for ZiCO",
        len(proxy_batches),
        config.proxies.zico_num_batches * config.proxies.zico_batch_size,
    )

    cache_path = output_dir / "candidate_cache.json"
    cache = CandidateCache.load(cache_path) if cache_path.exists() else CandidateCache()

    n_var = total_chromosome_length(config.search_space.num_intermediate_nodes, config.search_space.edges_per_node)
    archive: list[dict] = []

    problem = NASSearchProblem(
        n_var=n_var,
        cache=cache,
        archive=archive,
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

    algorithm = NSGA2(
        pop_size=config.nsga2.population_size,
        sampling=FloatRandomSampling(),
        crossover=TwoPointCrossover(prob=config.nsga2.crossover_prob),
        mutation=PolynomialMutation(prob=config.nsga2.mutation_prob or (1.0 / n_var)),
        eliminate_duplicates=True,
    )

    minimize(
        problem,
        algorithm,
        termination=("n_gen", config.nsga2.num_generations),
        seed=config.nsga2.seed,
        save_history=False,
        verbose=True,
    )

    num_valid = sum(1 for r in archive if r.get("valid", True))
    logger.info(
        "Search finished: %d evaluations logged, %d valid, %d unique candidates cached",
        len(archive),
        num_valid,
        len(cache),
    )

    cache.save(cache_path)
    dump_json(archive, output_dir / "search_archive.json")

    pareto_front = get_pareto_front(archive)
    dump_json(pareto_front, output_dir / "pareto_front.json")
    logger.info("Pareto front: %d architectures", len(pareto_front))

    topsis_ranking = rank_topsis(pareto_front, weights=config.nsga2.topsis_weights)
    dump_json(topsis_ranking, output_dir / "topsis_ranking.json")

    selected_architecture = topsis_ranking[0]
    dump_json(selected_architecture, output_dir / "selected_architecture.json")
    logger.info(
        "Selected architecture %s (topsis_score=%.4f)",
        selected_architecture["candidate_hash"][:12],
        selected_architecture["topsis_score"],
    )

    plot_3d_path = output_dir / "pareto_front_3d.png"
    save_pareto_front_3d_plot(archive, selected_architecture, plot_3d_path)
    if plot_3d_path.exists():
        logger.info("Saved 3D Pareto front plot to %s", plot_3d_path)

    plot_2d_path = output_dir / "pareto_front_2d.png"
    save_pareto_front_2d_plot(archive, selected_architecture, plot_2d_path)
    if plot_2d_path.exists():
        logger.info("Saved 2D Pareto front plot to %s", plot_2d_path)

    save_config(config, output_dir / "config.yaml")
    dump_json(get_run_manifest(), output_dir / "run_manifest.json")

    return {
        "selected_architecture": selected_architecture,
        "pareto_front": pareto_front,
        "topsis_ranking": topsis_ranking,
        "archive": archive,
    }
