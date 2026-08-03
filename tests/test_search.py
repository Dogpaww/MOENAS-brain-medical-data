from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from brainmri_nas.model.network import build_model
from brainmri_nas.search.candidate import CandidateCache, evaluate_candidate
from brainmri_nas.search.nsga2_problem import NASSearchProblem
from brainmri_nas.search.nsga2_runner import run_search
from brainmri_nas.search.topsis import get_pareto_front, rank_topsis
from brainmri_nas.search_space.chromosome import total_chromosome_length
from brainmri_nas.utils.config import Config, DatasetConfig, NSGA2Config, ProxyConfig, SearchSpaceConfig
from brainmri_nas.utils.serialization import dump_json, load_json

INPUT_CHANNELS, IMAGE_SIZE, NUM_CLASSES = 3, 16, 4
N_VAR = total_chromosome_length()
# Pinned to a single point (min == max) so these tests get a fixed, tiny,
# fast model regardless of the size genes' actual float value -- depth/width
# are chromosome-decoded now, not passed to evaluate_candidate directly.
FIXED_NUMBER_OF_CELLS_RANGE = (3, 3)
FIXED_INITIAL_CHANNELS_RANGE = (4, 4)


def _fixed_batches(num_batches=2, batch_size=2):
    batches = []
    g = torch.Generator().manual_seed(0)
    for _ in range(num_batches):
        x = torch.randn(batch_size, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, generator=g)
        y = torch.randint(0, NUM_CLASSES, (batch_size,), generator=g)
        batches.append((x, y))
    return batches


def _sample_chromosome(offset=0):
    return [((i * 37 + offset) % 97) / 97.0 for i in range(N_VAR)]


def _candidate_kwargs(**overrides):
    kwargs = dict(
        num_intermediate_nodes=4,
        edges_per_node=2,
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        stem_type="cifar",
        proxy_batches=_fixed_batches(),
        device=torch.device("cpu"),
        number_of_cells_range=FIXED_NUMBER_OF_CELLS_RANGE,
        initial_channels_range=FIXED_INITIAL_CHANNELS_RANGE,
    )
    kwargs.update(overrides)
    return kwargs


def test_candidate_hash_dedupes_chromosomes_decoding_to_same_genotype():
    chromosome = _sample_chromosome()
    # A tiny perturbation that stays within the same floor-bucket for every gene
    # should decode to an identical genotype and therefore hash the same.
    nudged = [min(g + 1e-6, 0.999999) for g in chromosome]

    cache = CandidateCache()
    record_a = evaluate_candidate(chromosome, cache=cache, **_candidate_kwargs())
    record_b = evaluate_candidate(nudged, cache=cache, **_candidate_kwargs())

    assert record_a["candidate_hash"] == record_b["candidate_hash"]
    assert len(cache) == 1


def test_candidate_cache_hit_returns_identical_record_object():
    chromosome = _sample_chromosome()
    cache = CandidateCache()
    kwargs = _candidate_kwargs()

    record_a = evaluate_candidate(chromosome, cache=cache, **kwargs)
    record_b = evaluate_candidate(chromosome, cache=cache, **kwargs)

    assert record_a is record_b  # second call was a cache hit, not a recomputation
    assert len(cache) == 1


def test_evaluate_candidate_produces_finite_valid_record():
    record = evaluate_candidate(_sample_chromosome(), cache=CandidateCache(), **_candidate_kwargs())
    assert record["valid"] is True
    assert record["error"] is None
    assert math.isfinite(record["log_synflow"])
    assert math.isfinite(record["zico"])
    assert math.isfinite(record["flops_billion"])


def test_invalid_candidate_is_penalized_not_crashed():
    # number_of_cells=2 is rejected by build_model's reduction-schedule guard.
    record = evaluate_candidate(
        _sample_chromosome(), cache=CandidateCache(), **_candidate_kwargs(number_of_cells_range=(2, 2))
    )
    assert record["valid"] is False
    assert record["error"] is not None
    assert record["log_synflow"] == -300.0
    assert record["zico"] == -300.0
    assert record["flops_billion"] == float("inf")


def test_objective_directions_in_nsga2_problem():
    cache = CandidateCache()
    archive: list[dict] = []
    problem = NASSearchProblem(
        n_var=N_VAR,
        cache=cache,
        archive=archive,
        num_intermediate_nodes=4,
        edges_per_node=2,
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        stem_type="cifar",
        proxy_batches=_fixed_batches(),
        device=torch.device("cpu"),
        number_of_cells_range=FIXED_NUMBER_OF_CELLS_RANGE,
        initial_channels_range=FIXED_INITIAL_CHANNELS_RANGE,
    )

    import numpy as np

    x = np.array(_sample_chromosome(), dtype=float)
    out: dict = {}
    problem._evaluate(x, out)

    assert len(archive) == 1
    record = archive[0]
    f1, f2, f3 = out["F"]
    assert f1 == pytest.approx(-record["log_synflow"])
    assert f2 == pytest.approx(-record["zico"])
    assert f3 == pytest.approx(record["flops_billion"])


def _make_record(log_synflow, zico, flops_billion, valid=True):
    return {
        "candidate_hash": f"h-{log_synflow}-{zico}-{flops_billion}",
        "log_synflow": log_synflow,
        "zico": zico,
        "flops_billion": flops_billion,
        "valid": valid,
    }


def test_pareto_front_excludes_dominated_records():
    dominant = _make_record(log_synflow=5.0, zico=5.0, flops_billion=1.0)
    dominated = _make_record(log_synflow=4.0, zico=4.0, flops_billion=2.0)  # worse on all 3 axes
    tradeoff = _make_record(log_synflow=6.0, zico=1.0, flops_billion=0.5)  # better synflow+flops, worse zico
    invalid = _make_record(log_synflow=float("nan"), zico=1.0, flops_billion=1.0)

    front = get_pareto_front([dominant, dominated, tradeoff, invalid])
    front_hashes = {r["candidate_hash"] for r in front}

    assert dominant["candidate_hash"] in front_hashes
    assert tradeoff["candidate_hash"] in front_hashes
    assert dominated["candidate_hash"] not in front_hashes
    assert invalid["candidate_hash"] not in front_hashes


def test_archive_and_pareto_front_json_round_trip(tmp_path: Path):
    records = [
        _make_record(log_synflow=5.0, zico=5.0, flops_billion=1.0),
        _make_record(log_synflow=4.0, zico=4.0, flops_billion=2.0),
    ]
    path = tmp_path / "search_archive.json"
    dump_json(records, path)
    reloaded = load_json(path)
    assert reloaded == records


def test_topsis_ranks_dominant_architecture_first():
    best = _make_record(log_synflow=10.0, zico=10.0, flops_billion=0.1)
    middle = _make_record(log_synflow=5.0, zico=5.0, flops_billion=0.5)
    worst = _make_record(log_synflow=1.0, zico=1.0, flops_billion=5.0)

    ranked = rank_topsis([worst, best, middle], weights=(0.4, 0.4, 0.2))

    assert [r["candidate_hash"] for r in ranked] == [
        best["candidate_hash"],
        middle["candidate_hash"],
        worst["candidate_hash"],
    ]
    scores = [r["topsis_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_save_pareto_front_3d_plot_writes_a_readable_image(tmp_path: Path):
    from PIL import Image

    from brainmri_nas.search.visualization import save_pareto_front_3d_plot

    records = [
        _make_record(log_synflow=10.0, zico=10.0, flops_billion=0.1),
        _make_record(log_synflow=5.0, zico=5.0, flops_billion=0.5),
        _make_record(log_synflow=1.0, zico=1.0, flops_billion=5.0),
        _make_record(log_synflow=8.0, zico=2.0, flops_billion=0.2),
    ]
    selected = records[0]
    path = tmp_path / "pareto_front_3d.png"

    save_pareto_front_3d_plot(records, selected, path)

    assert path.exists()
    img = Image.open(path)
    img.verify()
    assert path.stat().st_size > 0


def test_save_pareto_front_3d_plot_skips_cleanly_with_no_valid_records(tmp_path: Path):
    from brainmri_nas.search.visualization import save_pareto_front_3d_plot

    invalid = _make_record(log_synflow=float("nan"), zico=1.0, flops_billion=1.0)
    path = tmp_path / "pareto_front_3d.png"

    save_pareto_front_3d_plot([invalid], invalid, path)  # must not raise
    assert not path.exists()


def test_save_pareto_front_2d_plot_writes_a_readable_image(tmp_path: Path):
    from PIL import Image

    from brainmri_nas.search.visualization import save_pareto_front_2d_plot

    records = [
        _make_record(log_synflow=10.0, zico=10.0, flops_billion=0.1),
        _make_record(log_synflow=5.0, zico=5.0, flops_billion=0.5),
        _make_record(log_synflow=1.0, zico=1.0, flops_billion=5.0),
        _make_record(log_synflow=8.0, zico=2.0, flops_billion=0.2),
    ]
    path = tmp_path / "pareto_front_2d.png"

    save_pareto_front_2d_plot(records, records[0], path)

    assert path.exists()
    img = Image.open(path)
    img.verify()


def test_pareto_staircase_is_monotonic_non_increasing():
    import numpy as np

    from brainmri_nas.search.visualization import _pareto_staircase

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, size=30)
    y = rng.uniform(0, 10, size=30)

    step_x, step_y = _pareto_staircase(x, y)

    assert np.all(np.diff(step_x) > 0)  # strictly increasing x
    assert np.all(np.diff(step_y) < 0)  # strictly decreasing y -- the actual "downward curve" guarantee


def test_end_to_end_tiny_nsga2_search(synthetic_dataset_root: Path, tmp_path: Path):
    config = Config(
        dataset=DatasetConfig(
            data_root=str(synthetic_dataset_root),
            image_size=16,
            batch_size=4,
            num_workers=0,
        ),
        search_space=SearchSpaceConfig(
            initial_channels_min=4, initial_channels_max=4, number_of_cells_min=3, number_of_cells_max=3
        ),
        proxies=ProxyConfig(zico_batch_size=2, zico_num_batches=2),
        nsga2=NSGA2Config(population_size=4, num_generations=2, seed=1, device="cpu"),
    )

    output_dir = tmp_path / "search_run"
    result = run_search(config, output_dir)

    for filename in (
        "config.yaml",
        "dataset_report.json",
        "class_mapping.json",
        "split_indices.json",
        "proxy_sample_indices.json",
        "candidate_cache.json",
        "search_archive.json",
        "pareto_front.json",
        "topsis_ranking.json",
        "selected_architecture.json",
        "search.log",
        "run_manifest.json",
        "pareto_front_3d.png",
        "pareto_front_2d.png",
    ):
        assert (output_dir / filename).exists(), f"missing output file: {filename}"

    selected = result["selected_architecture"]
    assert selected["valid"] is True
    assert len(result["pareto_front"]) >= 1
    assert len(result["archive"]) >= config.nsga2.population_size

    # The saved genotype must be independently rebuildable into a working model.
    from brainmri_nas.search_space.genotype import NetworkGenotype

    genotype = NetworkGenotype.from_dict(selected["genotype"])
    model = build_model(
        genotype,
        input_channels=config.dataset.input_channels,
        num_classes=config.dataset.num_classes,
        image_size=config.dataset.image_size,
        initial_channels=selected["initial_channels"],
        number_of_cells=selected["number_of_cells"],
        drop_path_probability=0.0,
        stem_type=config.search_space.stem_type,
    )
    logits = model(torch.randn(2, config.dataset.input_channels, config.dataset.image_size, config.dataset.image_size))
    assert logits.shape == (2, config.dataset.num_classes)
