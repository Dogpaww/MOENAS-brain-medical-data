"""Per-candidate evaluation and caching (handoff §16, §23).

`evaluate_candidate` is the single per-candidate pipeline: decode the
chromosome, repair/validate the genotype, build a fresh model, run SynFlow +
ZiCO + FLOPs/params, record everything as plain (CPU, JSON-safe) data, then
let the model go out of scope. No optimizer, no multi-epoch training.

The candidate cache is keyed by a hash of the *decoded genotype*, not the
raw chromosome -- many different chromosomes bucket (via `floor`) to the
same discrete architecture, so hashing the genotype catches far more
duplicates than hashing the continuous NSGA-II decision vector would.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from brainmri_nas.model.network import build_model
from brainmri_nas.proxies.profiling import profile_flops_and_params
from brainmri_nas.proxies.synflow import compute_synflow
from brainmri_nas.proxies.zico import compute_zico
from brainmri_nas.search_space.chromosome import decode_chromosome, validate_and_repair_genotype
from brainmri_nas.search_space.genotype import NetworkGenotype
from brainmri_nas.utils.serialization import dump_json, load_json

INVALID_LOG_SYNFLOW = -300.0
INVALID_ZICO = -300.0


def compute_candidate_hash(genotype: NetworkGenotype) -> str:
    canonical = json.dumps(genotype.to_dict(), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CandidateCache:
    def __init__(self) -> None:
        self._records: dict[str, dict] = {}

    def __contains__(self, candidate_hash: str) -> bool:
        return candidate_hash in self._records

    def __len__(self) -> int:
        return len(self._records)

    def get(self, candidate_hash: str) -> dict | None:
        return self._records.get(candidate_hash)

    def put(self, candidate_hash: str, record: dict) -> None:
        self._records[candidate_hash] = record

    def all_records(self) -> list[dict]:
        return list(self._records.values())

    def save(self, path: str | Path) -> None:
        dump_json(self.all_records(), path)

    @staticmethod
    def load(path: str | Path) -> "CandidateCache":
        cache = CandidateCache()
        for record in load_json(path):
            cache.put(record["candidate_hash"], record)
        return cache


def _invalid_record(chromosome: Sequence[float], candidate_hash: str, genotype: NetworkGenotype, error: Exception) -> dict:
    return {
        "chromosome": [float(g) for g in chromosome],
        "candidate_hash": candidate_hash,
        "genotype": genotype.to_dict(),
        "synflow": 0.0,
        "log_synflow": INVALID_LOG_SYNFLOW,
        "zico": INVALID_ZICO,
        "flops": float("inf"),
        "flops_billion": float("inf"),
        "params": float("inf"),
        "params_million": float("inf"),
        "valid": False,
        "error": repr(error),
    }


def evaluate_candidate(
    chromosome: Sequence[float],
    *,
    cache: CandidateCache,
    num_intermediate_nodes: int,
    edges_per_node: int,
    input_channels: int,
    num_classes: int,
    image_size: int,
    initial_channels: int,
    number_of_cells: int,
    stem_type: str,
    proxy_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict:
    genotype = decode_chromosome(chromosome, num_intermediate_nodes, edges_per_node)
    genotype = validate_and_repair_genotype(genotype, num_intermediate_nodes, edges_per_node)
    candidate_hash = compute_candidate_hash(genotype)

    cached = cache.get(candidate_hash)
    if cached is not None:
        return cached

    input_shape = (input_channels, image_size, image_size)

    try:
        model = build_model(
            genotype,
            input_channels=input_channels,
            num_classes=num_classes,
            image_size=image_size,
            initial_channels=initial_channels,
            number_of_cells=number_of_cells,
            drop_path_probability=0.0,  # candidates are never trained during search
            stem_type=stem_type,
        )
        model.to(device)

        synflow_result = compute_synflow(model, input_shape=input_shape)
        zico_result = compute_zico(model, batches=proxy_batches, device=device)
        profile_result = profile_flops_and_params(model, input_shape=input_shape, device=device)
        del model

        record = {
            "chromosome": [float(g) for g in chromosome],
            "candidate_hash": candidate_hash,
            "genotype": genotype.to_dict(),
            "synflow": synflow_result["synflow"],
            "log_synflow": synflow_result["log_synflow"],
            "zico": zico_result["zico"],
            "flops": profile_result["flops"],
            "flops_billion": profile_result["flops_billion"],
            "params": profile_result["params"],
            "params_million": profile_result["params_million"],
            "valid": True,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 -- an invalid candidate must be penalized, not crash the search
        record = _invalid_record(chromosome, candidate_hash, genotype, exc)

    cache.put(candidate_hash, record)
    return record
