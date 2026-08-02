"""Pareto-front extraction and TOPSIS ranking (handoff §24).

`get_pareto_front` sorts on the same internal minimization objectives NSGA-II
used (`-log_synflow`, `-zico`, `flops_billion`). `rank_topsis` then ranks
architectures on the *natural-direction* criteria named in `criteria`/`benefit`
(log_synflow and zico as benefit criteria -- higher is better; flops_billion
as a cost criterion -- lower is better), which is why the sign convention
differs between the two functions: NSGA-II's objective space and TOPSIS's
criteria space are related but not identical.
"""

from __future__ import annotations

import math

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

DEFAULT_CRITERIA = ("log_synflow", "zico", "flops_billion")
DEFAULT_BENEFIT = (True, True, False)  # log_synflow: benefit, zico: benefit, flops_billion: cost


def _finite_valid_records(records: list[dict], criteria: tuple[str, ...]) -> list[dict]:
    valid = []
    for r in records:
        if not r.get("valid", True):
            continue
        if all(c in r and math.isfinite(float(r[c])) for c in criteria):
            valid.append(r)
    return valid


def get_pareto_front(records: list[dict]) -> list[dict]:
    valid = _finite_valid_records(records, DEFAULT_CRITERIA)
    if not valid:
        return []

    objectives = np.array(
        [[-r["log_synflow"], -r["zico"], r["flops_billion"]] for r in valid],
        dtype=float,
    )
    nd_idx = NonDominatedSorting().do(objectives, only_non_dominated_front=True)
    return [valid[i] for i in nd_idx]


def rank_topsis(
    records: list[dict],
    *,
    weights: tuple[float, ...] = (0.4, 0.4, 0.2),
    criteria: tuple[str, ...] = DEFAULT_CRITERIA,
    benefit: tuple[bool, ...] = DEFAULT_BENEFIT,
) -> list[dict]:
    eps = 1e-12
    valid = _finite_valid_records(records, criteria)
    if not valid:
        raise RuntimeError("No valid architectures available for TOPSIS ranking.")

    decision_matrix = np.array([[float(r[c]) for c in criteria] for r in valid], dtype=float)

    weights_arr = np.array(weights, dtype=float)
    weights_arr = weights_arr / (np.sum(weights_arr) + eps)

    norm = np.sqrt(np.sum(decision_matrix**2, axis=0)) + eps
    weighted_normalized = (decision_matrix / norm) * weights_arr

    ideal_best = np.zeros(len(criteria))
    ideal_worst = np.zeros(len(criteria))
    for j, is_benefit in enumerate(benefit):
        column = weighted_normalized[:, j]
        if is_benefit:
            ideal_best[j], ideal_worst[j] = column.max(), column.min()
        else:
            ideal_best[j], ideal_worst[j] = column.min(), column.max()

    dist_to_best = np.sqrt(np.sum((weighted_normalized - ideal_best) ** 2, axis=1))
    dist_to_worst = np.sqrt(np.sum((weighted_normalized - ideal_worst) ** 2, axis=1))
    closeness = dist_to_worst / (dist_to_best + dist_to_worst + eps)

    ranked = []
    for i, record in enumerate(valid):
        entry = dict(record)
        entry["topsis_distance_to_ideal_best"] = float(dist_to_best[i])
        entry["topsis_distance_to_ideal_worst"] = float(dist_to_worst[i])
        entry["topsis_score"] = float(closeness[i])
        ranked.append(entry)

    ranked.sort(key=lambda r: r["topsis_score"], reverse=True)
    return ranked
