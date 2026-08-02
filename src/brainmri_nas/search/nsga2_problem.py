"""pymoo problem wrapper (handoff §22).

Objective vector: `f1 = -log10(SynFlow)`, `f2 = -ZiCO`, `f3 = FLOPs` --
pymoo always minimizes, and SynFlow/ZiCO should be maximized, so both get
negated; FLOPs is a genuine cost and is minimized directly.

The decision vector `x` lives in `[0, 1)` (handoff chromosome convention);
`xu` is kept a hair under 1.0 so pymoo's box constraints can never hand the
decoder a gene of exactly 1.0, which `validate_chromosome` would reject.

Every evaluated candidate (cache hits included, so the archive reflects
what NSGA-II actually looked at each generation) is appended to `archive`.
"""

from __future__ import annotations

import numpy as np
import torch
from pymoo.core.problem import ElementwiseProblem

from brainmri_nas.search.candidate import CandidateCache, evaluate_candidate

CHROMOSOME_UPPER_BOUND = 1.0 - 1e-9


class NASSearchProblem(ElementwiseProblem):
    def __init__(
        self,
        *,
        n_var: int,
        cache: CandidateCache,
        archive: list[dict],
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
    ):
        super().__init__(
            n_var=n_var,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.zeros(n_var),
            xu=np.full(n_var, CHROMOSOME_UPPER_BOUND),
        )
        self.cache = cache
        self.archive = archive
        self._candidate_kwargs = dict(
            num_intermediate_nodes=num_intermediate_nodes,
            edges_per_node=edges_per_node,
            input_channels=input_channels,
            num_classes=num_classes,
            image_size=image_size,
            initial_channels=initial_channels,
            number_of_cells=number_of_cells,
            stem_type=stem_type,
            proxy_batches=proxy_batches,
            device=device,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        record = evaluate_candidate(x, cache=self.cache, **self._candidate_kwargs)

        out["F"] = np.array(
            [-record["log_synflow"], -record["zico"], record["flops_billion"]],
            dtype=float,
        )
        self.archive.append(record)
