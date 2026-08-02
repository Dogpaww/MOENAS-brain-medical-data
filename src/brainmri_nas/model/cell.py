"""A DARTS-style cell built directly from a `CellGenotype`.

Structurally follows the legacy `Cell` design (which is standard DARTS, not
itself buggy): `preprocess0`/`preprocess1` align the two input states'
channel count (and, if the previous cell was a reduction cell, spatial
resolution too); each intermediate node sums `edges_per_node` (operation ->
attention) branches; the cell output concatenates every intermediate node on
the channel axis (handoff §12: shape contracts for addition/concatenation).

`num_intermediate_nodes`/`edges_per_node` are inferred from the genotype
itself (`len(concat_nodes)` and `len(edges) // len(concat_nodes)`) rather
than passed in separately, so a `Cell` is fully reconstructible from a
`CellGenotype` alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from brainmri_nas.search_space.attentions import ATTN
from brainmri_nas.search_space.genotype import CellGenotype
from brainmri_nas.search_space.operations import OPS, Identity, ReLUConvBN, FactorizedReduce, drop_path


class Cell(nn.Module):
    def __init__(
        self,
        genotype: CellGenotype,
        c_prev_prev: int,
        c_prev: int,
        c: int,
        reduction: bool,
        reduction_prev: bool,
        drop_path_probability: float = 0.0,
    ):
        super().__init__()
        self.genotype = genotype
        self.reduction = reduction
        self.drop_path_probability = drop_path_probability

        self.num_intermediate_nodes = len(genotype.concat_nodes)
        if self.num_intermediate_nodes == 0 or len(genotype.edges) % self.num_intermediate_nodes != 0:
            raise ValueError(
                f"Cell genotype has {len(genotype.edges)} edges and "
                f"{self.num_intermediate_nodes} concat nodes -- edges must divide evenly across nodes."
            )
        self.edges_per_node = len(genotype.edges) // self.num_intermediate_nodes
        self.multiplier = self.num_intermediate_nodes

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(c_prev, c, 1, 1, 0)

        self.ops = nn.ModuleList()
        self.attns = nn.ModuleList()
        for edge in genotype.edges:
            stride = 2 if reduction and edge.source < 2 else 1
            self.ops.append(OPS[edge.operation](c, stride, True))
            self.attns.append(ATTN[edge.attention](c))

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        edge_idx = 0
        for _ in range(self.num_intermediate_nodes):
            branch_outputs = []
            for _ in range(self.edges_per_node):
                edge = self.genotype.edges[edge_idx]
                op = self.ops[edge_idx]
                attn = self.attns[edge_idx]

                h = attn(op(states[edge.source]))
                if self.training and self.drop_path_probability > 0.0 and not isinstance(op, Identity):
                    h = drop_path(h, self.drop_path_probability)

                branch_outputs.append(h)
                edge_idx += 1

            states.append(sum(branch_outputs))

        return torch.cat([states[i] for i in self.genotype.concat_nodes], dim=1)
