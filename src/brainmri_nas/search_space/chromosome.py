"""Chromosome layout and deterministic decoding (handoff §9-11).

Chromosome = a flat tuple of floats in [0, 1), length `chromosome_length()`
for the topology portion, or `total_chromosome_length()` including the 2
trailing size genes (depth, width). Layout: for each of the 2 cell types
(normal, then reduction), for each of `num_intermediate_nodes` intermediate
nodes in order, for each of `edges_per_node` incoming edges, 3 consecutive
genes: (operation, source, attention). With the defaults (4 nodes, 2
edges/node) that's 2 * 4 * 2 * 3 = 48 topology genes, followed by 2 size
genes (number_of_cells, initial_channels) -- 50 genes total.

Decoding is pure arithmetic -- no `random.*` call anywhere in this module.
The valid source range for intermediate node `i` (0-based) is `{0, ..., i+1}`
(cell inputs are states 0 and 1, plus every earlier intermediate node),
exactly the bounds handoff §10 spells out. This is a deliberate fix vs. the
legacy repo's `Population.setup_NAS`, whose off-by-one bound only ever
allows source 0 for the first node's two edges.

`validate_and_repair_genotype` operates one level up, on an already-decoded
`NetworkGenotype` rather than the raw float chromosome (which is in-range by
construction). It exists so a genotype loaded from hand-edited or
externally-produced JSON can't silently encode a different architecture: an
out-of-range source is deterministically repaired via modulo, never
resampled.

Size genes (`decode_size_genes`/`decode_full_chromosome`) were added after
measuring that a fixed `number_of_cells`/`initial_channels` (the original
Stage 1 design, kept for apples-to-apples proxy comparison) collapses the
FLOPs objective into a narrow band while SynFlow/ZiCO stay wide, making the
3D Pareto front look like a flat slant instead of a real trade-off surface.
`number_of_cells` decodes to a discrete uniform bucket (depth's cost scales
roughly linearly, so a linear range is fine); `initial_channels` decodes on
a log scale (width's FLOPs cost scales roughly quadratically, so a log-scale
gene spends its resolution more evenly across the practical range) -- see
`docs/` discussion in the NSGA-II topsis_weights rebalance for the measured
elasticities behind this and the accompanying weight change.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from brainmri_nas.search_space.attentions import ATTENTION_PRIMITIVES
from brainmri_nas.search_space.genotype import CellGenotype, EdgeGene, NetworkGenotype
from brainmri_nas.search_space.operations import PRIMITIVES

NUM_CELL_TYPES = 2  # normal, reduction
GENES_PER_EDGE = 3  # operation, source, attention
DEFAULT_NUM_INTERMEDIATE_NODES = 4
DEFAULT_EDGES_PER_NODE = 2

NUM_SIZE_GENES = 2  # number_of_cells, initial_channels
DEFAULT_NUMBER_OF_CELLS_RANGE = (3, 11)  # lower bound matches NASNetwork's own floor
DEFAULT_INITIAL_CHANNELS_RANGE = (8, 48)

Chromosome = Sequence[float]


def valid_source_count(node_index: int) -> int:
    """Number of valid source states for intermediate node `node_index` (0-based):
    the 2 cell inputs plus every earlier intermediate node."""
    return 2 + node_index


def chromosome_length(
    num_intermediate_nodes: int = DEFAULT_NUM_INTERMEDIATE_NODES,
    edges_per_node: int = DEFAULT_EDGES_PER_NODE,
) -> int:
    return NUM_CELL_TYPES * num_intermediate_nodes * edges_per_node * GENES_PER_EDGE


def total_chromosome_length(
    num_intermediate_nodes: int = DEFAULT_NUM_INTERMEDIATE_NODES,
    edges_per_node: int = DEFAULT_EDGES_PER_NODE,
) -> int:
    """Topology genes plus the 2 trailing size genes -- this is the length
    NSGA-II's decision vector actually uses now, not `chromosome_length()`."""
    return chromosome_length(num_intermediate_nodes, edges_per_node) + NUM_SIZE_GENES


def decode_size_genes(
    chromosome: Chromosome,
    num_intermediate_nodes: int = DEFAULT_NUM_INTERMEDIATE_NODES,
    edges_per_node: int = DEFAULT_EDGES_PER_NODE,
    *,
    number_of_cells_range: tuple[int, int] = DEFAULT_NUMBER_OF_CELLS_RANGE,
    initial_channels_range: tuple[int, int] = DEFAULT_INITIAL_CHANNELS_RANGE,
) -> tuple[int, int]:
    """Reads the 2 genes immediately after the topology portion. Returns
    (number_of_cells, initial_channels)."""
    offset = chromosome_length(num_intermediate_nodes, edges_per_node)
    expected = offset + NUM_SIZE_GENES
    if len(chromosome) != expected:
        raise ValueError(f"Chromosome has length {len(chromosome)}, expected {expected} (with size genes).")

    depth_gene, width_gene = chromosome[offset], chromosome[offset + 1]

    cells_low, cells_high = number_of_cells_range
    num_cell_choices = cells_high - cells_low + 1
    number_of_cells = cells_low + _bucket(depth_gene, num_cell_choices)

    channels_low, channels_high = initial_channels_range
    raw_channels = channels_low * (channels_high / channels_low) ** width_gene
    initial_channels = min(max(round(raw_channels), channels_low), channels_high)

    return number_of_cells, initial_channels


def decode_full_chromosome(
    chromosome: Chromosome,
    num_intermediate_nodes: int = DEFAULT_NUM_INTERMEDIATE_NODES,
    edges_per_node: int = DEFAULT_EDGES_PER_NODE,
    *,
    number_of_cells_range: tuple[int, int] = DEFAULT_NUMBER_OF_CELLS_RANGE,
    initial_channels_range: tuple[int, int] = DEFAULT_INITIAL_CHANNELS_RANGE,
) -> tuple[NetworkGenotype, int, int]:
    """Convenience wrapper: topology genes -> NetworkGenotype (via the
    unchanged `decode_chromosome`) plus the 2 trailing size genes ->
    (number_of_cells, initial_channels)."""
    topology_length = chromosome_length(num_intermediate_nodes, edges_per_node)
    genotype = decode_chromosome(chromosome[:topology_length], num_intermediate_nodes, edges_per_node)
    number_of_cells, initial_channels = decode_size_genes(
        chromosome,
        num_intermediate_nodes,
        edges_per_node,
        number_of_cells_range=number_of_cells_range,
        initial_channels_range=initial_channels_range,
    )
    return genotype, number_of_cells, initial_channels


def validate_chromosome(
    chromosome: Chromosome,
    num_intermediate_nodes: int = DEFAULT_NUM_INTERMEDIATE_NODES,
    edges_per_node: int = DEFAULT_EDGES_PER_NODE,
) -> None:
    expected = chromosome_length(num_intermediate_nodes, edges_per_node)
    if len(chromosome) != expected:
        raise ValueError(f"Chromosome has length {len(chromosome)}, expected {expected}.")
    for i, gene in enumerate(chromosome):
        if not (0.0 <= gene < 1.0):
            raise ValueError(f"Chromosome gene {i} = {gene!r} is outside the required range [0, 1).")


def _bucket(gene: float, num_choices: int) -> int:
    index = int(math.floor(gene * num_choices))
    # Defensive clamp only: for gene in [0, 1) this is already in [0, num_choices - 1].
    return min(max(index, 0), num_choices - 1)


def _decode_cell(
    genes: Chromosome,
    num_intermediate_nodes: int,
    edges_per_node: int,
) -> CellGenotype:
    edges: list[EdgeGene] = []
    gene_idx = 0
    for node_index in range(num_intermediate_nodes):
        n_valid_sources = valid_source_count(node_index)
        for _ in range(edges_per_node):
            op_gene, source_gene, attn_gene = genes[gene_idx], genes[gene_idx + 1], genes[gene_idx + 2]
            gene_idx += 3
            edges.append(
                EdgeGene(
                    source=_bucket(source_gene, n_valid_sources),
                    operation=PRIMITIVES[_bucket(op_gene, len(PRIMITIVES))],
                    attention=ATTENTION_PRIMITIVES[_bucket(attn_gene, len(ATTENTION_PRIMITIVES))],
                )
            )
    concat_nodes = tuple(range(2, 2 + num_intermediate_nodes))
    return CellGenotype(edges=tuple(edges), concat_nodes=concat_nodes)


def decode_chromosome(
    chromosome: Chromosome,
    num_intermediate_nodes: int = DEFAULT_NUM_INTERMEDIATE_NODES,
    edges_per_node: int = DEFAULT_EDGES_PER_NODE,
) -> NetworkGenotype:
    validate_chromosome(chromosome, num_intermediate_nodes, edges_per_node)
    half = chromosome_length(num_intermediate_nodes, edges_per_node) // 2
    normal = _decode_cell(chromosome[:half], num_intermediate_nodes, edges_per_node)
    reduction = _decode_cell(chromosome[half:], num_intermediate_nodes, edges_per_node)
    return NetworkGenotype(normal=normal, reduction=reduction)


def _validate_and_repair_cell(
    cell: CellGenotype,
    num_intermediate_nodes: int,
    edges_per_node: int,
) -> CellGenotype:
    expected_num_edges = num_intermediate_nodes * edges_per_node
    if len(cell.edges) != expected_num_edges:
        raise ValueError(f"Cell has {len(cell.edges)} edges, expected {expected_num_edges}.")

    expected_concat = tuple(range(2, 2 + num_intermediate_nodes))
    if tuple(cell.concat_nodes) != expected_concat:
        raise ValueError(f"Cell concat_nodes {cell.concat_nodes} != expected {expected_concat}.")

    repaired_edges: list[EdgeGene] = []
    edge_idx = 0
    for node_index in range(num_intermediate_nodes):
        n_valid_sources = valid_source_count(node_index)
        for _ in range(edges_per_node):
            edge = cell.edges[edge_idx]
            edge_idx += 1

            if edge.operation not in PRIMITIVES:
                raise ValueError(f"Edge {edge_idx - 1}: unknown operation {edge.operation!r}.")
            if edge.attention not in ATTENTION_PRIMITIVES:
                raise ValueError(f"Edge {edge_idx - 1}: unknown attention {edge.attention!r}.")

            source = edge.source
            if not (0 <= source < n_valid_sources):
                # Deterministic repair, never a random resample.
                source = source % n_valid_sources

            repaired_edges.append(EdgeGene(source=source, operation=edge.operation, attention=edge.attention))

    return CellGenotype(edges=tuple(repaired_edges), concat_nodes=cell.concat_nodes)


def validate_and_repair_genotype(
    genotype: NetworkGenotype,
    num_intermediate_nodes: int = DEFAULT_NUM_INTERMEDIATE_NODES,
    edges_per_node: int = DEFAULT_EDGES_PER_NODE,
) -> NetworkGenotype:
    return NetworkGenotype(
        normal=_validate_and_repair_cell(genotype.normal, num_intermediate_nodes, edges_per_node),
        reduction=_validate_and_repair_cell(genotype.reduction, num_intermediate_nodes, edges_per_node),
    )
