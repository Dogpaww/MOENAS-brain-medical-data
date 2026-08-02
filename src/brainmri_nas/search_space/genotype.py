"""Decoded architecture representation (handoff §9).

A `NetworkGenotype` is the fully-resolved, discrete description of one
architecture: which operation and attention module sits on each cell edge,
and which node each edge reads from. It contains no floats and no reference
to the chromosome that produced it, so it can be saved to JSON and used to
rebuild an identical model in a completely separate process (handoff §14).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeGene:
    source: int
    operation: str
    attention: str

    def to_dict(self) -> dict:
        return {"source": self.source, "operation": self.operation, "attention": self.attention}

    @staticmethod
    def from_dict(data: dict) -> "EdgeGene":
        return EdgeGene(source=int(data["source"]), operation=str(data["operation"]), attention=str(data["attention"]))


@dataclass(frozen=True)
class CellGenotype:
    edges: tuple[EdgeGene, ...]
    concat_nodes: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "edges": [e.to_dict() for e in self.edges],
            "concat_nodes": list(self.concat_nodes),
        }

    @staticmethod
    def from_dict(data: dict) -> "CellGenotype":
        return CellGenotype(
            edges=tuple(EdgeGene.from_dict(e) for e in data["edges"]),
            concat_nodes=tuple(int(n) for n in data["concat_nodes"]),
        )


@dataclass(frozen=True)
class NetworkGenotype:
    normal: CellGenotype
    reduction: CellGenotype

    def to_dict(self) -> dict:
        return {"normal": self.normal.to_dict(), "reduction": self.reduction.to_dict()}

    @staticmethod
    def from_dict(data: dict) -> "NetworkGenotype":
        return NetworkGenotype(
            normal=CellGenotype.from_dict(data["normal"]),
            reduction=CellGenotype.from_dict(data["reduction"]),
        )
