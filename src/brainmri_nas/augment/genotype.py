"""Serializable augmentation policy representation.

A policy is a plain JSON array of steps, never raw torchvision transform
objects. Each `AugmentationStep` is the *searchable* representation of one
operation: `probability` (chance it's applied at all) plus `strength_s`/
`strength_a`, the two hyperparameters of a SapAugment-style (Hu et al. 2021)
curve that maps a sample's current loss rank to how strongly this op should
be applied to *that* sample -- there is no longer a single fixed magnitude
per policy, since the whole point is that magnitude now varies per sample.
`order` is stored explicitly (not just implied by list position) so a
policy loaded out of order can still be rebuilt correctly.

`ResolvedAugmentationStep` is the *concrete*, ephemeral counterpart: what
you get after evaluating a step's curve at one specific sample's loss rank,
with an actual `magnitude` and torchvision-ready `parameters` dict -- this
is what `transform_builder.py` turns into a real transform. It's built
fresh per sample, per read, and is never itself saved to JSON.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AugmentationStep:
    name: str
    order: int
    probability: float
    strength_s: float
    strength_a: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "order": self.order,
            "probability": self.probability,
            "strength_s": self.strength_s,
            "strength_a": self.strength_a,
        }

    @staticmethod
    def from_dict(data: dict) -> "AugmentationStep":
        return AugmentationStep(
            name=str(data["name"]),
            order=int(data["order"]),
            probability=float(data["probability"]),
            strength_s=float(data["strength_s"]),
            strength_a=float(data["strength_a"]),
        )


@dataclass(frozen=True)
class AugmentationPolicy:
    steps: tuple[AugmentationStep, ...]

    def ordered_steps(self) -> tuple[AugmentationStep, ...]:
        return tuple(sorted(self.steps, key=lambda s: s.order))

    def to_dict(self) -> list[dict]:
        return [s.to_dict() for s in self.ordered_steps()]

    @staticmethod
    def from_dict(data: list[dict]) -> "AugmentationPolicy":
        return AugmentationPolicy(steps=tuple(AugmentationStep.from_dict(d) for d in data))


@dataclass(frozen=True)
class ResolvedAugmentationStep:
    """One step's operation, resolved to a concrete magnitude/parameters for
    one specific sample at one specific loss rank. Ephemeral -- built fresh
    per `__getitem__` call, never persisted."""

    name: str
    order: int
    probability: float
    magnitude: float
    parameters: dict
