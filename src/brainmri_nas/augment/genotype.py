"""Serializable augmentation policy representation (handoff §8).

A policy is a plain JSON array of steps, never raw torchvision transform
objects -- each step records `operation` (`name`), `probability`,
`magnitude`, and `order`, plus the concrete `parameters` a transform builder
would need to reconstruct the actual torchvision op. `order` is stored
explicitly (not just implied by list position) so a policy loaded out of
order can still be rebuilt correctly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AugmentationStep:
    name: str
    order: int
    probability: float
    magnitude: float
    parameters: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "order": self.order,
            "probability": self.probability,
            "magnitude": self.magnitude,
            "parameters": dict(self.parameters),
        }

    @staticmethod
    def from_dict(data: dict) -> "AugmentationStep":
        return AugmentationStep(
            name=str(data["name"]),
            order=int(data["order"]),
            probability=float(data["probability"]),
            magnitude=float(data["magnitude"]),
            parameters=dict(data["parameters"]),
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
