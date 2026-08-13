"""Legacy-style fixed augmentation: pick 2 of 7 ops, apply both to every
image, no per-sample adaptivity (branch: fixed_da).

This reproduces the *mechanism* of the legacy repo's augmentation search
(`Evaluate.genetic_algorithm` / `auto_search_daapolicy` in the original
`evaluate.py`, confirmed live-called from `so_ga.py`): a genetic algorithm
searches over WHICH 2 of a fixed op menu to combine, not over continuous
per-op parameters, and once chosen, both ops are applied to every training
image with no notion of "this sample is currently easy/hard for the model."
That is a real, deliberate design point of the original -- SapAugment's
whole premise (magnitude driven by a sample's own loss rank, `search_space.py`
in the `augment/` package) doesn't exist here at all.

Two things are still improved over the legacy code on purpose, because they
are correctness properties independent of "adaptive vs fixed", not part of
what's being compared:

  - Decoding is pure arithmetic, no `random.*` call anywhere in this module
    (`search_space/chromosome.py`'s docstring explains why this matters: the
    legacy `so_ga.py` calls `random.choice()` *inside* decode, so the same
    chromosome can decode to a different policy on every evaluation).
  - The op menu itself (`AUGMENTATION_OPS`, `MAGNITUDE_RANGES` in
    `search_space.py`) is the already MRI-safety-vetted one this project
    uses everywhere else, not the legacy menu (which includes unsafe ranges
    such as 30-degree rotation and hue/saturation jitter).

Magnitude and probability, for whichever 2 ops are selected: fixed at each
op's safety-vetted maximum (`MAGNITUDE_RANGES[name][1]`) and applied with
probability 1.0. This is a stated default, not something the legacy paper
specifies (the legacy search picks *which* ops, not their strength) -- the
alternative of leaving probability/magnitude as additional searched genes
was considered and rejected because it would blur the actual comparison
this branch exists to make: "adaptive per-sample strength vs. fixed
strength", not "3-gene-per-op search vs. 2-gene op-only search".
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from brainmri_nas.augment.genotype import AugmentationStep, ResolvedAugmentationStep
from brainmri_nas.augment.search_space import AUGMENTATION_OPS, MAGNITUDE_RANGES, build_parameters

NUM_OPS_TO_SELECT = 2
FIXED_PROBABILITY = 1.0


def chromosome_length() -> int:
    return NUM_OPS_TO_SELECT


def validate_chromosome(chromosome: Sequence[float]) -> None:
    expected = chromosome_length()
    if len(chromosome) != expected:
        raise ValueError(f"Legacy augmentation chromosome has length {len(chromosome)}, expected {expected}.")
    for i, gene in enumerate(chromosome):
        if not (0.0 <= gene < 1.0):
            raise ValueError(f"Legacy augmentation chromosome gene {i} = {gene!r} is outside the required range [0, 1).")


def decode_legacy_chromosome(chromosome: Sequence[float]) -> tuple[str, str]:
    """2 genes -> 2 distinct op names, deterministically, no `random.*`.

    Gene 0 picks the first op from the full menu; gene 1 picks the second
    from the remaining `len(AUGMENTATION_OPS) - 1` ops, so a repeat is
    structurally impossible rather than checked-and-rejected."""
    validate_chromosome(chromosome)

    first_index = min(int(math.floor(chromosome[0] * len(AUGMENTATION_OPS))), len(AUGMENTATION_OPS) - 1)
    first_op = AUGMENTATION_OPS[first_index]

    remaining = [op for op in AUGMENTATION_OPS if op != first_op]
    second_index = min(int(math.floor(chromosome[1] * len(remaining))), len(remaining) - 1)
    second_op = remaining[second_index]

    return first_op, second_op


def resolve_fixed_step(name: str, order: int) -> ResolvedAugmentationStep:
    """The fixed-magnitude, always-applied counterpart of `search_space.py`'s
    `resolve_step` -- same op menu and same safe ranges, but the output does
    not depend on any sample's loss rank."""
    magnitude = MAGNITUDE_RANGES[name][1]
    return ResolvedAugmentationStep(
        name=name,
        order=order,
        probability=FIXED_PROBABILITY,
        magnitude=magnitude,
        parameters=build_parameters(name, magnitude),
    )


def legacy_steps(op1: str, op2: str) -> tuple[AugmentationStep, AugmentationStep]:
    """AugmentationStep records for the 2 selected ops, for JSON logging
    (`legacy_policy_archive.json` / `selected_legacy_policy.json`) -- these
    are descriptive only; `resolve_fixed_step` is what actually builds the
    transform, since strength_s/strength_a have no meaning in a fixed policy."""
    return (
        AugmentationStep(name=op1, order=0, probability=FIXED_PROBABILITY, strength_s=0.0, strength_a=0.0),
        AugmentationStep(name=op2, order=1, probability=FIXED_PROBABILITY, strength_s=0.0, strength_a=0.0),
    )
