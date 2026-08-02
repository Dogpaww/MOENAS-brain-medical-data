"""MRI-safe augmentation search space (handoff §7).

Conservative by design: only the ops handoff §7 explicitly recommends are
included, each restricted to the recommended magnitude range. Explicitly
excluded (per §7's "avoid or justify carefully" list): vertical flips, large
rotations, strong color jitter, hue/saturation changes, large crops, severe
affine deformation, large-area random erasing.

Chromosome layout: 2 genes per op (probability, magnitude), each in [0, 1),
in the fixed canonical order below -- that canonical order *is* each step's
`order` field, so "order" doesn't need its own search dimension. A policy
where every probability gene decodes near 0 is the identity policy; there's
no separate identity gene.

Decoding is pure arithmetic, no randomness, mirroring the NAS chromosome
decoder in `search_space/chromosome.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

from brainmri_nas.augment.genotype import AugmentationPolicy, AugmentationStep

# Canonical pipeline order: geometric ops, then photometric ops, then
# tensor-space erasing. horizontal_flip and random_erasing use RandomApply's
# probability natively via torchvision (RandomHorizontalFlip(p=...),
# RandomErasing(p=...)) rather than magnitude, but keep a magnitude range
# entry for schema uniformity.
AUGMENTATION_OPS: tuple[str, ...] = (
    "rotation",
    "affine_translation",
    "random_resized_crop",
    "brightness",
    "contrast",
    "horizontal_flip",
    "random_erasing",
)

PIL_SPACE_OPS: frozenset[str] = frozenset(
    {"rotation", "affine_translation", "random_resized_crop", "brightness", "contrast", "horizontal_flip"}
)
TENSOR_SPACE_OPS: frozenset[str] = frozenset({"random_erasing"})

# (low, high) magnitude bounds, per handoff §7's recommended ranges.
MAGNITUDE_RANGES: dict[str, tuple[float, float]] = {
    "rotation": (0.0, 15.0),  # degrees, up to +/-15 deg
    "affine_translation": (0.0, 0.10),  # translate fraction, up to 5-10%
    "random_resized_crop": (0.0, 0.15),  # subtracted from 1.0 -> scale_min in [0.85, 1.0]
    "brightness": (0.0, 0.2),  # mild
    "contrast": (0.0, 0.2),  # mild
    "horizontal_flip": (0.0, 0.0),  # unused -- flip is controlled by probability alone
    "random_erasing": (0.0, 0.05),  # small-area
}

GENES_PER_OP = 2  # probability, magnitude


def chromosome_length() -> int:
    return len(AUGMENTATION_OPS) * GENES_PER_OP


def validate_chromosome(chromosome: Sequence[float]) -> None:
    expected = chromosome_length()
    if len(chromosome) != expected:
        raise ValueError(f"Augmentation chromosome has length {len(chromosome)}, expected {expected}.")
    for i, gene in enumerate(chromosome):
        if not (0.0 <= gene < 1.0):
            raise ValueError(f"Augmentation chromosome gene {i} = {gene!r} is outside the required range [0, 1).")


def _build_parameters(name: str, magnitude: float) -> dict:
    if name == "rotation":
        return {"degrees": magnitude}
    if name == "affine_translation":
        return {"translate": [magnitude, magnitude]}
    if name == "random_resized_crop":
        scale_min = max(1.0 - magnitude, 0.5)
        return {"scale_min": scale_min, "scale_max": 1.0}
    if name == "brightness":
        return {"brightness": magnitude}
    if name == "contrast":
        return {"contrast": magnitude}
    if name == "horizontal_flip":
        return {}
    if name == "random_erasing":
        return {"scale_min": 1e-3, "scale_max": max(magnitude, 1e-3)}
    raise ValueError(f"Unknown augmentation operation {name!r}.")


def decode_chromosome(chromosome: Sequence[float]) -> AugmentationPolicy:
    validate_chromosome(chromosome)

    steps = []
    for order, name in enumerate(AUGMENTATION_OPS):
        # float(...): chromosome may be a numpy array (e.g. from pymoo) -- keep
        # decoded policies plain-Python-native, never leak numpy scalar types.
        probability_gene = float(chromosome[order * GENES_PER_OP])
        magnitude_gene = float(chromosome[order * GENES_PER_OP + 1])

        low, high = MAGNITUDE_RANGES[name]
        magnitude = low + magnitude_gene * (high - low)

        steps.append(
            AugmentationStep(
                name=name,
                order=order,
                probability=probability_gene,
                magnitude=magnitude,
                parameters=_build_parameters(name, magnitude),
            )
        )

    return AugmentationPolicy(steps=tuple(steps))
