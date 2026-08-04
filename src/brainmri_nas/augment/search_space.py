"""MRI-safe, sample-adaptive augmentation search space (handoff §7, extended
with SapAugment-style per-sample adaptivity -- Hu et al. 2021).

Conservative by design: only the ops handoff §7 explicitly recommends are
included, each restricted to the recommended magnitude range. Explicitly
excluded (per §7's "avoid or justify carefully" list): vertical flips, large
rotations, strong color jitter, hue/saturation changes, large crops, severe
affine deformation, large-area random erasing.

Chromosome layout: 3 genes per op (probability, strength_s, strength_a),
each in [0, 1), in the fixed canonical order below -- that canonical order
*is* each step's `order` field, so "order" doesn't need its own search
dimension. A policy where every probability gene decodes near 0 is the
identity policy; there's no separate identity gene.

`strength_s`/`strength_a` don't map to a fixed magnitude directly -- they
parameterize a curve (`resolve_step`, via the regularized incomplete beta
function, exactly as in the SapAugment paper) from a *sample's current loss
rank* to how strongly this op should be applied to that specific sample.
Easy samples (low relative loss) get pushed toward the strong end of the
op's magnitude range; hard samples get pushed toward the mild end. This is
why there's no single "magnitude" gene anymore -- magnitude is now a
function evaluated per sample, not a policy-wide constant.

Decoding the chromosome itself is still pure arithmetic, no randomness,
mirroring the NAS chromosome decoder in `search_space/chromosome.py`.

Class-specific strength (`class_scales`): after the per-op genes, `num_classes`
trailing genes each decode to a log-scale multiplier in `CLASS_SCALE_RANGE`,
one per true class. `resolve_step` applies a sample's class multiplier to
its already-computed per-sample `lam` (clipped back to [0, 1]) before
mapping into the op's magnitude range -- so a class scale nudges the whole
per-sample adaptivity curve up or down for that class without overriding
the underlying "easy sample -> stronger augmentation" logic. This exists
because loss ranks are computed globally across all classes (`LossCache`),
so a persistently-harder class (e.g. one with more visually diverse
presentations) clusters toward the "hard" end of that global ranking even
for its own easiest samples, systematically under-augmenting it relative to
what within-class-relative difficulty would suggest.
"""

from __future__ import annotations

from collections.abc import Sequence

from scipy.special import betainc

from brainmri_nas.augment.genotype import AugmentationPolicy, AugmentationStep, ResolvedAugmentationStep

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

# (low, high) magnitude bounds, per handoff §7's recommended ranges. This is
# the range strength_s/strength_a's curve maps a loss rank *into* -- not a
# fixed value anymore.
MAGNITUDE_RANGES: dict[str, tuple[float, float]] = {
    "rotation": (0.0, 15.0),  # degrees, up to +/-15 deg
    "affine_translation": (0.0, 0.10),  # translate fraction, up to 5-10%
    "random_resized_crop": (0.0, 0.15),  # subtracted from 1.0 -> scale_min in [0.85, 1.0]
    "brightness": (0.0, 0.2),  # mild
    "contrast": (0.0, 0.2),  # mild
    "horizontal_flip": (0.0, 0.0),  # unused -- flip is controlled by probability alone
    "random_erasing": (0.0, 0.05),  # small-area
}

# strength_s: "sharpness" of the loss-rank -> strength curve (paper tests
# s in {1, 10, 100}); decoded on a log scale since sharpness parameters are
# conventionally searched that way, not linearly.
STRENGTH_S_RANGE: tuple[float, float] = (1.0, 100.0)
# strength_a: shape/center of the curve, must stay strictly inside (0, 1) --
# it's used as s*(1-a) and s*a, both of which must stay positive for a
# valid beta function.
STRENGTH_A_RANGE: tuple[float, float] = (0.05, 0.95)

GENES_PER_OP = 3  # probability, strength_s, strength_a

DEFAULT_NUM_CLASSES = 4
CLASS_SCALE_RANGE: tuple[float, float] = (0.5, 2.0)


def chromosome_length(num_classes: int = DEFAULT_NUM_CLASSES) -> int:
    return len(AUGMENTATION_OPS) * GENES_PER_OP + num_classes


def validate_chromosome(chromosome: Sequence[float], num_classes: int = DEFAULT_NUM_CLASSES) -> None:
    expected = chromosome_length(num_classes)
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


def decode_chromosome(chromosome: Sequence[float], num_classes: int = DEFAULT_NUM_CLASSES) -> AugmentationPolicy:
    validate_chromosome(chromosome, num_classes)

    steps = []
    for order, name in enumerate(AUGMENTATION_OPS):
        # float(...): chromosome may be a numpy array (e.g. from pymoo) -- keep
        # decoded policies plain-Python-native, never leak numpy scalar types.
        probability_gene = float(chromosome[order * GENES_PER_OP])
        s_gene = float(chromosome[order * GENES_PER_OP + 1])
        a_gene = float(chromosome[order * GENES_PER_OP + 2])

        s_low, s_high = STRENGTH_S_RANGE
        strength_s = s_low * (s_high / s_low) ** s_gene  # log-scale interpolation

        a_low, a_high = STRENGTH_A_RANGE
        strength_a = a_low + a_gene * (a_high - a_low)

        steps.append(
            AugmentationStep(
                name=name,
                order=order,
                probability=probability_gene,
                strength_s=strength_s,
                strength_a=strength_a,
            )
        )

    class_scale_offset = len(AUGMENTATION_OPS) * GENES_PER_OP
    scale_low, scale_high = CLASS_SCALE_RANGE
    class_scales = tuple(
        scale_low * (scale_high / scale_low) ** float(chromosome[class_scale_offset + i])  # log-scale
        for i in range(num_classes)
    )

    return AugmentationPolicy(steps=tuple(steps), class_scales=class_scales)


def sap_strength(strength_s: float, strength_a: float, loss_rank: float) -> float:
    """The SapAugment policy function f_{s,a}(loss_rank) -> lambda in [0, 1]
    (handoff-adjacent, but this is the paper's Eq. 1: `1 - I(s(1-a), s*a; l)`,
    I being the regularized incomplete beta function). loss_rank=0 (easiest
    sample) should map toward strong augmentation, loss_rank=1 (hardest)
    toward mild -- `betainc` is increasing in its third argument, so this
    inversion (`1 - ...`) is what gives that direction."""
    p = strength_s * (1.0 - strength_a)
    q = strength_s * strength_a
    return float(1.0 - betainc(p, q, loss_rank))


def resolve_step(step: AugmentationStep, loss_rank: float, class_scale: float = 1.0) -> ResolvedAugmentationStep:
    """Evaluate one policy step's curve at a specific sample's loss rank,
    producing a concrete magnitude/parameters for that sample right now.
    `class_scale` (default 1.0 = no change) multiplies the resulting lambda
    before it's clipped back into [0, 1] and mapped into the op's magnitude
    range -- see module docstring."""
    lam = sap_strength(step.strength_s, step.strength_a, loss_rank)
    lam = min(max(lam * class_scale, 0.0), 1.0)
    low, high = MAGNITUDE_RANGES[step.name]
    magnitude = low + lam * (high - low)

    return ResolvedAugmentationStep(
        name=step.name,
        order=step.order,
        probability=step.probability,
        magnitude=magnitude,
        parameters=_build_parameters(step.name, magnitude),
    )
