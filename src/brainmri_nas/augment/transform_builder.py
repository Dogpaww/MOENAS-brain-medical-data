"""Reconstruct an actual torchvision pipeline from a policy, resolved at one
specific sample's loss rank (handoff §8: "a policy builder must reconstruct
the actual torchvision pipeline"; extended for sample-adaptivity -- each
call resolves fresh, since a different sample or a refreshed loss cache
means different magnitudes).

Routes each step into Stage 1's `build_train_transform` PIL-space or
tensor-space slot, so the resize -> augmentation -> tensor -> normalize ->
tensor-augmentation ordering from handoff §6 is preserved without
duplicating that pipeline assembly logic here.
"""

from __future__ import annotations

from torchvision import transforms

from brainmri_nas.augment.genotype import AugmentationPolicy, ResolvedAugmentationStep
from brainmri_nas.augment.search_space import TENSOR_SPACE_OPS, resolve_step
from brainmri_nas.data.transforms import build_train_transform


def build_transform_step(step: ResolvedAugmentationStep, image_size: int):
    if step.name == "horizontal_flip":
        return transforms.RandomHorizontalFlip(p=step.probability)

    if step.name == "random_erasing":
        return transforms.RandomErasing(
            p=step.probability,
            scale=(step.parameters["scale_min"], step.parameters["scale_max"]),
        )

    if step.name == "rotation":
        base_op = transforms.RandomRotation(degrees=step.parameters["degrees"])
    elif step.name == "affine_translation":
        translate = step.parameters["translate"]
        base_op = transforms.RandomAffine(degrees=0, translate=(translate[0], translate[1]))
    elif step.name == "random_resized_crop":
        base_op = transforms.RandomResizedCrop(
            image_size,
            scale=(step.parameters["scale_min"], step.parameters["scale_max"]),
            antialias=True,
        )
    elif step.name == "brightness":
        base_op = transforms.ColorJitter(brightness=step.parameters["brightness"])
    elif step.name == "contrast":
        base_op = transforms.ColorJitter(contrast=step.parameters["contrast"])
    else:
        raise ValueError(f"Unknown augmentation operation {step.name!r}.")

    return transforms.RandomApply([base_op], p=step.probability)


def build_sample_adaptive_transform(
    policy: AugmentationPolicy, image_size: int, loss_rank: float
) -> transforms.Compose:
    pil_ops = []
    tensor_ops = []

    for step in policy.ordered_steps():
        resolved = resolve_step(step, loss_rank)
        transform = build_transform_step(resolved, image_size)
        if resolved.name in TENSOR_SPACE_OPS:
            tensor_ops.append(transform)
        else:
            pil_ops.append(transform)

    return build_train_transform(image_size, pil_augmentation_ops=pil_ops, tensor_augmentation_ops=tensor_ops)
