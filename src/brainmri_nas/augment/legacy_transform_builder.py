"""Fixed (non-adaptive) transform pipeline for the 2 legacy-selected ops
(branch: fixed_da). Mirrors `transform_builder.py`'s
`build_sample_adaptive_transform` structure exactly, reusing its
`build_transform_step` so the torchvision construction for a given op name
lives in exactly one place regardless of which search mechanism resolved it
-- only *how the magnitude was decided* differs between the two builders.
"""

from __future__ import annotations

from torchvision import transforms

from brainmri_nas.augment.legacy_search_space import resolve_fixed_step
from brainmri_nas.augment.search_space import TENSOR_SPACE_OPS
from brainmri_nas.augment.transform_builder import build_transform_step
from brainmri_nas.data.transforms import build_train_transform


def build_legacy_transform(op1: str, op2: str, image_size: int) -> transforms.Compose:
    pil_ops = []
    tensor_ops = []

    for order, name in enumerate((op1, op2)):
        resolved = resolve_fixed_step(name, order)
        transform = build_transform_step(resolved, image_size)
        if resolved.name in TENSOR_SPACE_OPS:
            tensor_ops.append(transform)
        else:
            pil_ops.append(transform)

    return build_train_transform(image_size, pil_augmentation_ops=pil_ops, tensor_augmentation_ops=tensor_ops)
