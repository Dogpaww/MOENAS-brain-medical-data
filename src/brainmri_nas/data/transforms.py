"""Transform builders (handoff §5-6).

`build_train_transform` and `build_eval_transform` each return a **fresh**
`transforms.Compose` on every call -- never share one transform instance
across train/val/test (the exact bug in the legacy `evaluate.py`, which
passes a single `data_transform` object to all three `ImageFolder`s).

`build_eval_transform` is deterministic: resize -> grayscale(3) ->
tensor -> normalize, no randomness, used for validation, test, and the
train-eval view of the training split.

`build_train_transform` inserts optional augmentation ops in two places
(handoff §6): PIL-space ops between grayscale and tensor conversion, and
tensor-space ops (e.g. random erasing) after normalization. Both default to
empty, so calling it with no augmentation ops gives the same deterministic
pipeline as eval -- Stage 3 plugs real MRI-safe policies into these
callables without touching this module again.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from torchvision import transforms

NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def build_train_transform(
    image_size: int,
    pil_augmentation_ops: Sequence[Callable] = (),
    tensor_augmentation_ops: Sequence[Callable] = (),
) -> transforms.Compose:
    ops = [
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.Grayscale(num_output_channels=3),
        *pil_augmentation_ops,
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        *tensor_augmentation_ops,
    ]
    return transforms.Compose(ops)
