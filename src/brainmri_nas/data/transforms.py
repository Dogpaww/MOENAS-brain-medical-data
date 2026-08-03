"""Transform builders (handoff §5-6).

`build_train_transform` and `build_eval_transform` each return a **fresh**
`transforms.Compose` on every call -- never share one transform instance
across train/val/test (the exact bug in the legacy `evaluate.py`, which
passes a single `data_transform` object to all three `ImageFolder`s).

`build_eval_transform` is deterministic: resize+pad -> grayscale(3) ->
tensor -> per-image normalize, no randomness, used for validation, test,
and the train-eval view of the training split.

`build_train_transform` inserts optional augmentation ops in two places
(handoff §6): PIL-space ops between grayscale and tensor conversion, and
tensor-space ops (e.g. random erasing) after normalization. Both default to
empty, so calling it with no augmentation ops gives the same deterministic
pipeline as eval -- Stage 3 plugs real MRI-safe policies into these
callables without touching this module again.

Two deliberate choices, both driven by measuring the actual Kaggle dataset
rather than assuming:

- `ResizeLongerSideAndPad` instead of a plain stretch-to-square resize:
  ~21% of images in this dataset are non-square (up to notably elongated),
  and a plain `Resize((size, size))` scales width/height independently,
  distorting tumor shape by an amount that has nothing to do with the
  tumor itself. This preserves aspect ratio (one scale factor for both
  axes) and pads the shorter side with black instead, so no real anatomy
  ever gets stretched or cropped away.
- `PerImageNormalize` instead of a fixed `Normalize(mean=0.5, std=0.5)`:
  measured mean pixel intensity differs substantially between Training/
  and Testing/ for the same class (e.g. glioma_tumor: 35/255 in Training
  vs. 62/255 in Testing) -- a fixed constant doesn't correct for that, it
  just lets the model calibrate on Training's brightness range and then
  see systematically different numbers at test time. Normalizing each
  image by its own mean/std removes absolute brightness as a variable
  entirely, for both splits, by construction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

NORMALIZE_EPSILON = 1e-6


class ResizeLongerSideAndPad:
    """Resize preserving aspect ratio so the longer side matches `size`,
    then center-pad the shorter side with `fill` to reach `size` x `size`.
    """

    def __init__(self, size: int, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        scale = self.size / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        resized = TF.resize(img, [new_height, new_width], antialias=True)

        pad_left = (self.size - new_width) // 2
        pad_right = self.size - new_width - pad_left
        pad_top = (self.size - new_height) // 2
        pad_bottom = self.size - new_height - pad_top

        return TF.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size}, fill={self.fill})"


class PerImageNormalize:
    """Standardize a tensor by its own mean/std (computed over all pixels
    of that image), rather than a fixed dataset-wide constant."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + NORMALIZE_EPSILON)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            ResizeLongerSideAndPad(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            PerImageNormalize(),
        ]
    )


def build_train_transform(
    image_size: int,
    pil_augmentation_ops: Sequence[Callable] = (),
    tensor_augmentation_ops: Sequence[Callable] = (),
) -> transforms.Compose:
    ops = [
        ResizeLongerSideAndPad(image_size),
        transforms.Grayscale(num_output_channels=3),
        *pil_augmentation_ops,
        transforms.ToTensor(),
        PerImageNormalize(),
        *tensor_augmentation_ops,
    ]
    return transforms.Compose(ops)
