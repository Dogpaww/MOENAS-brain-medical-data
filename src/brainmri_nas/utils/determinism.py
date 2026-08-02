"""Deterministic seeding helpers.

Centralized so every entry point (search, augmentation search, final training,
tests) seeds the same set of RNGs the same way instead of relying on
whatever ambient random state happens to exist.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
