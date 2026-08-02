"""JSON-safe conversion helpers for dataclasses, tuples, and numpy/torch scalars.

Search/training records must be plain JSON (per handoff: no model objects, no
CUDA tensors, no numpy types with non-standard repr) so every save path in
this project should route through `to_jsonable` rather than calling
`json.dump` directly on a dataclass or numpy value.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def dump_json(value: Any, path: str | Path, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(value), f, indent=indent)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)
