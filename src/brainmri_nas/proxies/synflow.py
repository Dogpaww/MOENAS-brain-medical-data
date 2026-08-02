"""SynFlow zero-cost proxy (handoff §17).

Runs entirely on CPU in float64, regardless of the model's current device:
Apple's MPS backend has no float64 support at all (a hardware limitation,
not a missing torch feature), and SynFlow's score is a product accumulated
across every layer's parameters, which needs the extra precision to stay
numerically sane. The model is restored to its original device and dtype
before returning, so callers can immediately follow this with ZiCO/FLOPs
profiling on whatever device they were already using.

Only `nn.Conv2d`/`nn.Linear` weights are linearized (via `named_parameters`,
not `state_dict`, so BatchNorm running-stat buffers are never touched) --
matching what the score is actually supposed to measure.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


@torch.no_grad()
def _linearize(model: nn.Module) -> dict[str, torch.Tensor]:
    signs = {}
    for name, param in model.named_parameters():
        signs[name] = torch.sign(param)
        param.abs_()
    return signs


@torch.no_grad()
def _restore_signs(model: nn.Module, signs: dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        param.mul_(signs[name])


def compute_synflow(model: nn.Module, *, input_shape: tuple[int, int, int]) -> dict:
    """input_shape: (channels, height, width). Batch size is fixed at 1 per
    handoff §17's recommendation -- SynFlow's all-ones trick doesn't need a
    real batch, just the right shape."""
    original_device = next(model.parameters()).device
    original_dtype = next(model.parameters()).dtype
    was_training = model.training

    model.to("cpu")
    model.eval()
    model.double()

    signs = _linearize(model)
    try:
        model.zero_grad(set_to_none=True)
        dummy_input = torch.ones([1, *input_shape], dtype=torch.float64)
        output = model(dummy_input)
        torch.sum(output).backward()

        total = torch.zeros((), dtype=torch.float64)
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight.grad is not None:
                total = total + (module.weight * module.weight.grad).abs().sum()

        raw_score = float(total.item())
    finally:
        _restore_signs(model, signs)
        model.zero_grad(set_to_none=True)
        model.to(device=original_device, dtype=original_dtype)
        model.train(was_training)

    if not math.isfinite(raw_score):
        raw_score = 0.0
    safe_score = max(raw_score, 1e-300)

    return {"synflow": raw_score, "log_synflow": math.log10(safe_score)}
