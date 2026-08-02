"""FLOPs/param profiling (handoff §20) and optional peak-memory profiling (§21).

FLOPs/params are always measured at the model's actual configured input
size (handoff §20/§30 item 22: never profile at a different resolution than
the one used for search/training) via `thop`, which reports multiply-accumulate
operations under the name "flops" -- the same loose-but-conventional naming
the legacy repo and most NAS papers use, kept here for consistency with the
handoff's required field names.

Peak-memory profiling is best-effort and platform-dependent (handoff §21
explicitly marks it optional and not part of the default NSGA-II objective
vector, so this is *not* wired into search by default): CUDA gets real
peak-allocated/reserved tracking; CPU falls back to whole-process peak RSS
(`resource.getrusage`, approximate -- not model-isolated); MPS has no public
peak-memory API at all, so it reports current allocation after the backward
pass with an explicit caveat rather than pretending it's a peak.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
from thop import profile as thop_profile


def profile_flops_and_params(
    model: nn.Module,
    *,
    input_shape: tuple[int, int, int],
    device: torch.device,
) -> dict:
    model = model.to(device)
    was_training = model.training
    model.eval()

    dummy_input = torch.randn(1, *input_shape, device=device)
    with torch.no_grad():
        flops, params = thop_profile(model, inputs=(dummy_input,), verbose=False)

    model.train(was_training)

    return {
        "input_shape": [1, *input_shape],
        "flops": float(flops),
        "flops_billion": float(flops) / 1e9,
        "params": float(params),
        "params_million": float(params) / 1e6,
    }


def profile_peak_memory(
    model: nn.Module,
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    batch_size: int,
    device: torch.device,
    precision: str = "fp32",
) -> dict:
    model = model.to(device)
    was_training = model.training
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(batch_size, *input_shape, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)

    result = {
        "profiling_batch_size": batch_size,
        "resolution": list(input_shape[-2:]),
        "precision": precision,
        "device": str(device),
    }

    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(precision == "amp")):
                loss = loss_fn(model(x), y)
            loss.backward()
            result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        elif device.type == "cpu":
            import resource

            model.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            # ru_maxrss is bytes on macOS/BSD, kilobytes on Linux.
            scale = 1 if sys.platform == "darwin" else 1024
            result["peak_allocated_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)
            result["peak_reserved_bytes"] = None
            result["note"] = "CPU peak is whole-process RSS (approximate, not model-isolated)."
        else:
            model.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            result["peak_allocated_bytes"] = (
                int(torch.mps.current_allocated_memory()) if hasattr(torch, "mps") else None
            )
            result["peak_reserved_bytes"] = None
            result["note"] = (
                "MPS exposes no peak-memory API; this is current allocation after the "
                "backward pass, not a true peak."
            )
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    return result
