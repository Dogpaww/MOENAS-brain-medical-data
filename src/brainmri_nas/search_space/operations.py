"""DARTS-style operation primitives (handoff §13: initial operation space).

Every op is built as `OPS[name](C, stride, affine)` and obeys the shape
contract from handoff §12:
    stride 1: [B, C, H, W]     -> [B, C, H, W]
    stride 2: [B, C, H, W]     -> [B, C, H/2, W/2]
(channel count is always preserved here; the network builder controls
channel growth between cells, not individual ops).

Ported from the legacy repo's `operations.py`, trimmed to the 8 ops the
handoff lists as the starting search space -- InvertedResidual, MBConv,
OctaveConv, BlurPool, GroupNorm-style convs, etc. are dropped until this
smaller space has passed shape/backward/proxy/memory tests (handoff §13).
"""

from __future__ import annotations

import torch
import torch.nn as nn

PRIMITIVES: tuple[str, ...] = (
    "none",
    "skip_connect",
    "avg_pool_3x3",
    "max_pool_3x3",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
)


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Zero(nn.Module):
    """Outputs all-zero, at the target stride/spatial resolution."""

    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class ReLUConvBN(nn.Module):
    """1x1 conv used to align channels/identity-preprocess a cell input state."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int, padding: int, affine: bool = True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class FactorizedReduce(nn.Module):
    """Stride-2 channel-preserving (or growing) downsample used for skip_connect
    at stride 2, and to preprocess a cell input coming from a reduction cell."""

    def __init__(self, c_in: int, c_out: int, affine: bool = True):
        super().__init__()
        assert c_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        # Slicing [:, :, 1:, 1:] before the second stride-2 conv gives it a
        # different phase, so the two halves cover the full input instead of
        # sampling the same offset twice.
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)


class SepConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int, padding: int, affine: bool = True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size, stride=stride, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size, stride=1, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class DilConv(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        affine: bool = True,
    ):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=c_in,
                bias=False,
            ),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


OPS: dict[str, callable] = {
    "none": lambda c, stride, affine: Zero(stride),
    "skip_connect": lambda c, stride, affine: (
        Identity() if stride == 1 else FactorizedReduce(c, c, affine=affine)
    ),
    "avg_pool_3x3": lambda c, stride, affine: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False
    ),
    "max_pool_3x3": lambda c, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    "sep_conv_3x3": lambda c, stride, affine: SepConv(c, c, 3, stride, 1, affine=affine),
    "sep_conv_5x5": lambda c, stride, affine: SepConv(c, c, 5, stride, 2, affine=affine),
    "dil_conv_3x3": lambda c, stride, affine: DilConv(c, c, 3, stride, 2, 2, affine=affine),
    "dil_conv_5x5": lambda c, stride, affine: DilConv(c, c, 5, stride, 4, 2, affine=affine),
}


def drop_path(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """Stochastic depth on a residual branch. Device-agnostic (the legacy
    version hardcoded `torch.cuda.FloatTensor`, which crashes off-CUDA)."""
    if drop_prob <= 0.0:
        return x
    keep_prob = 1.0 - drop_prob
    mask = torch.empty(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
    return x.div(keep_prob).mul(mask)
