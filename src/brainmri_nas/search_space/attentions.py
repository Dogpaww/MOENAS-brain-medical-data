"""Attention modules (handoff §13: initial attention choices).

`ATTN[name](channels)` -> nn.Module, applied after an operation's output
inside a cell edge. Only `identity`, `SE`, `CBAM` are included -- BAM, GE,
DoubleAttention are dropped from the legacy `attentions.py` for the same
reason the exotic conv ops are dropped (start small, add after tests pass).

Neither SE nor CBAM needs spatial dimensions, so unlike the legacy
`ATTNS[name](c, height, width)` signature, these take channel count only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from brainmri_nas.search_space.operations import Identity

ATTENTION_PRIMITIVES: tuple[str, ...] = ("identity", "SE", "CBAM")


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.avg_pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale.expand_as(x)


class _ChannelPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)], dim=1)


class _CBAMChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        return torch.sigmoid(attn).unsqueeze(-1).unsqueeze(-1).expand_as(x)


class _CBAMSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = _ChannelPool()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_compressed = self.compress(x)
        x_out = self.bn(self.conv(x_compressed))
        return torch.sigmoid(x_out)


class ConvolutionalBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = _CBAMChannelAttention(channels, reduction)
        self.spatial_attention = _CBAMSpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


ATTN: dict[str, callable] = {
    "identity": lambda channels: Identity(),
    "SE": lambda channels: SqueezeAndExcitation(channels),
    "CBAM": lambda channels: ConvolutionalBAM(channels),
}
