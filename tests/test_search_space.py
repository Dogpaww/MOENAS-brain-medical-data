from __future__ import annotations

import pytest
import torch

from brainmri_nas.model.cell import Cell
from brainmri_nas.search_space.attentions import ATTN, ATTENTION_PRIMITIVES
from brainmri_nas.search_space.chromosome import chromosome_length, decode_chromosome
from brainmri_nas.search_space.operations import OPS, PRIMITIVES

BATCH, CHANNELS, HEIGHT, WIDTH = 2, 8, 16, 16


@pytest.mark.parametrize("op_name", PRIMITIVES)
def test_op_preserves_shape_at_stride_1(op_name):
    op = OPS[op_name](CHANNELS, 1, True)
    x = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    out = op(x)
    assert out.shape == (BATCH, CHANNELS, HEIGHT, WIDTH)


@pytest.mark.parametrize("op_name", PRIMITIVES)
def test_op_halves_spatial_dims_at_stride_2(op_name):
    op = OPS[op_name](CHANNELS, 2, True)
    x = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    out = op(x)
    assert out.shape == (BATCH, CHANNELS, HEIGHT // 2, WIDTH // 2)


@pytest.mark.parametrize("attn_name", ATTENTION_PRIMITIVES)
def test_attention_preserves_shape(attn_name):
    attn = ATTN[attn_name](CHANNELS)
    x = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    out = attn(x)
    assert out.shape == x.shape


def test_branch_addition_requires_matching_shape():
    a = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    b = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    assert (a + b).shape == a.shape
    with pytest.raises(RuntimeError):
        _ = a + torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH // 2)


def _sample_genotype():
    # A fixed, non-degenerate chromosome exercising a mix of ops/attentions/sources.
    length = chromosome_length()
    values = [((i * 37) % 97) / 97.0 for i in range(length)]
    return decode_chromosome(values)


def test_normal_cell_forward_and_backward():
    genotype = _sample_genotype()
    cell = Cell(genotype.normal, c_prev_prev=24, c_prev=24, c=8, reduction=False, reduction_prev=False)

    s0 = torch.randn(BATCH, 24, HEIGHT, WIDTH, requires_grad=True)
    s1 = torch.randn(BATCH, 24, HEIGHT, WIDTH, requires_grad=True)
    out = cell(s0, s1)

    assert out.shape == (BATCH, 8 * cell.multiplier, HEIGHT, WIDTH)
    out.sum().backward()
    assert s0.grad is not None and s1.grad is not None


def test_reduction_cell_halves_spatial_dims():
    genotype = _sample_genotype()
    cell = Cell(genotype.reduction, c_prev_prev=24, c_prev=24, c=16, reduction=True, reduction_prev=False)

    s0 = torch.randn(BATCH, 24, HEIGHT, WIDTH, requires_grad=True)
    s1 = torch.randn(BATCH, 24, HEIGHT, WIDTH, requires_grad=True)
    out = cell(s0, s1)

    assert out.shape == (BATCH, 16 * cell.multiplier, HEIGHT // 2, WIDTH // 2)
    out.sum().backward()
    assert s0.grad is not None and s1.grad is not None


def test_cell_with_reduction_prev_aligns_mismatched_spatial_inputs():
    genotype = _sample_genotype()
    # s0 comes from two cells back (pre-reduction resolution); s1 is the
    # immediately preceding (reduction) cell's output, already downsampled.
    cell = Cell(genotype.normal, c_prev_prev=24, c_prev=32, c=16, reduction=False, reduction_prev=True)

    s0 = torch.randn(BATCH, 24, HEIGHT, WIDTH, requires_grad=True)
    s1 = torch.randn(BATCH, 32, HEIGHT // 2, WIDTH // 2, requires_grad=True)
    out = cell(s0, s1)

    assert out.shape == (BATCH, 16 * cell.multiplier, HEIGHT // 2, WIDTH // 2)
    out.sum().backward()
    assert s0.grad is not None and s1.grad is not None
