from __future__ import annotations

import copy
import math

import torch

from brainmri_nas.model.network import build_model
from brainmri_nas.proxies.profiling import profile_flops_and_params
from brainmri_nas.proxies.synflow import compute_synflow
from brainmri_nas.proxies.zico import compute_zico
from brainmri_nas.search_space.chromosome import chromosome_length, decode_chromosome

INPUT_CHANNELS, IMAGE_SIZE, NUM_CLASSES = 3, 16, 4


def _tiny_model():
    length = chromosome_length()
    chromosome = [((i * 41) % 89) / 89.0 for i in range(length)]
    genotype = decode_chromosome(chromosome)
    return build_model(
        genotype,
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        initial_channels=4,
        number_of_cells=3,
        drop_path_probability=0.0,
        stem_type="cifar",
    )


def _fixed_batches(num_batches=4, batch_size=4):
    batches = []
    g = torch.Generator().manual_seed(0)
    for _ in range(num_batches):
        x = torch.randn(batch_size, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, generator=g)
        y = torch.randint(0, NUM_CLASSES, (batch_size,), generator=g)
        batches.append((x, y))
    return batches


def test_synflow_is_finite_and_nonnegative():
    model = _tiny_model()
    result = compute_synflow(model, input_shape=(INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    assert math.isfinite(result["synflow"])
    assert math.isfinite(result["log_synflow"])
    assert result["synflow"] >= 0.0


def test_synflow_restores_parameter_signs_and_clears_gradients():
    model = _tiny_model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    compute_synflow(model, input_shape=(INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))

    for name, p in model.named_parameters():
        assert torch.allclose(p.detach(), before[name], atol=1e-6), f"{name} was not restored"
        assert p.grad is None or torch.all(p.grad == 0)


def test_synflow_restores_original_device_and_dtype():
    model = _tiny_model()
    model.to(torch.float32)
    compute_synflow(model, input_shape=(INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    for p in model.parameters():
        assert p.dtype == torch.float32
        assert p.device.type == "cpu"


def test_synflow_is_stable_across_repeated_evaluation():
    model = _tiny_model()
    result_a = compute_synflow(model, input_shape=(INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    result_b = compute_synflow(model, input_shape=(INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    assert result_a["synflow"] == result_b["synflow"]
    assert result_a["log_synflow"] == result_b["log_synflow"]


def test_zico_is_finite():
    model = _tiny_model()
    result = compute_zico(model, batches=_fixed_batches(), device=torch.device("cpu"))
    assert math.isfinite(result["zico"])


def test_zico_clears_gradients_after_evaluation():
    model = _tiny_model()
    compute_zico(model, batches=_fixed_batches(), device=torch.device("cpu"))
    for p in model.parameters():
        assert p.grad is None or torch.all(p.grad == 0)


def test_zico_deterministic_given_same_model_and_batches():
    model = _tiny_model()
    model_copy = copy.deepcopy(model)
    batches = _fixed_batches()

    result_a = compute_zico(model, batches=batches, device=torch.device("cpu"))
    result_b = compute_zico(model_copy, batches=batches, device=torch.device("cpu"))
    assert result_a["zico"] == result_b["zico"]


def test_flops_and_params_profiled_at_configured_resolution():
    model = _tiny_model()
    result = profile_flops_and_params(
        model, input_shape=(INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=torch.device("cpu")
    )
    assert result["input_shape"] == [1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE]
    assert result["flops"] > 0
    assert result["params"] > 0
    assert result["flops_billion"] == result["flops"] / 1e9
    assert result["params_million"] == result["params"] / 1e6
