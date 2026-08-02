from __future__ import annotations

from pathlib import Path

import torch

from brainmri_nas.model.network import build_model
from brainmri_nas.search_space.chromosome import chromosome_length, decode_chromosome
from brainmri_nas.training.checkpoint import (
    cpu_state_dict,
    load_checkpoint,
    rebuild_model_from_checkpoint,
    save_checkpoint,
)

INPUT_CHANNELS, IMAGE_SIZE, NUM_CLASSES = 3, 16, 4
MODEL_CONFIG = dict(
    input_channels=INPUT_CHANNELS,
    num_classes=NUM_CLASSES,
    image_size=IMAGE_SIZE,
    initial_channels=4,
    number_of_cells=3,
    drop_path_probability=0.0,
    stem_type="cifar",
)


def _genotype():
    length = chromosome_length()
    chromosome = [((i * 41) % 89) / 89.0 for i in range(length)]
    return chromosome, decode_chromosome(chromosome)


def test_cpu_state_dict_is_an_independent_snapshot():
    _, genotype = _genotype()
    model = build_model(genotype, **MODEL_CONFIG)

    snapshot = cpu_state_dict(model)
    snapshot_before = {k: v.clone() for k, v in snapshot.items()}

    # Mutate the live model in place -- the snapshot must not change,
    # unlike the `best_model = model` bug this guards against.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    for name, value in snapshot.items():
        assert torch.equal(value, snapshot_before[name])
        assert value.device.type == "cpu"


def test_save_load_checkpoint_round_trip(tmp_path: Path):
    chromosome, genotype = _genotype()
    model = build_model(genotype, **MODEL_CONFIG)
    path = tmp_path / "best_checkpoint.pt"

    save_checkpoint(
        path,
        model=model,
        genotype=genotype,
        model_config=MODEL_CONFIG,
        class_to_idx={"a": 0, "b": 1, "c": 2, "d": 3},
        image_size=IMAGE_SIZE,
        epoch=7,
        validation_metrics={"macro_auc": 0.83},
        chromosome=chromosome,
        augmentation_policy=None,
    )

    payload = load_checkpoint(path)
    assert payload["epoch"] == 7
    assert payload["image_size"] == IMAGE_SIZE
    assert payload["class_to_idx"] == {"a": 0, "b": 1, "c": 2, "d": 3}
    assert payload["validation_metrics"] == {"macro_auc": 0.83}
    assert payload["chromosome"] == chromosome
    assert payload["genotype"] == genotype.to_dict()
    assert payload["model_config"] == MODEL_CONFIG
    assert set(payload["model_state"].keys()) == set(model.state_dict().keys())


def test_rebuild_model_from_checkpoint_matches_saved_weights(tmp_path: Path):
    _, genotype = _genotype()
    model = build_model(genotype, **MODEL_CONFIG)
    path = tmp_path / "best_checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        genotype=genotype,
        model_config=MODEL_CONFIG,
        class_to_idx={},
        image_size=IMAGE_SIZE,
        epoch=1,
        validation_metrics={},
    )

    payload = load_checkpoint(path)
    rebuilt = rebuild_model_from_checkpoint(payload)

    for name, value in model.state_dict().items():
        assert torch.equal(rebuilt.state_dict()[name], value)

    logits = rebuilt(torch.randn(2, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    assert logits.shape == (2, NUM_CLASSES)


def test_inference_after_reload_matches_original_model(tmp_path: Path):
    _, genotype = _genotype()
    model = build_model(genotype, **MODEL_CONFIG)
    model.eval()

    fixed_input = torch.randn(3, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        original_logits = model(fixed_input)

    path = tmp_path / "best_checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        genotype=genotype,
        model_config=MODEL_CONFIG,
        class_to_idx={},
        image_size=IMAGE_SIZE,
        epoch=1,
        validation_metrics={},
    )

    reloaded = rebuild_model_from_checkpoint(load_checkpoint(path))
    reloaded.eval()
    with torch.no_grad():
        reloaded_logits = reloaded(fixed_input)

    assert torch.allclose(original_logits, reloaded_logits, atol=1e-6)
