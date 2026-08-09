from __future__ import annotations

import pytest
import torch

from brainmri_nas.model.genotype_io import load_genotype, save_genotype
from brainmri_nas.model.network import build_model
from brainmri_nas.search_space.chromosome import (
    chromosome_length,
    decode_chromosome,
    validate_and_repair_genotype,
    validate_chromosome,
)
from brainmri_nas.search_space.genotype import CellGenotype, EdgeGene, NetworkGenotype


def _sample_chromosome():
    length = chromosome_length()
    return [((i * 37) % 97) / 97.0 for i in range(length)]


def _build_kwargs(genotype):
    return dict(
        genotype=genotype,
        input_channels=3,
        num_classes=4,
        image_size=32,
        initial_channels=4,
        number_of_cells=3,
        drop_path_probability=0.0,
        stem_type="cifar",
    )


def test_same_chromosome_produces_identical_genotype():
    chromosome = _sample_chromosome()
    genotype_a = decode_chromosome(chromosome)
    genotype_b = decode_chromosome(chromosome)
    assert genotype_a == genotype_b


def test_different_chromosome_can_produce_different_genotype():
    chromosome = _sample_chromosome()
    mutated = list(chromosome)
    mutated[0] = (mutated[0] + 0.5) % 1.0
    assert decode_chromosome(chromosome) != decode_chromosome(mutated)


def test_chromosome_length_validation():
    with pytest.raises(ValueError, match="length"):
        validate_chromosome([0.1, 0.2, 0.3])


def test_chromosome_range_validation():
    bad = _sample_chromosome()
    bad[0] = 1.0  # out of the required [0, 1) range
    with pytest.raises(ValueError, match="range"):
        validate_chromosome(bad)


def test_same_genotype_produces_structurally_identical_model():
    genotype = decode_chromosome(_sample_chromosome())
    model_a = build_model(**_build_kwargs(genotype))
    model_b = build_model(**_build_kwargs(genotype))

    shapes_a = {name: tuple(p.shape) for name, p in model_a.state_dict().items()}
    shapes_b = {name: tuple(p.shape) for name, p in model_b.state_dict().items()}
    assert shapes_a == shapes_b


def test_classifier_dropout_defaults_to_disabled():
    genotype = decode_chromosome(_sample_chromosome())
    model = build_model(**_build_kwargs(genotype))
    assert model.classifier_dropout.p == 0.0


def test_classifier_dropout_is_inactive_in_eval_mode_but_active_in_train_mode():
    genotype = decode_chromosome(_sample_chromosome())
    kwargs = _build_kwargs(genotype)
    kwargs["classifier_dropout_probability"] = 0.9
    model = build_model(**kwargs)
    x = torch.randn(4, 3, 32, 32)

    model.eval()
    with torch.no_grad():
        out_a = model(x)
        out_b = model(x)
    assert torch.equal(out_a, out_b)  # eval mode: dropout disabled, deterministic

    model.train()
    torch.manual_seed(0)
    with torch.no_grad():
        out_c = model(x)
    torch.manual_seed(1)
    with torch.no_grad():
        out_d = model(x)
    assert not torch.equal(out_c, out_d)  # train mode: dropout active, stochastic


def test_invalid_source_is_deterministically_repaired_not_random():
    # Node 0's valid sources are {0, 1} (count=2); source=5 is out of range.
    bad_edge = EdgeGene(source=5, operation="skip_connect", attention="identity")
    other_edges = tuple(
        EdgeGene(source=0, operation="skip_connect", attention="identity") for _ in range(7)
    )
    bad_cell = CellGenotype(edges=(bad_edge,) + other_edges, concat_nodes=(2, 3, 4, 5))
    bad_genotype = NetworkGenotype(normal=bad_cell, reduction=bad_cell)

    repaired_a = validate_and_repair_genotype(bad_genotype)
    repaired_b = validate_and_repair_genotype(bad_genotype)

    assert repaired_a == repaired_b  # deterministic, not resampled
    assert repaired_a.normal.edges[0].source == 5 % 2  # modulo repair, per handoff §10
    assert repaired_a.normal.edges[0].source == 1


def test_repair_rejects_unknown_operation_and_attention():
    bad_edges = tuple(EdgeGene(source=0, operation="skip_connect", attention="identity") for _ in range(8))
    bad_edges = (EdgeGene(source=0, operation="not_a_real_op", attention="identity"),) + bad_edges[1:]
    bad_cell = CellGenotype(edges=bad_edges, concat_nodes=(2, 3, 4, 5))
    bad_genotype = NetworkGenotype(normal=bad_cell, reduction=bad_cell)

    with pytest.raises(ValueError, match="operation"):
        validate_and_repair_genotype(bad_genotype)


def test_genotype_json_round_trip(tmp_path):
    genotype = decode_chromosome(_sample_chromosome())
    path = tmp_path / "genotype.json"
    save_genotype(genotype, path)
    reloaded = load_genotype(path)

    assert reloaded == genotype

    model = build_model(**_build_kwargs(reloaded))
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 4)
