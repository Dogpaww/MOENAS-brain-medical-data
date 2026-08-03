from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from brainmri_nas.augment.genotype import AugmentationPolicy
from brainmri_nas.augment.policy_search import run_augmentation_search
from brainmri_nas.augment.search_space import AUGMENTATION_OPS, chromosome_length, decode_chromosome
from brainmri_nas.augment.transform_builder import build_augmented_train_transform
from brainmri_nas.data.loader import build_dataset_bundle
from brainmri_nas.model.network import build_model
from brainmri_nas.search.nsga2_runner import run_search
from brainmri_nas.search_space.chromosome import decode_chromosome as decode_arch_chromosome
from brainmri_nas.utils.config import Config, AugmentationConfig, DatasetConfig, NSGA2Config, ProxyConfig, SearchSpaceConfig

N_VAR = chromosome_length()


def _sample_chromosome(offset=0):
    return [((i * 53 + offset) % 101) / 101.0 for i in range(N_VAR)]


def test_decode_is_deterministic():
    chromosome = _sample_chromosome()
    policy_a = decode_chromosome(chromosome)
    policy_b = decode_chromosome(chromosome)
    assert policy_a == policy_b


def test_policy_json_round_trip():
    policy = decode_chromosome(_sample_chromosome())
    reloaded = AugmentationPolicy.from_dict(policy.to_dict())
    assert reloaded == policy
    # And it really is plain JSON-compatible data, not transform objects.
    import json

    json.dumps(policy.to_dict())


def test_policy_steps_follow_canonical_order():
    policy = decode_chromosome(_sample_chromosome())
    ordered_names = tuple(s.name for s in policy.ordered_steps())
    assert ordered_names == AUGMENTATION_OPS
    orders = [s["order"] for s in policy.to_dict()]
    assert orders == sorted(orders)


def test_every_step_stores_required_fields():
    policy = decode_chromosome(_sample_chromosome())
    for step_dict in policy.to_dict():
        assert set(step_dict) == {"name", "order", "probability", "magnitude", "parameters"}
        assert 0.0 <= step_dict["probability"] < 1.0


def test_transform_pipeline_ordering_matches_handoff_spec():
    # All-probability-1 policy so every op is guaranteed to be present/applied.
    chromosome = [0.999999 for _ in range(N_VAR)]
    policy = decode_chromosome(chromosome)
    pipeline = build_augmented_train_transform(policy, image_size=32)

    kinds = [type(t).__name__ for t in pipeline.transforms]
    assert kinds[0] == "ResizeLongerSideAndPad"
    assert "Grayscale" in kinds
    to_tensor_idx = kinds.index("ToTensor")
    normalize_idx = kinds.index("PerImageNormalize")
    erasing_idx = kinds.index("RandomErasing")

    assert normalize_idx == to_tensor_idx + 1  # tensor conversion -> normalization, adjacent
    assert erasing_idx > normalize_idx  # tensor-space augmentation comes after normalization
    # Every PIL-space op (RandomApply-wrapped ops + the flip) must precede ToTensor.
    pil_positions = [i for i, name in enumerate(kinds) if name in ("RandomApply", "RandomHorizontalFlip")]
    assert len(pil_positions) == 6  # rotation, affine_translation, crop, brightness, contrast, flip
    assert all(pos < to_tensor_idx for pos in pil_positions)


def test_pipeline_runs_end_to_end_on_a_real_image(tmp_path: Path):
    img = Image.new("RGB", (40, 40), color=(120, 60, 200))
    policy = decode_chromosome(_sample_chromosome())
    pipeline = build_augmented_train_transform(policy, image_size=32)
    tensor = pipeline(img)
    assert tensor.shape == torch.Size([3, 32, 32])


def test_val_transform_is_deterministic_across_repeated_reads(synthetic_dataset_root: Path):
    bundle = build_dataset_bundle(
        synthetic_dataset_root, image_size=32, validation_fraction=0.2, split_seed=1, batch_size=4
    )
    batch_a_x, batch_a_y = next(iter(bundle.val_loader))
    batch_b_x, batch_b_y = next(iter(bundle.val_loader))
    assert torch.equal(batch_a_x, batch_b_x)
    assert torch.equal(batch_a_y, batch_b_y)


def test_two_policy_trial_models_start_from_identical_weights():
    genotype = decode_arch_chromosome(_arch_chromosome())
    build_kwargs = dict(
        input_channels=3,
        num_classes=4,
        image_size=16,
        initial_channels=4,
        number_of_cells=3,
        drop_path_probability=0.0,
        stem_type="cifar",
    )
    base_model = build_model(genotype, **build_kwargs)
    initial_state = {name: value.detach().clone() for name, value in base_model.state_dict().items()}

    model_a = build_model(genotype, **build_kwargs)
    model_a.load_state_dict(initial_state)
    model_b = build_model(genotype, **build_kwargs)
    model_b.load_state_dict(initial_state)

    for name in initial_state:
        assert torch.equal(model_a.state_dict()[name], initial_state[name])
        assert torch.equal(model_b.state_dict()[name], initial_state[name])
        assert torch.equal(model_a.state_dict()[name], model_b.state_dict()[name])


def _arch_chromosome(offset=0):
    from brainmri_nas.search_space.chromosome import chromosome_length as arch_chromosome_length

    length = arch_chromosome_length()
    return [((i * 37 + offset) % 97) / 97.0 for i in range(length)]


def test_train_trial_model_emits_per_epoch_progress_and_val_auc(caplog):
    import logging

    from torch.utils.data import DataLoader, TensorDataset

    from brainmri_nas.augment.trial_training import train_trial_model

    caplog.set_level(logging.INFO, logger="brainmri_nas.augmentation_search")

    genotype = decode_arch_chromosome(_arch_chromosome())
    model = build_model(
        genotype,
        input_channels=3,
        num_classes=4,
        image_size=16,
        initial_channels=4,
        number_of_cells=3,
        drop_path_probability=0.0,
        stem_type="cifar",
    )

    g = torch.Generator().manual_seed(0)
    x_train = torch.randn(8, 3, 16, 16, generator=g)
    y_train = torch.randint(0, 4, (8,), generator=g)
    x_val = torch.randn(4, 3, 16, 16, generator=g)
    y_val = torch.randint(0, 4, (4,), generator=g)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=4)

    result = train_trial_model(
        model,
        train_loader,
        val_loader,
        epochs=2,
        learning_rate=0.025,
        weight_decay=3e-4,
        device=torch.device("cpu"),
        num_classes=4,
    )

    messages = [r.getMessage() for r in caplog.records]
    assert any("trial epoch 1/2" in m and "val_macro_auc" in m for m in messages)
    assert any("trial epoch 2/2" in m and "val_macro_auc" in m for m in messages)
    assert "val_macro_auc" in result


def test_end_to_end_tiny_augmentation_search(synthetic_dataset_root: Path, tmp_path: Path):
    search_config = Config(
        dataset=DatasetConfig(data_root=str(synthetic_dataset_root), image_size=16, batch_size=4, num_workers=0),
        search_space=SearchSpaceConfig(initial_channels=4, number_of_cells=3),
        proxies=ProxyConfig(zico_batch_size=2, zico_num_batches=2),
        nsga2=NSGA2Config(population_size=4, num_generations=1, seed=1, device="cpu"),
    )
    search_output_dir = tmp_path / "search_run"
    search_result = run_search(search_config, search_output_dir)
    assert search_result["selected_architecture"]["valid"] is True

    augmentation_config = Config(
        dataset=search_config.dataset,
        search_space=search_config.search_space,
        augmentation=AugmentationConfig(
            population_size=2, num_generations=1, trial_epochs=1, seed=1, device="cpu"
        ),
    )
    augmentation_output_dir = tmp_path / "augmentation_run"
    result = run_augmentation_search(
        augmentation_config,
        selected_architecture_path=search_output_dir / "selected_architecture.json",
        split_indices_path=search_output_dir / "split_indices.json",
        output_dir=augmentation_output_dir,
    )

    for filename in (
        "augmentation_config.yaml",
        "policy_archive.json",
        "selected_policy.json",
        "augmentation_search.log",
        "run_manifest.json",
    ):
        assert (augmentation_output_dir / filename).exists(), f"missing output file: {filename}"

    assert len(result["archive"]) >= augmentation_config.augmentation.population_size

    selected_policy = result["selected_policy"]
    assert 0.0 <= selected_policy["val_macro_auc"] <= 1.0
    assert len(selected_policy["policy"]) == len(AUGMENTATION_OPS)

    # The saved policy is genuinely reconstructible into a real transform pipeline.
    policy = AugmentationPolicy.from_dict(selected_policy["policy"])
    pipeline = build_augmented_train_transform(policy, image_size=16)
    img = Image.new("RGB", (20, 20), color=(10, 10, 10))
    tensor = pipeline(img)
    assert tensor.shape == torch.Size([3, 16, 16])
