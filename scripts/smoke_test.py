#!/usr/bin/env python
"""End-to-end smoke test (handoff §32).

Exercises every stage of the pipeline on a small stratified data subset:
dataset validation, chromosome decode, forward/backward, SynFlow, ZiCO,
FLOPs/param profiling, a tiny NSGA-II search, architecture selection, one
augmentation policy trial, one epoch of final training, checkpoint
save/reload, and validation inference. Runs on CPU (or MPS/CUDA if
available); reports peak VRAM only when CUDA is present.

If `dataset.data_root` in `--config` doesn't point at a valid, already-
downloaded dataset, this falls back to a small synthetic dataset generated
in a temp directory -- clearly logged as such, since that is NOT a
real-data validation. Point `--data-root` at the extracted Kaggle
brain-tumor-classification-mri folder for a real check.

Usage:
    python -u scripts/smoke_test.py 2>&1 | tee smoke_test.log
    python -u scripts/smoke_test.py --data-root /path/to/brain_tumor_mri
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split

from brainmri_nas.augment.sample_adaptive_dataset import build_sample_adaptive_loader
from brainmri_nas.augment.search_space import chromosome_length as augmentation_chromosome_length
from brainmri_nas.augment.search_space import decode_chromosome as decode_augmentation_chromosome
from brainmri_nas.augment.trial_training import train_trial_model
from brainmri_nas.data.loader import DatasetValidationError, build_dataset_bundle, describe_dataset
from brainmri_nas.data.split import save_split_indices
from brainmri_nas.model.network import build_model
from brainmri_nas.proxies.profiling import profile_flops_and_params
from brainmri_nas.proxies.synflow import compute_synflow
from brainmri_nas.proxies.zico import compute_zico
from brainmri_nas.search.nsga2_runner import run_search
from brainmri_nas.search.proxy_samples import build_fixed_proxy_batches
from brainmri_nas.search_space.chromosome import decode_full_chromosome, total_chromosome_length
from brainmri_nas.search_space.genotype import NetworkGenotype
from brainmri_nas.training.checkpoint import load_checkpoint, rebuild_model_from_checkpoint
from brainmri_nas.training.final_training import run_final_training
from brainmri_nas.utils.config import Config, NSGA2Config, ProxyConfig, TrainingConfig, load_config
from brainmri_nas.utils.determinism import seed_everything
from brainmri_nas.utils.device import resolve_device
from brainmri_nas.utils.loss_cache import LossCache
from brainmri_nas.utils.serialization import dump_json

TOTAL_STEPS = 14
SYNTHETIC_CLASSES = ("glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor")


def _step(n: int, description: str) -> None:
    print(f"\n[{n}/{TOTAL_STEPS}] {description}")


def _make_synthetic_dataset(root: Path, *, train_per_class: int = 24, test_per_class: int = 6) -> None:
    for split, count in (("Training", train_per_class), ("Testing", test_per_class)):
        for class_idx, class_name in enumerate(SYNTHETIC_CLASSES):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                color = ((class_idx * 60) % 255, (i * 20) % 255, 120)
                Image.new("RGB", (48, 48), color=color).save(class_dir / f"{class_name}_{i:03d}.png")


def _stratified_subsample(targets: list[int], indices: tuple[int, ...], max_samples: int, seed: int) -> tuple[int, ...]:
    indices = list(indices)
    if len(indices) <= max_samples:
        return tuple(sorted(indices))
    labels = [targets[i] for i in indices]
    sampled, _ = train_test_split(indices, train_size=max_samples, random_state=seed, stratify=labels)
    return tuple(sorted(sampled))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/search.yaml")
    parser.add_argument("--data-root", default=None, help="Override dataset.data_root from --config.")
    parser.add_argument("--output-dir", default="outputs/smoke_test")
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--max-val-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t_start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    if args.data_root:
        config.dataset.data_root = args.data_root

    synthetic_dirs: list[Path] = []

    try:
        # -- Step 1: validate dataset folders --------------------------------
        _step(1, "Validating dataset folders")
        data_root = Path(config.dataset.data_root)
        try:
            report = describe_dataset(data_root)
            print(f"    found a real dataset at {data_root}")
        except DatasetValidationError as exc:
            print(f"    no valid dataset at {data_root} ({exc})")
            print("    falling back to a small SYNTHETIC dataset -- this is NOT a real-data check.")
            tmp_dir = Path(tempfile.mkdtemp(prefix="brainmri_smoke_"))
            synthetic_dirs.append(tmp_dir)
            data_root = tmp_dir / "brain_tumor_mri"
            _make_synthetic_dataset(data_root)
            config.dataset.data_root = str(data_root)
            report = describe_dataset(data_root)
        for split, counts in report["splits"].items():
            print(f"    {split}: {counts}")

        seed_everything(args.seed)
        device = resolve_device("auto")
        print(f"\nUsing device={device}")

        # -- Step 2: load a small stratified subset --------------------------
        _step(2, "Loading a small stratified subset")
        full_bundle = build_dataset_bundle(
            data_root,
            image_size=config.dataset.image_size,
            validation_fraction=config.dataset.validation_fraction,
            split_seed=config.dataset.split_seed,
            batch_size=config.dataset.batch_size,
            num_workers=0,
        )
        targets = full_bundle.train_loader.dataset.dataset.targets
        smoke_train_indices = _stratified_subsample(targets, full_bundle.train_indices, args.max_train_samples, args.seed)
        smoke_val_indices = _stratified_subsample(targets, full_bundle.val_indices, args.max_val_samples, args.seed)

        # run_search() derives its split path from its own output_dir rather than
        # accepting one as a parameter (unlike run_augmentation_search/
        # run_final_training), so pre-seed it there -- its "reuse if it already
        # exists" logic then picks up this small subset instead of computing its
        # own full-size split, which is the whole point of this step.
        search_output_dir = output_dir / "search_run"
        search_output_dir.mkdir(parents=True, exist_ok=True)
        split_indices_path = search_output_dir / "split_indices.json"
        save_split_indices(smoke_train_indices, smoke_val_indices, split_indices_path)

        batch_size = max(1, min(config.dataset.batch_size, len(smoke_train_indices)))
        bundle = build_dataset_bundle(
            data_root,
            image_size=config.dataset.image_size,
            validation_fraction=config.dataset.validation_fraction,
            split_seed=config.dataset.split_seed,
            batch_size=batch_size,
            num_workers=0,
            split_indices_path=split_indices_path,
        )
        print(f"    {len(bundle.train_indices)} train / {len(bundle.val_indices)} val samples, classes={bundle.classes}")

        input_shape = (config.dataset.input_channels, config.dataset.image_size, config.dataset.image_size)

        # -- Step 3: decode one chromosome ------------------------------------
        _step(3, "Decoding one chromosome")
        n_var = total_chromosome_length(config.search_space.num_intermediate_nodes, config.search_space.edges_per_node)
        demo_chromosome = [((i * 37) % 97) / 97.0 for i in range(n_var)]
        genotype, demo_number_of_cells, demo_initial_channels = decode_full_chromosome(
            demo_chromosome,
            config.search_space.num_intermediate_nodes,
            config.search_space.edges_per_node,
            number_of_cells_range=(config.search_space.number_of_cells_min, config.search_space.number_of_cells_max),
            initial_channels_range=(config.search_space.initial_channels_min, config.search_space.initial_channels_max),
        )
        print(
            f"    normal edges={len(genotype.normal.edges)}, reduction edges={len(genotype.reduction.edges)}, "
            f"number_of_cells={demo_number_of_cells}, initial_channels={demo_initial_channels}"
        )

        # -- Step 4: forward and backward -------------------------------------
        _step(4, "Building the model and running forward + backward")
        model_config = dict(
            input_channels=config.dataset.input_channels,
            num_classes=bundle.num_classes,
            image_size=config.dataset.image_size,
            initial_channels=demo_initial_channels,
            number_of_cells=demo_number_of_cells,
            drop_path_probability=config.search_space.drop_path_probability,
            stem_type=config.search_space.stem_type,
        )
        model = build_model(genotype, **model_config)
        model.to(device)
        dummy_x = torch.randn(2, *input_shape, device=device)
        logits = model(dummy_x)
        logits.sum().backward()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"    logits shape={tuple(logits.shape)}, parameters={num_params:,}")
        assert logits.shape == (2, bundle.num_classes)

        # -- Step 5: SynFlow ----------------------------------------------------
        _step(5, "Calculating SynFlow")
        synflow_result = compute_synflow(model, input_shape=input_shape)
        print(f"    synflow={synflow_result['synflow']:.6g} log_synflow={synflow_result['log_synflow']:.4f}")
        assert math.isfinite(synflow_result["log_synflow"])

        # -- Step 6: ZiCO ---------------------------------------------------------
        _step(6, "Calculating ZiCO")
        zico_batch_size = max(1, min(4, len(bundle.train_indices) // 2))
        proxy_batches = build_fixed_proxy_batches(
            bundle,
            num_batches=2,
            batch_size=zico_batch_size,
            seed=args.seed,
            proxy_indices_path=output_dir / "proxy_sample_indices.json",
        )
        zico_result = compute_zico(model, batches=proxy_batches, device=device)
        print(f"    zico={zico_result['zico']:.6g}")
        assert math.isfinite(zico_result["zico"])

        # -- Step 7: FLOPs and parameters -----------------------------------------
        _step(7, "Profiling FLOPs and parameters")
        profile_result = profile_flops_and_params(model, input_shape=input_shape, device=device)
        print(f"    flops={profile_result['flops_billion']:.4f}B params={profile_result['params_million']:.4f}M")

        del model

        # -- Step 8: NSGA-II, population 4, 2 generations -----------------------
        _step(8, "Running NSGA-II (population=4, generations=2)")
        search_config = Config(
            dataset=config.dataset,
            search_space=config.search_space,
            proxies=ProxyConfig(zico_batch_size=zico_batch_size, zico_num_batches=2, proxy_sample_seed=args.seed),
            nsga2=NSGA2Config(population_size=4, num_generations=2, seed=args.seed, device=str(device)),
        )
        search_result = run_search(search_config, search_output_dir)

        # -- Step 9: select one architecture ---------------------------------------
        _step(9, "Selecting one architecture (TOPSIS)")
        selected_architecture = search_result["selected_architecture"]
        assert selected_architecture["valid"]
        print(
            f"    selected candidate_hash={selected_architecture['candidate_hash'][:12]} "
            f"topsis_score={selected_architecture['topsis_score']:.4f}"
        )

        # -- Step 10: test one augmentation policy -----------------------------------
        _step(10, "Testing one augmentation policy")
        augmentation_n_var = augmentation_chromosome_length(num_classes=bundle.num_classes)
        demo_policy_chromosome = [((i * 53) % 101) / 101.0 for i in range(augmentation_n_var)]
        policy = decode_augmentation_chromosome(demo_policy_chromosome, num_classes=bundle.num_classes)

        selected_genotype = NetworkGenotype.from_dict(selected_architecture["genotype"])
        selected_model_config = dict(
            model_config,
            initial_channels=selected_architecture["initial_channels"],
            number_of_cells=selected_architecture["number_of_cells"],
        )
        trial_model = build_model(selected_genotype, **selected_model_config)
        loss_cache = LossCache(num_samples=len(bundle.train_indices), total_epochs=1)
        policy_train_loader = build_sample_adaptive_loader(
            data_root / "Training",
            bundle.train_indices,
            image_size=config.dataset.image_size,
            batch_size=batch_size,
            policy=policy,
            loss_cache=loss_cache,
            num_workers=0,
        )
        trial_result = train_trial_model(
            trial_model,
            policy_train_loader,
            bundle.val_loader,
            epochs=1,
            learning_rate=0.025,
            weight_decay=3e-4,
            device=device,
            num_classes=bundle.num_classes,
            loss_cache=loss_cache,
        )
        print(f"    one-trial val_macro_auc={trial_result['val_macro_auc']:.4f}")
        del trial_model, policy_train_loader, loss_cache

        selected_policy_path = output_dir / "selected_policy.json"
        dump_json(
            {"chromosome": demo_policy_chromosome, "policy": policy.to_dict(), "val_macro_auc": trial_result["val_macro_auc"]},
            selected_policy_path,
        )

        # -- Step 11: train one epoch (final training loop) ------------------------
        _step(11, "Training one epoch (final training loop)")
        training_config = Config(
            dataset=config.dataset,
            search_space=config.search_space,
            training=TrainingConfig(
                physical_batch_size=batch_size,
                gradient_accumulation_steps=1,
                final_epochs=1,
                precision="amp",
                device=str(device),
                seed=args.seed,
            ),
        )
        training_output_dir = output_dir / "training_run"
        training_result = run_final_training(
            training_config,
            selected_architecture_path=search_output_dir / "selected_architecture.json",
            split_indices_path=split_indices_path,
            selected_policy_path=selected_policy_path,
            output_dir=training_output_dir,
        )
        print(f"    trained 1 epoch, best_epoch={training_result['best_epoch']}")

        # -- Step 12: save and reload a checkpoint --------------------------------------
        _step(12, "Saving and reloading the checkpoint")
        checkpoint_path = training_output_dir / "best_checkpoint.pt"
        payload = load_checkpoint(checkpoint_path)
        reloaded_model = rebuild_model_from_checkpoint(payload)
        reloaded_model.to(device)
        print(f"    reloaded checkpoint from epoch {payload['epoch']}")

        # -- Step 13: run validation inference ------------------------------------------
        _step(13, "Running validation inference with the reloaded model")
        reloaded_model.eval()
        val_x, val_y = next(iter(bundle.val_loader))
        with torch.no_grad():
            val_logits = reloaded_model(val_x.to(device))
        assert val_logits.shape == (val_x.size(0), bundle.num_classes)
        print(f"    inference logits shape={tuple(val_logits.shape)}")

        # -- Step 14: peak VRAM (CUDA only) -----------------------------------------------
        _step(14, "Reporting peak VRAM (CUDA only)")
        if torch.cuda.is_available():
            print(f"    peak VRAM allocated: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
        else:
            print(f"    CUDA not available (ran on {device}) -- skipping.")

        elapsed = time.time() - t_start
        print(f"\nSMOKE TEST PASSED in {elapsed:.1f}s. Outputs written to {output_dir}")

    finally:
        for tmp_dir in synthetic_dirs:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
