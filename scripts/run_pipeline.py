#!/usr/bin/env python
"""Run the full pipeline in one command: architecture search -> augmentation
policy search -> final training.

Equivalent to running run_search.py, run_augmentation_search.py, and
run_final_training.py in sequence, with the output directories already
wired together correctly.

Each stage is skipped if its output already exists (e.g. re-running after
final training crashed partway through won't redo the architecture or
augmentation search) -- pass --force to rerun everything from scratch.

Usage:
    python -u scripts/run_pipeline.py --config configs/search.yaml 2>&1 | tee run_pipeline.log
    python -u scripts/run_pipeline.py --skip-augmentation   # train with no augmentation policy
    python -u scripts/run_pipeline.py --force                # ignore existing outputs, rerun all stages
"""

from __future__ import annotations

import argparse
from pathlib import Path

from brainmri_nas.augment.policy_search import run_augmentation_search
from brainmri_nas.search.nsga2_runner import run_search
from brainmri_nas.training.final_training import run_final_training
from brainmri_nas.utils.config import load_config
from brainmri_nas.utils.serialization import load_json


def _banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/search.yaml")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Parent directory; stage outputs land in <output-dir>/{search_run,augmentation_run,training_run}.",
    )
    parser.add_argument(
        "--skip-augmentation",
        action="store_true",
        help="Skip augmentation policy search; train with the identity transform only.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun every stage even if its output already exists.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    search_dir = output_dir / "search_run"
    augmentation_dir = output_dir / "augmentation_run"
    training_dir = output_dir / "training_run"

    # -- Stage 1: architecture search -----------------------------------------
    selected_architecture_path = search_dir / "selected_architecture.json"
    if args.force or not selected_architecture_path.exists():
        _banner("STAGE 1/3: NSGA-II architecture search")
        result = run_search(config, search_dir)
        print(f"Selected architecture: {result['selected_architecture']['candidate_hash']}")
    else:
        print(f"[skip] architecture search already done -> {selected_architecture_path}")

    # -- Stage 2: augmentation policy search -----------------------------------
    selected_policy_path = None
    if args.skip_augmentation:
        print("[skip] augmentation search disabled (--skip-augmentation); training with no augmentation policy.")
    else:
        selected_policy_path = augmentation_dir / "selected_policy.json"
        if args.force or not selected_policy_path.exists():
            _banner("STAGE 2/3: augmentation policy search")
            result = run_augmentation_search(
                config,
                selected_architecture_path=selected_architecture_path,
                split_indices_path=search_dir / "split_indices.json",
                output_dir=augmentation_dir,
            )
            print(f"Selected policy val_macro_auc={result['selected_policy']['val_macro_auc']:.4f}")
        else:
            print(f"[skip] augmentation search already done -> {selected_policy_path}")

    # -- Stage 3: final training ------------------------------------------------
    test_metrics_path = training_dir / "test_metrics.json"
    if args.force or not test_metrics_path.exists():
        _banner("STAGE 3/3: final training")
        result = run_final_training(
            config,
            selected_architecture_path=selected_architecture_path,
            split_indices_path=search_dir / "split_indices.json",
            selected_policy_path=selected_policy_path,
            output_dir=training_dir,
        )
        test_metrics = result["test_metrics"]
        print(f"Best epoch: {result['best_epoch']}")
    else:
        print(f"[skip] final training already done -> {test_metrics_path}")
        test_metrics = load_json(test_metrics_path)

    _banner("DONE")
    print(f"Test accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1:  {test_metrics['macro_f1']:.4f}")
    print(f"Test macro AUC: {test_metrics['macro_auc']:.4f}")
    print(f"Full results in: {output_dir}")


if __name__ == "__main__":
    main()
