#!/usr/bin/env python
"""Train the selected architecture from scratch with the selected augmentation policy.

Requires a prior `run_search.py` output directory. Augmentation is one of:
  - `--augmentation-output-dir`: sample-adaptive policy (selected_policy.json)
  - `--legacy-augmentation-output-dir`: fixed 2-op policy (selected_legacy_policy.json)
  - neither: no augmentation (identity transform only)

`--seed` overrides `training.seed` from the config, so repeated runs differ
only in the seed. The seed used is recorded in the output's training_config.yaml.

Usage:
    python scripts/run_final_training.py \
        --config configs/search.yaml \
        --search-output-dir outputs/search_run \
        --augmentation-output-dir outputs/augmentation_run \
        --output-dir outputs/training_run

    # fixed policy, seed 2
    python scripts/run_final_training.py --config configs/figshare.yaml \
        --search-output-dir outputs/main1/search_run \
        --legacy-augmentation-output-dir outputs/fixedda_run/augmentation_run \
        --seed 2 --output-dir outputs/zstar_fixed_s2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from brainmri_nas.training.final_training import run_final_training
from brainmri_nas.utils.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/search.yaml")
    parser.add_argument("--search-output-dir", default="outputs/search_run")
    augmentation = parser.add_mutually_exclusive_group()
    augmentation.add_argument(
        "--augmentation-output-dir",
        default=None,
        help="Directory containing selected_policy.json (sample-adaptive). Omit both augmentation flags to train "
        "without augmentation.",
    )
    augmentation.add_argument(
        "--legacy-augmentation-output-dir",
        default=None,
        help="Directory containing selected_legacy_policy.json (fixed 2-op policy).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Overrides training.seed from the config.")
    parser.add_argument("--output-dir", default="outputs/training_run")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.seed is not None:
        config.training.seed = args.seed
    search_output_dir = Path(args.search_output_dir)

    selected_policy_path = None
    if args.augmentation_output_dir:
        selected_policy_path = Path(args.augmentation_output_dir) / "selected_policy.json"
    selected_legacy_policy_path = None
    if args.legacy_augmentation_output_dir:
        selected_legacy_policy_path = Path(args.legacy_augmentation_output_dir) / "selected_legacy_policy.json"
    for policy_path in (selected_policy_path, selected_legacy_policy_path):
        if policy_path is not None and not policy_path.exists():
            raise SystemExit(f"{policy_path} does not exist.")

    result = run_final_training(
        config,
        selected_architecture_path=search_output_dir / "selected_architecture.json",
        split_indices_path=search_output_dir / "split_indices.json",
        selected_policy_path=selected_policy_path,
        selected_legacy_policy_path=selected_legacy_policy_path,
        output_dir=args.output_dir,
    )
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Test accuracy: {result['test_metrics']['accuracy']:.4f}")
    print(f"Test macro F1: {result['test_metrics']['macro_f1']:.4f}")
    print(f"Test macro AUC: {result['test_metrics']['macro_auc']:.4f}")


if __name__ == "__main__":
    main()
