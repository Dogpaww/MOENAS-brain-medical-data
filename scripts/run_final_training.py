#!/usr/bin/env python
"""Train the selected architecture from scratch with the selected augmentation policy.

Requires a prior `run_search.py` output directory. `--augmentation-output-dir`
is optional -- omit it to train without augmentation (identity transform only).

Usage:
    python scripts/run_final_training.py \
        --config configs/search.yaml \
        --search-output-dir outputs/search_run \
        --augmentation-output-dir outputs/augmentation_run \
        --output-dir outputs/training_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from brainmri_nas.training.final_training import run_final_training
from brainmri_nas.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/search.yaml")
    parser.add_argument("--search-output-dir", default="outputs/search_run")
    parser.add_argument(
        "--augmentation-output-dir",
        default=None,
        help="Directory containing selected_policy.json. Omit to train without augmentation.",
    )
    parser.add_argument("--output-dir", default="outputs/training_run")
    args = parser.parse_args()

    config = load_config(args.config)
    search_output_dir = Path(args.search_output_dir)

    selected_policy_path = None
    if args.augmentation_output_dir:
        selected_policy_path = Path(args.augmentation_output_dir) / "selected_policy.json"

    result = run_final_training(
        config,
        selected_architecture_path=search_output_dir / "selected_architecture.json",
        split_indices_path=search_output_dir / "split_indices.json",
        selected_policy_path=selected_policy_path,
        output_dir=args.output_dir,
    )
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Test accuracy: {result['test_metrics']['accuracy']:.4f}")
    print(f"Test macro F1: {result['test_metrics']['macro_f1']:.4f}")
    print(f"Test macro AUC: {result['test_metrics']['macro_auc']:.4f}")


if __name__ == "__main__":
    main()
