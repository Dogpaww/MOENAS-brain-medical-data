#!/usr/bin/env python
"""Run augmentation policy search for an already-selected architecture.

Requires a prior `run_search.py` output directory (for selected_architecture.json
and split_indices.json).

Usage:
    python scripts/run_augmentation_search.py \
        --config configs/search.yaml \
        --search-output-dir outputs/search_run \
        --output-dir outputs/augmentation_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from brainmri_nas.augment.policy_search import run_augmentation_search
from brainmri_nas.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/search.yaml")
    parser.add_argument("--search-output-dir", default="outputs/search_run")
    parser.add_argument("--output-dir", default="outputs/augmentation_run")
    args = parser.parse_args()

    config = load_config(args.config)
    search_output_dir = Path(args.search_output_dir)

    result = run_augmentation_search(
        config,
        selected_architecture_path=search_output_dir / "selected_architecture.json",
        split_indices_path=search_output_dir / "split_indices.json",
        output_dir=args.output_dir,
    )
    print(f"Selected policy val_macro_auc={result['selected_policy']['val_macro_auc']:.4f}")


if __name__ == "__main__":
    main()
