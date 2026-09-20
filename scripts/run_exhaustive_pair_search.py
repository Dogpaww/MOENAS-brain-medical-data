#!/usr/bin/env python
"""Evaluate every pair of the 7 augmentation operators with the legacy
fixed-policy trial protocol, `--repeats` times each, and select the pair with
the highest mean validation macro-AUC.

Writes selected_legacy_policy.json in the legacy format, so the output
directory can be passed straight to --legacy-augmentation-output-dir
(run_final_training.py) or its selected_legacy_policy.json to
--fixed-augmentation-policy (run_baseline.py). Rerunning with the same
arguments resumes where it stopped.

--repeats is required on purpose: choose it before running, not after.

Usage:
    python -u scripts/run_exhaustive_pair_search.py --config configs/figshare.yaml \
        --search-output-dir outputs/main1/search_run \
        --output-dir outputs/exhaustive_pair_search --repeats 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from brainmri_nas.augment.exhaustive_pair_search import run_exhaustive_pair_search
from brainmri_nas.utils.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/figshare.yaml")
    parser.add_argument("--search-output-dir", required=True, help="Supplies selected_architecture.json and split_indices.json.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, required=True, help="Trials per pair; pairs are ranked by their mean.")
    args = parser.parse_args(argv)

    search_output_dir = Path(args.search_output_dir)
    result = run_exhaustive_pair_search(
        load_config(args.config),
        selected_architecture_path=search_output_dir / "selected_architecture.json",
        split_indices_path=search_output_dir / "split_indices.json",
        output_dir=args.output_dir,
        repeats=args.repeats,
    )
    selected = result["selected_policy"]
    print(f"\nSelected: {'+'.join(selected['ops'])}  mean val macro-AUC {selected['val_macro_auc']:.4f} "
          f"(std {selected['val_macro_auc_std']:.4f}, n={selected['num_evaluations']})")


if __name__ == "__main__":
    main()
