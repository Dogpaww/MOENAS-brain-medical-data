#!/usr/bin/env python
"""Write test_predictions.csv for finished baseline runs from their checkpoints.

For each matching run directory with a best_checkpoint.pt, rebuilds the model,
re-evaluates the test set, and writes the per-image predictions only if the
recomputed confusion matrix and macro-AUC match the saved test_metrics.json.
Runs that already have the CSV are skipped unless --force is given.

Exits with status 1 if any run fails to reproduce its saved metrics.

Usage:
    python scripts/backfill_predictions.py
    python scripts/backfill_predictions.py --runs "outputs/baseline_*_96_*"
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from brainmri_nas.baselines.predictions import backfill_run_predictions
from brainmri_nas.utils.config import load_config
from brainmri_nas.utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="outputs/baseline_*", help="Glob matching baseline run directories.")
    parser.add_argument("--config", default="configs/figshare.yaml", help="Supplies the dataset location and loader settings.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true", help="Recreate test_predictions.csv even if it exists.")
    args = parser.parse_args()

    run_dirs = sorted(d for d in glob.glob(args.runs) if Path(d).is_dir())
    if not run_dirs:
        raise SystemExit(f"No run directories match {args.runs!r}.")

    config = load_config(args.config)
    device = resolve_device(args.device)
    print(f"Device: {device}\n")

    mismatches = 0
    for run_dir in run_dirs:
        result = backfill_run_predictions(run_dir, config, device, force=args.force)
        name = Path(run_dir).name
        status = result["status"]
        if status == "written":
            print(f"WRITTEN   {name:44s} accuracy={result['recomputed_accuracy']:.4f} AUC diff={result['auc_difference']:.1e}")
        elif status == "mismatch":
            mismatches += 1
            print(f"MISMATCH  {name:44s} {result['reason']}")
        else:
            print(f"skipped   {name:44s} {result['reason']}")

    if mismatches:
        raise SystemExit(f"\n{mismatches} run(s) did not reproduce their saved metrics; no CSV written for them.")


if __name__ == "__main__":
    main()
