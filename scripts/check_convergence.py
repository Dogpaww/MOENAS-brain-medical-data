#!/usr/bin/env python
"""Apply the pre-registered convergence rule to baseline runs.

Reads each run's training_history.csv and baseline_config.json, reports how
much validation accuracy and macro-AUC rose over the final 20% of epochs,
and flags any model setting whose seed-averaged rise exceeds the thresholds
in `brainmri_nas/training/convergence.py`. A flagged model should be
retrained with a longer budget in both augmentation arms.

Each seed may appear once per model setting and arm. The original
no-augmentation runs (outputs/baseline_<model>_96) were repeated on the
current VM as baseline_<model>_96_none with the same seed, so the default
glob selects only the suffixed directories.

Usage:
    python scripts/check_convergence.py
    python scripts/check_convergence.py --runs "outputs/baseline_*_96_*"
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from brainmri_nas.training.convergence import (
    ACCURACY_THRESHOLD_POINTS,
    AUC_THRESHOLD_POINTS,
    describe_run,
    flag_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="outputs/baseline_*_96_*", help="Glob matching baseline run directories.")
    args = parser.parse_args()

    run_dirs = sorted(
        d for d in glob.glob(args.runs)
        if (Path(d) / "training_history.csv").exists() and (Path(d) / "baseline_config.json").exists()
    )
    if not run_dirs:
        raise SystemExit(f"No completed baseline runs match {args.runs!r}.")

    runs = [describe_run(d) for d in run_dirs]
    print(f"Rule: flag if validation accuracy rises > {ACCURACY_THRESHOLD_POINTS} pts or macro-AUC > "
          f"{AUC_THRESHOLD_POINTS} pts over the final 20% of epochs (seed-averaged per arm).\n")
    print(f"{'run':44s} {'arm':15s} {'seed':>4s} {'ep':>4s} {'Δacc':>7s} {'ΔAUC':>7s}")
    for r in runs:
        print(f"{Path(r['run_dir']).name:44s} {r['augmentation']:15s} {str(r['seed']):>4s} {r['epochs']:>4d} "
              f"{r['accuracy_rise']:+7.2f} {r['auc_rise']:+7.2f}{'  *' if r['flagged'] else ''}")

    print("\nModel settings (seed-averaged):")
    try:
        models = flag_models(runs)
    except ValueError as error:
        raise SystemExit(f"\n{error}")
    for m in models:
        arms = "  ".join(
            f"{arm}: n={s['num_seeds']} Δacc={s['accuracy_rise']:+.2f} ΔAUC={s['auc_rise']:+.2f}{' *' if s['flagged'] else ''}"
            for arm, s in m["arms"].items()
        )
        label = f"{m['model_name']} ({m['optimizer_name']}, {m['epochs']} ep, {m['image_size']}px)"
        print(f"  {'FLAGGED ' if m['flagged'] else '        '}{label:42s} {arms}")


if __name__ == "__main__":
    main()
