"""Pre-registered convergence check for deciding whether a model's epoch
budget was long enough.

Why this exists: baselines share one fixed budget, and a slow-converging
model can be shortchanged by it. Extending one model's training is fair in
principle, but only when a written rule decides it -- applied to every run,
using validation data only, fixed before the results it will be applied to.
Eyeballing curves and extending whichever model looks unfinished is the
unfair version.

The rule, for a run of E epochs:
  - compare the mean over the last 20% of epochs with the mean over the
    20% before that (50 epochs: 41-50 against 31-40);
  - flag the run if mean validation accuracy rose by more than
    ACCURACY_THRESHOLD_POINTS, or mean validation macro-AUC by more than
    AUC_THRESHOLD_POINTS.

A model is flagged when, for either augmentation arm, that rise averaged
over all of its seeds exceeds a threshold. Averaging over seeds matters: a
single run's late wiggle is noise, and flagging a model whenever ANY seed
flags would make extension more likely simply by running more seeds.

A flagged model is retrained with a longer budget in BOTH arms, so a longer
schedule is never mixed up with the effect of augmentation. With cosine
annealing the rise in the final low-learning-rate epochs is partly the
schedule itself, which is why the thresholds are not set at zero.

The thresholds were chosen after seeing the seed-1 curves, and are fixed
from here on for every later seed and run.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

WINDOW_FRACTION = 0.2
ACCURACY_THRESHOLD_POINTS = 0.5
AUC_THRESHOLD_POINTS = 0.2


def late_improvement(history_rows: list[dict]) -> tuple[float, float]:
    """(accuracy rise, macro-AUC rise) in percentage points, last window
    minus the window before it, from `training_history.csv` rows."""
    num_epochs = len(history_rows)
    window = max(1, math.floor(num_epochs * WINDOW_FRACTION))
    if num_epochs < 2 * window:
        raise ValueError(f"Need at least {2 * window} epochs of history, got {num_epochs}.")

    def mean(key: str, start: int, stop: int) -> float:
        values = [float(row[key]) for row in history_rows[start:stop]]
        return sum(values) / len(values)

    last_start = num_epochs - window
    prev_start = last_start - window
    accuracy_rise = 100 * (mean("val_accuracy", last_start, num_epochs) - mean("val_accuracy", prev_start, last_start))
    auc_rise = 100 * (mean("val_macro_auc", last_start, num_epochs) - mean("val_macro_auc", prev_start, last_start))
    return accuracy_rise, auc_rise


def is_flagged(accuracy_rise: float, auc_rise: float) -> bool:
    return accuracy_rise > ACCURACY_THRESHOLD_POINTS or auc_rise > AUC_THRESHOLD_POINTS


def describe_run(run_dir: str | Path) -> dict:
    """Identify a baseline run from its own saved config, never its folder name."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "baseline_config.json").read_text())
    with open(run_dir / "training_history.csv") as f:
        rows = list(csv.DictReader(f))
    accuracy_rise, auc_rise = late_improvement(rows)
    return {
        "run_dir": str(run_dir),
        "model_name": config["model_name"],
        # Runs from before these fields existed were SGD with no augmentation.
        "optimizer_name": config.get("optimizer_name", "sgd"),
        "augmentation": config.get("augmentation", "none"),
        "epochs": config["epochs"],
        "image_size": config["image_size"],
        "seed": config.get("seed"),
        "accuracy_rise": accuracy_rise,
        "auc_rise": auc_rise,
        "flagged": is_flagged(accuracy_rise, auc_rise),
    }


def flag_models(runs: list[dict]) -> list[dict]:
    """Group runs into one model setting (model, optimizer, epochs, image
    size) per augmentation arm, average each arm's rises over its seeds, and
    flag the setting if any arm's averages exceed a threshold."""
    arms: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        key = (run["model_name"], run["optimizer_name"], run["epochs"], run["image_size"])
        arms[key][run["augmentation"]].append(run)

    results = []
    for (model_name, optimizer_name, epochs, image_size), by_arm in sorted(arms.items()):
        arm_summaries = {}
        for arm, arm_runs in sorted(by_arm.items()):
            # A repeat of the same seed (e.g. a rerun of an old run on a new
            # machine) is not an extra seed; averaging it in would overweight
            # that seed. Refuse rather than pick one silently.
            by_seed = defaultdict(list)
            for r in arm_runs:
                by_seed[r["seed"]].append(r["run_dir"])
            duplicates = {seed: dirs for seed, dirs in by_seed.items() if len(dirs) > 1}
            if duplicates:
                raise ValueError(
                    f"{model_name} ({optimizer_name}, {epochs} ep, {image_size}px), arm {arm!r}: more than one run "
                    f"for the same seed {duplicates}. Narrow the run selection so each seed appears once."
                )
            accuracy_rise = sum(r["accuracy_rise"] for r in arm_runs) / len(arm_runs)
            auc_rise = sum(r["auc_rise"] for r in arm_runs) / len(arm_runs)
            arm_summaries[arm] = {
                "num_seeds": len(arm_runs),
                "accuracy_rise": accuracy_rise,
                "auc_rise": auc_rise,
                "flagged": is_flagged(accuracy_rise, auc_rise),
            }
        results.append(
            {
                "model_name": model_name,
                "optimizer_name": optimizer_name,
                "epochs": epochs,
                "image_size": image_size,
                "arms": arm_summaries,
                "flagged": any(s["flagged"] for s in arm_summaries.values()),
            }
        )
    return results
