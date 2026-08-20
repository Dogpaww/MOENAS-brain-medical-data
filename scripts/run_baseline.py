#!/usr/bin/env python
"""Train an ImageNet-pretrained baseline model on the same figshare corpus,
patient split, and evaluation code as the NAS pipeline, for a controlled
comparison against the searched architecture (see HANDOFF.md's "Baseline
benchmark plan" section).

Requires an existing split_indices.json -- this script never computes a
fresh split, so point it at the searched-architecture run's split (copy it
in, the same trick used for the `main1` controlled-comparison run itself):

    mkdir -p outputs/baseline_resnet18_96
    cp outputs/main1/search_run/split_indices.json outputs/baseline_resnet18_96/

Default resolution is 96x96, matching the NAS pipeline's own image size, so
the comparison isn't confounded by baselines seeing more pixels than the
searched architecture does -- this is the primary number to report. Pass
--image-size 224 (into a *different* --output-dir, e.g. baseline_resnet18_224,
so it doesn't overwrite the 96x96 run) for a secondary, clearly-labeled
"baseline at its native ImageNet resolution" data point.

Usage:
    python -u scripts/run_baseline.py \
        --model resnet18 \
        --config configs/figshare.yaml \
        --split-indices outputs/baseline_resnet18_96/split_indices.json \
        --output-dir outputs/baseline_resnet18_96 \
        2>&1 | tee run_baseline_resnet18_96.log
"""

from __future__ import annotations

import argparse
from pathlib import Path

from brainmri_nas.baselines.registry import MODEL_REGISTRY
from brainmri_nas.baselines.train_baseline import run_baseline_training
from brainmri_nas.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY), help="Baseline model to train.")
    parser.add_argument("--config", default="configs/figshare.yaml")
    parser.add_argument(
        "--split-indices",
        required=True,
        help="Path to an existing split_indices.json to reuse (must already exist).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--image-size",
        type=int,
        default=96,
        help="Resize target; defaults to 96 to match the NAS pipeline's own resolution "
        "(config.dataset.image_size), holding resolution constant so the comparison isn't "
        "confounded by baselines seeing more pixels than the searched architecture does. "
        "Pass 224 explicitly for a secondary run at ResNet's native ImageNet resolution.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Fine-tuning epochs (pretrained weights need far fewer than the 200-epoch from-scratch recipe).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="SGD fine-tuning LR (the pipeline's 0.025 is tuned for from-scratch training).",
    )
    parser.add_argument("--weight-decay", type=float, default=7e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adamw"],
        default="sgd",
        help="Defaults to sgd, matching every other baseline and the NAS pipeline itself "
        "(see utils/optim.py for why). Pass adamw only for transformer baselines (e.g. "
        "deit_small) -- SGD is a documented mismatch for fine-tuning ViTs and produces a "
        "training loss that freezes rather than converges. Always run the sgd version "
        "first; only fall back to adamw if that run shows the freeze/degrade signature.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    split_indices_path = Path(args.split_indices)
    if not split_indices_path.exists():
        raise SystemExit(
            f"{split_indices_path} does not exist.\n"
            "Copy an existing split_indices.json in first, e.g.:\n"
            f"    mkdir -p {Path(args.output_dir)}\n"
            f"    cp outputs/main1/search_run/split_indices.json {split_indices_path}\n"
            "This is required so the baseline is trained/evaluated on the exact same "
            "patient split as the searched architecture -- never a freshly computed one."
        )

    config = load_config(args.config)
    result = run_baseline_training(
        config,
        model_name=args.model,
        split_indices_path=split_indices_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        optimizer_name=args.optimizer,
        label_smoothing=args.label_smoothing,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
    )

    print(f"\nBest epoch: {result['best_epoch']}")
    print(f"Test accuracy:  {result['test_metrics']['accuracy']:.4f}")
    print(f"Test macro F1:  {result['test_metrics']['macro_f1']:.4f}")
    print(f"Test macro AUC: {result['test_metrics']['macro_auc']:.4f}")
    print(
        f"Model size: {result['flops_params']['params_million']:.2f}M params, "
        f"{result['flops_params']['flops_billion']:.3f} GFLOPs"
    )


if __name__ == "__main__":
    main()
