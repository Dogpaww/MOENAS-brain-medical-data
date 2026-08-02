#!/usr/bin/env python
"""Run NSGA-II zero-cost architecture search.

Usage:
    python scripts/run_search.py --config configs/search.yaml --output-dir outputs/search_run
"""

from __future__ import annotations

import argparse

from brainmri_nas.search.nsga2_runner import run_search
from brainmri_nas.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/search.yaml")
    parser.add_argument("--output-dir", default="outputs/search_run")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_search(config, args.output_dir)
    print(f"Selected architecture: {result['selected_architecture']['candidate_hash']}")


if __name__ == "__main__":
    main()
