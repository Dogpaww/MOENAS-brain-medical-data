"""Capture the git commit hash and key package versions for run manifests.

Handoff §33 requires every output directory to record software versions and
the git commit hash. This never raises: outside a git repo, or if a package
is missing, the corresponding field is just "unknown".
"""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version


def get_git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def get_software_versions() -> dict[str, str]:
    return {
        "torch": _pkg_version("torch"),
        "torchvision": _pkg_version("torchvision"),
        "numpy": _pkg_version("numpy"),
        "scikit-learn": _pkg_version("scikit-learn"),
    }


def get_run_manifest() -> dict[str, object]:
    return {
        "git_commit": get_git_commit_hash(),
        "software_versions": get_software_versions(),
    }
