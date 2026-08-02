"""Regression test for scripts/smoke_test.py (handoff §32, acceptance §35 item 25).

Runs the actual script as a subprocess -- not just its underlying library
calls -- so this exercises the exact command a user or CI would run,
including CLI argument parsing.

The synthetic-fallback test forces `--data-root` to a path that's guaranteed
not to exist, rather than relying on `data/brain_tumor_mri` being absent --
that assumption held when this test was first written, but the repo may
have a real downloaded dataset there now (as it does whenever someone's
actually working with real data), so the test must not depend on ambient
repo state to pick which code path it exercises.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_test.py"
REAL_DATA_ROOT = REPO_ROOT / "data" / "brain_tumor_mri"


def _run_smoke_test(output_dir: Path, *, data_root: Path | None) -> subprocess.CompletedProcess:
    args = [
        sys.executable,
        "-u",
        str(SMOKE_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--max-train-samples",
        "16",
        "--max-val-samples",
        "4",
        "--seed",
        "0",
    ]
    if data_root is not None:
        args += ["--data-root", str(data_root)]

    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)


def _assert_passed_with_all_outputs(result: subprocess.CompletedProcess, output_dir: Path) -> None:
    assert result.returncode == 0, f"smoke_test.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "SMOKE TEST PASSED" in result.stdout

    for step_num in range(1, 15):
        assert f"[{step_num}/14]" in result.stdout, f"step {step_num} did not run"

    search_dir = output_dir / "search_run"
    training_dir = output_dir / "training_run"
    for path in (
        search_dir / "selected_architecture.json",
        search_dir / "pareto_front.json",
        search_dir / "topsis_ranking.json",
        training_dir / "best_checkpoint.pt",
        training_dir / "test_metrics.json",
        training_dir / "confusion_matrix.png",
        output_dir / "selected_policy.json",
    ):
        assert path.exists(), f"missing expected output: {path}"


def test_smoke_test_falls_back_to_synthetic_data_when_none_is_configured(tmp_path: Path):
    output_dir = tmp_path / "smoke_output"
    missing_data_root = tmp_path / "definitely_does_not_exist"

    result = _run_smoke_test(output_dir, data_root=missing_data_root)

    assert "falling back to a small SYNTHETIC dataset" in result.stdout
    _assert_passed_with_all_outputs(result, output_dir)


def test_smoke_test_uses_real_data_when_present():
    if not (REAL_DATA_ROOT / "Training").is_dir():
        import pytest

        pytest.skip(f"no real dataset at {REAL_DATA_ROOT}")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "smoke_output"
        result = _run_smoke_test(output_dir, data_root=REAL_DATA_ROOT)

        assert "found a real dataset" in result.stdout
        assert "falling back to a small SYNTHETIC dataset" not in result.stdout
        _assert_passed_with_all_outputs(result, output_dir)
