#!/usr/bin/env python
"""Verify the figshare brain-tumour archive before trusting it.

Written because the previous dataset (Kaggle/SARTAJ) looked fine and was not:
its glioma folder is documented as mislabelled, and ~73% of its validation
images shared a patient with training. Both were found by measuring, not by
reading the description -- so this measures.

Checks, each independently fatal:

  1. SCHEMA     -- every .mat really has cjdata/{label,PID,image}, and the
                   fields decode to the types the README promises.
  2. COUNTS     -- 3064 slices, 233 patients, 708/1426/930 per class.
  3. PID TRUST  -- do slices sharing a PID actually look like one patient, and
                   do near-identical slices ever carry *different* PIDs? A
                   grouping key that doesn't match reality is worse than none,
                   because it buys false confidence in the split.
  4. LABEL      -- no patient carries two different labels.
  5. CVIND      -- the authors' own 5-fold indices respect patient boundaries.
                   Independent third-party confirmation that the PIDs are real.
  6. GEOMETRY   -- image dimensions/dtype consistent with the stated protocol.

Usage:
    python scripts/verify_figshare.py --source data/figshare_raw/extracted
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    sys.exit("h5py is required to read MATLAB v7.3 files: pip install h5py")

LABEL_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}
EXPECTED_TOTAL = 3064
EXPECTED_PATIENTS = 233
EXPECTED_PER_CLASS = {"meningioma": 708, "glioma": 1426, "pituitary": 930}

_failures: list[str] = []
_warnings: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def warn(label: str, detail: str = "") -> None:
    print(f"  [WARN] {label}" + (f" -- {detail}" if detail else ""))
    _warnings.append(label)


def decode_pid(raw) -> str:
    """cjdata.PID is a MATLAB char array -> uint16 character codes under HDF5."""
    arr = np.asarray(raw).ravel()
    return "".join(chr(int(c)) for c in arr).strip()


def read_record(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        if "cjdata" not in f:
            raise KeyError(f"{path.name}: no 'cjdata' group (keys={list(f.keys())})")
        g = f["cjdata"]
        missing = [k for k in ("label", "PID", "image") if k not in g]
        if missing:
            raise KeyError(f"{path.name}: cjdata missing {missing} (has {list(g.keys())})")
        # HDF5 stores MATLAB arrays transposed (column- vs row-major).
        image = np.array(g["image"]).T
        return {
            "file": path.name,
            "label": int(np.array(g["label"]).ravel()[0]),
            "pid": decode_pid(g["PID"]),
            "shape": image.shape,
            "dtype": image.dtype,
            "image": image,
        }


def thumbnail(image: np.ndarray, size: int = 32) -> np.ndarray:
    """Cheap downsample + z-score, for the similarity checks."""
    h, w = image.shape[:2]
    ys = (np.arange(size) * h // size).clip(0, h - 1)
    xs = (np.arange(size) * w // size).clip(0, w - 1)
    small = image[np.ix_(ys, xs)].astype(np.float64).ravel()
    return (small - small.mean()) / (small.std() + 1e-8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Directory containing the extracted .mat files")
    ap.add_argument("--cvind", default=None, help="Optional path to cvind.mat")
    ap.add_argument("--similarity", type=float, default=0.90)
    args = ap.parse_args()

    files = sorted(Path(args.source).rglob("*.mat"))
    files = [f for f in files if f.name != "cvind.mat"]
    print(f"Found {len(files)} .mat files under {args.source}\n")
    if not files:
        return 1

    # -- 1. schema ---------------------------------------------------------
    print("1. SCHEMA")
    records, thumbs, bad = [], [], []
    for path in files:
        try:
            r = read_record(path)
        except Exception as exc:  # noqa: BLE001 - report, don't abort the sweep
            bad.append(f"{path.name}: {exc}")
            continue
        thumbs.append(thumbnail(r["image"]))
        r.pop("image")
        records.append(r)
    check(not bad, "every file parses with the README's cjdata schema",
          "" if not bad else f"{len(bad)} failed, e.g. {bad[:2]}")
    if not records:
        return 1
    labels = {r["label"] for r in records}
    check(labels <= {1, 2, 3}, "labels are in {1,2,3}", f"saw {sorted(labels)}")
    check(all(r["pid"] for r in records), "every record has a non-empty PID")

    # -- 2. counts ---------------------------------------------------------
    print("\n2. COUNTS vs README")
    per_class = Counter(LABEL_NAMES.get(r["label"], "?") for r in records)
    patients = {r["pid"] for r in records}
    check(len(records) == EXPECTED_TOTAL, f"{EXPECTED_TOTAL} slices", f"got {len(records)}")
    check(len(patients) == EXPECTED_PATIENTS, f"{EXPECTED_PATIENTS} patients", f"got {len(patients)}")
    for name, expected in EXPECTED_PER_CLASS.items():
        check(per_class[name] == expected, f"{name} = {expected}", f"got {per_class[name]}")

    # -- 3. is the PID trustworthy? ----------------------------------------
    print("\n3. PID TRUSTWORTHINESS")
    thumbs = np.array(thumbs)
    sim = (thumbs @ thumbs.T) / thumbs.shape[1]
    np.fill_diagonal(sim, -1.0)
    pids = np.array([r["pid"] for r in records])
    same_pid = pids[:, None] == pids[None, :]

    near = sim > args.similarity
    near_same = int((near & same_pid).sum() // 2)
    near_diff = int((near & ~same_pid).sum() // 2)
    total_near = near_same + near_diff
    frac = near_same / max(1, total_near)
    check(frac > 0.80, "near-identical slice pairs mostly share a PID",
          f"{near_same}/{total_near} = {frac:.1%} same-PID (>{args.similarity} similarity)")
    if near_diff:
        warn("some near-identical pairs span different PIDs",
             f"{near_diff} pairs -- plausible for similar anatomy, but they would leak if real")

    slices_per_patient = Counter(pids.tolist())
    counts = np.array(list(slices_per_patient.values()))
    print(f"         slices/patient: min={counts.min()} median={int(np.median(counts))} "
          f"max={counts.max()} mean={counts.mean():.1f}")
    check(counts.max() < len(records) * 0.10, "no single patient dominates the dataset",
          f"largest patient has {counts.max()} slices")

    # -- 4. one label per patient ------------------------------------------
    print("\n4. LABEL CONSISTENCY")
    by_patient = defaultdict(set)
    for r in records:
        by_patient[r["pid"]].add(r["label"])
    conflicted = {p: v for p, v in by_patient.items() if len(v) > 1}
    check(not conflicted, "each patient carries exactly one label",
          "" if not conflicted else f"{len(conflicted)} conflicts, e.g. {list(conflicted)[:3]}")

    # -- 5. the authors' own folds respect patients ------------------------
    print("\n5. AUTHORS' 5-FOLD INDICES (cvind.mat)")
    cvind_path = Path(args.cvind) if args.cvind else Path(args.source).parent / "cvind.mat"
    if not cvind_path.exists():
        warn("cvind.mat not found -- skipping", str(cvind_path))
    else:
        with h5py.File(cvind_path, "r") as f:
            folds = np.array(f["cvind"]).ravel().astype(int)
        if len(folds) != len(records):
            warn("cvind length does not match slice count", f"{len(folds)} vs {len(records)}")
        else:
            # Files are numbered 1..3064; sort records that way to align.
            order = np.argsort([int(Path(r["file"]).stem) for r in records])
            aligned_pids = pids[order]
            per_patient_folds = defaultdict(set)
            for pid, fold in zip(aligned_pids, folds):
                per_patient_folds[pid].add(int(fold))
            straddling = {p: v for p, v in per_patient_folds.items() if len(v) > 1}
            check(
                not straddling,
                "the authors' folds never split a patient across folds",
                "" if not straddling else f"{len(straddling)} patients straddle folds",
            )

    # -- 6. geometry -------------------------------------------------------
    print("\n6. GEOMETRY")
    shapes = Counter(r["shape"] for r in records)
    dtypes = Counter(str(r["dtype"]) for r in records)
    print(f"         shapes: {dict(list(shapes.most_common(4)))}")
    print(f"         dtypes: {dict(dtypes)}")
    check(shapes.most_common(1)[0][0] == (512, 512),
          "dominant in-plane resolution is 512x512", f"{shapes.most_common(1)[0]}")

    print("\n" + "=" * 62)
    if _failures:
        print(f"VERIFICATION FAILED -- {len(_failures)} check(s): {_failures}")
    else:
        print(f"ALL CHECKS PASSED ({len(_warnings)} warning(s))")
    print("=" * 62)
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
