#!/usr/bin/env python
"""Verify the FETAL_PLANES_DB archive before trusting it.

Same discipline as `verify_figshare.py`: measure the dataset rather than read
its description. The claims that matter here are the ones the experiment rests
on -- that the metadata CSV lines up with the files on disk, and that
`Patient_num` is a grouping key worth splitting on.

Checks, each independently fatal unless marked:

  1. LAYOUT    -- the metadata CSV and the image folder are found, and the CSV
                  parses with the columns this pipeline needs.
  2. COUNTS    -- 12,400 images, 1,792 patients, and the six published
                  per-class counts.
  3. FILES     -- every CSV row has an image on disk and every image on disk
                  has a CSV row; no duplicated image names.
  4. PATIENT   -- do near-identical images ever carry different Patient_num?
     TRUST       A grouping key that does not match reality is worse than
                 none, because it buys false confidence in the split.
  5. OFFICIAL  -- does the published `Train` column keep each patient wholly on
     SPLIT       one side? (Reported, not fatal: prepare_fetal_planes.py builds
                 its own patient-level split regardless.)
  6. GEOMETRY  -- image sizes, colour mode, and how much of the data is
                 greyscale in practice, since the pipeline forces greyscale.

Usage:
    python scripts/verify_fetal_planes.py --source data/fetal_raw/extracted
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required to read the images: pip install pillow")

REQUIRED_COLUMNS = {"Image_name", "Patient_num", "Plane", "Train"}
EXPECTED_TOTAL = 12400
EXPECTED_PATIENTS = 1792
# Per-class counts published with the dataset (Zenodo record / Sci Rep paper).
EXPECTED_PER_CLASS = {
    "Fetal abdomen": 711,
    "Fetal brain": 3092,
    "Fetal femur": 1040,
    "Fetal thorax": 1718,
    "Maternal cervix": 1626,
    "Other": 4213,
}

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


def find_metadata_csv(source: Path) -> Path | None:
    """The archive's layout is not documented field by field, so locate the
    metadata file by content: the CSV carrying the columns we need."""
    candidates = sorted(p for p in source.rglob("*.csv") if not p.name.startswith("._"))
    for path in candidates:
        try:
            columns = set(read_csv(path)[0])
        except Exception:  # noqa: BLE001 - a CSV we cannot parse is simply not the one
            continue
        if REQUIRED_COLUMNS <= columns:
            return path
    if candidates:
        print(f"         CSVs found but none carried {sorted(REQUIRED_COLUMNS)}: "
              f"{[p.name for p in candidates]}")
    return None


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV whose delimiter is sniffed, since the published file uses
    ';' while ',' is the obvious guess."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = [row for row in reader]
        return list(reader.fieldnames or []), rows


def thumbnail(path: Path, size: int = 32) -> np.ndarray:
    """Cheap downsample + z-score, for the similarity check."""
    with Image.open(path) as img:
        small = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float64).ravel()
    return (small - small.mean()) / (small.std() + 1e-8)


def near_duplicate_pairs(thumbs: np.ndarray, threshold: float, block: int = 512):
    """Indices of image pairs whose thumbnails correlate above `threshold`.

    Done in blocks: the full 12,400 x 12,400 similarity matrix is ~1.2 GB in
    float64, which is avoidable for a check that only needs the pairs.
    """
    n, dim = thumbs.shape
    for start in range(0, n, block):
        stop = min(start + block, n)
        sim = (thumbs[start:stop] @ thumbs.T) / dim
        for local, row in enumerate(sim):
            i = start + local
            row[: i + 1] = -1.0  # upper triangle only, no self-pairs
            for j in np.flatnonzero(row > threshold):
                yield i, int(j)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Directory containing the extracted archive")
    ap.add_argument("--similarity", type=float, default=0.95,
                    help="Correlation above which two images count as near-identical (default 0.95).")
    ap.add_argument("--skip-similarity", action="store_true",
                    help="Skip check 4, which reads every image (a few minutes).")
    args = ap.parse_args()

    source = Path(args.source)

    # -- 1. layout ---------------------------------------------------------
    print("1. LAYOUT")
    images_on_disk = sorted(p for p in source.rglob("*.png") if not p.name.startswith("._"))
    check(bool(images_on_disk), "image files found", f"{len(images_on_disk)} .png under {source}")
    csv_path = find_metadata_csv(source)
    if not check(csv_path is not None, f"a metadata CSV with {sorted(REQUIRED_COLUMNS)}",
                 str(csv_path) if csv_path else "none found"):
        return 1
    columns, rows = read_csv(csv_path)
    print(f"         columns: {columns}")
    check(bool(rows), "CSV has rows", f"{len(rows)} rows")
    if not rows:
        return 1

    # -- 2. counts ---------------------------------------------------------
    print("\n2. COUNTS vs the published description")
    per_class = Counter(r["Plane"].strip() for r in rows)
    patients = {r["Patient_num"].strip() for r in rows}
    check(len(rows) == EXPECTED_TOTAL, f"{EXPECTED_TOTAL} images", f"got {len(rows)}")
    check(len(patients) == EXPECTED_PATIENTS, f"{EXPECTED_PATIENTS} patients", f"got {len(patients)}")
    for name, expected in EXPECTED_PER_CLASS.items():
        check(per_class.get(name, 0) == expected, f"{name} = {expected}", f"got {per_class.get(name, 0)}")
    unexpected = set(per_class) - set(EXPECTED_PER_CLASS)
    if unexpected:
        warn("unexpected Plane values", f"{sorted(unexpected)}")
    images_per_patient = Counter(r["Patient_num"].strip() for r in rows)
    print(f"         images per patient: mean {np.mean(list(images_per_patient.values())):.1f}, "
          f"max {max(images_per_patient.values())}")

    # -- 3. CSV vs disk ----------------------------------------------------
    print("\n3. CSV ROWS vs FILES ON DISK")
    by_stem = {p.stem: p for p in images_on_disk}
    check(len(by_stem) == len(images_on_disk), "image names are unique across folders",
          f"{len(images_on_disk) - len(by_stem)} collisions")
    named = [r["Image_name"].strip() for r in rows]
    check(len(set(named)) == len(named), "no duplicated Image_name in the CSV",
          f"{len(named) - len(set(named))} duplicates")
    missing = [n for n in named if Path(n).stem not in by_stem]
    orphans = set(by_stem) - {Path(n).stem for n in named}
    check(not missing, "every CSV row has an image on disk",
          "" if not missing else f"{len(missing)} missing, e.g. {missing[:3]}")
    check(not orphans, "every image on disk has a CSV row",
          "" if not orphans else f"{len(orphans)} orphans, e.g. {sorted(orphans)[:3]}")

    # -- 4. is Patient_num trustworthy? ------------------------------------
    print("\n4. PATIENT_NUM TRUSTWORTHINESS")
    if args.skip_similarity:
        print("         skipped (--skip-similarity)")
    else:
        usable = [(r, by_stem[Path(r["Image_name"].strip()).stem]) for r in rows
                  if Path(r["Image_name"].strip()).stem in by_stem]
        print(f"         reading {len(usable)} images ...")
        thumbs = np.array([thumbnail(path) for _, path in usable])
        pids = np.array([r["Patient_num"].strip() for r, _ in usable])
        same, cross, examples = 0, 0, []
        for i, j in near_duplicate_pairs(thumbs, args.similarity):
            if pids[i] == pids[j]:
                same += 1
            else:
                cross += 1
                if len(examples) < 3:
                    examples.append(f"{usable[i][0]['Image_name']} ({pids[i]}) ~ {usable[j][0]['Image_name']} ({pids[j]})")
        total = same + cross
        fraction = same / max(1, total)
        check(total == 0 or fraction > 0.80,
              "near-identical image pairs mostly share a Patient_num",
              f"{same}/{total} = {fraction:.1%} same-patient (>{args.similarity} correlation)")
        if cross:
            warn("some near-identical pairs span different patients",
                 f"{cross} pairs, e.g. {examples}")

    # -- 5. the published Train column -------------------------------------
    print("\n5. THE PUBLISHED TRAIN/TEST SPLIT (reported, not fatal)")
    sides = defaultdict(set)
    for r in rows:
        sides[r["Patient_num"].strip()].add(r["Train"].strip())
    straddling = {p: v for p, v in sides.items() if len(v) > 1}
    n_train = sum(1 for r in rows if r["Train"].strip() in {"1", "True", "true"})
    print(f"         Train=1: {n_train} images, Train=0: {len(rows) - n_train} images")
    if straddling:
        warn("the published split puts some patients on both sides",
             f"{len(straddling)} of {len(sides)} patients, e.g. {sorted(straddling)[:3]}")
        print("         -> use prepare_fetal_planes.py's own patient-level split, not this column")
    else:
        print(f"  [ok  ] the published split is patient-disjoint ({len(sides)} patients)")

    # -- 6. geometry -------------------------------------------------------
    print("\n6. GEOMETRY")
    sample = images_on_disk[:: max(1, len(images_on_disk) // 400)]
    sizes, modes, greyscale = Counter(), Counter(), 0
    for path in sample:
        with Image.open(path) as img:
            sizes[img.size] += 1
            modes[img.mode] += 1
            rgb = np.asarray(img.convert("RGB"))
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
            greyscale += 1
    print(f"         sampled {len(sample)} images")
    print(f"         sizes: {dict(sizes.most_common(4))}")
    print(f"         modes: {dict(modes)}")
    check(greyscale / max(1, len(sample)) > 0.95,
          "images are greyscale in content (the pipeline discards colour)",
          f"{greyscale}/{len(sample)} have identical RGB channels")

    print("\n" + "=" * 62)
    if _failures:
        print(f"VERIFICATION FAILED -- {len(_failures)} check(s): {_failures}")
    else:
        print(f"ALL CHECKS PASSED ({len(_warnings)} warning(s))")
    print("=" * 62)
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
