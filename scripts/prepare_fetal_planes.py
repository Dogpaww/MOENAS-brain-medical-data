#!/usr/bin/env python
"""Convert FETAL_PLANES_DB into this pipeline's layout.

Source: https://doi.org/10.5281/zenodo.3904280 (CC BY 4.0)
  12,400 maternal-fetal ultrasound images from 1,792 patients, 6 classes,
  acquired at two hospitals on six machine models by several operators. The
  metadata CSV carries, per image:
      Image_name, Patient_num, Plane, Brain_plane, Operator, US_Machine, Train

Why this dataset as a second benchmark: it ships **patient numbers**, so the
train/test boundary is drawn per patient exactly as it is for figshare, and it
has six classes and a test split large enough to resolve differences of about
a tenth of a point -- where figshare's 474 test images are worth 0.21 points
each.

What this writes:

    <output>/Training/{fetal_abdomen,...,other}/*.png
    <output>/Testing/{fetal_abdomen,...,other}/*.png
    <output>/patient_ids.json          relative image path -> patient number
    <output>/fetal_manifest.json       provenance + split summary

Two differences from `prepare_figshare.py`, both forced by the data:

  - A patient here contributes images to *several* planes, so the split cannot
    stratify on "the patient's class". It uses the pipeline's own
    `grouped_stratified_split`, which balances classes while keeping every
    patient wholly on one side.
  - The published `Train` column is ignored. We build our own patient-level
    split so the protocol matches figshare's, and `verify_fetal_planes.py`
    reports separately whether that published column is patient-disjoint.

Usage:
    python scripts/prepare_fetal_planes.py --source data/fetal_raw/extracted \
        --output data/fetal_planes
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from brainmri_nas.data.split import grouped_stratified_split

REQUIRED_COLUMNS = {"Image_name", "Patient_num", "Plane"}
# Published plane names -> folder names. Fixed here rather than slugified on
# the fly so a renamed or unexpected class fails loudly instead of quietly
# creating a seventh folder.
PLANE_TO_CLASS = {
    "Fetal abdomen": "fetal_abdomen",
    "Fetal brain": "fetal_brain",
    "Fetal femur": "fetal_femur",
    "Fetal thorax": "fetal_thorax",
    "Maternal cervix": "maternal_cervix",
    "Other": "other",
}


def read_metadata(path: Path) -> list[dict[str, str]]:
    """Read the metadata CSV, choosing the delimiter by which one actually
    yields the columns we need.

    Two quirks of the published file, both of which silently break the obvious
    implementation: it is ';'-separated (so `csv.Sniffer` can land on ','), and
    its header ends `US_Machine;Train ` -- a trailing space that makes the last
    column literally named "Train ", with the values carrying it too. Names and
    values are therefore stripped.
    """
    seen: list[str] = []
    for delimiter in (";", ",", "\t"):
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            fields = [name.strip() for name in (reader.fieldnames or [])]
            if not REQUIRED_COLUMNS <= set(fields):
                seen = fields or seen
                continue
            rows = [{k.strip(): (v or "").strip() for k, v in row.items() if k is not None} for row in reader]
            if not rows:
                raise ValueError(f"{path} has no rows.")
            return rows
    raise ValueError(
        f"{path} is missing column(s) {sorted(REQUIRED_COLUMNS - set(seen))}; found {sorted(seen)}."
    )


def find_metadata_csv(source: Path) -> Path:
    for path in sorted(p for p in source.rglob("*.csv") if not p.name.startswith("._")):
        try:
            read_metadata(path)
        except Exception:  # noqa: BLE001 - a CSV we cannot parse is simply not the one
            continue
        return path
    raise SystemExit(
        f"No metadata CSV with {sorted(REQUIRED_COLUMNS)} found under {source}. "
        "Point --source at the extracted FETAL_PLANES_ZENODO.zip."
    )


def index_images(source: Path) -> dict[str, Path]:
    """Image stem -> path, so CSV rows can be matched to files wherever the
    archive puts them."""
    index: dict[str, Path] = {}
    for path in source.rglob("*.png"):
        if not path.name.startswith("._"):
            index.setdefault(path.stem, path)
    return index


def save_greyscale(source_path: Path, destination: Path, max_side: int) -> None:
    """Write a greyscale PNG, optionally capping the longer side.

    The pipeline resizes to 96 px and discards colour anyway, so a cap costs
    nothing in fidelity and saves decode time on every epoch of every run --
    this dataset is 4x figshare's size, where that cost is felt.
    """
    with Image.open(source_path) as img:
        img = img.convert("L")
        if max_side and max(img.size) > max_side:
            scale = max_side / max(img.size)
            img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)
        img.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Directory holding the extracted archive (searched recursively).")
    parser.add_argument("--output", default="data/fetal_planes")
    parser.add_argument("--test-fraction", type=float, default=0.15,
                        help="Fraction of PATIENTS held out as Testing/ (default 0.15, matching the figshare data).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-side", type=int, default=512,
                        help="Cap the longer image side in pixels; 0 keeps the original size (default 512).")
    parser.add_argument("--exclude-plane", action="append", default=[],
                        help="Published plane name to drop, e.g. --exclude-plane Other. Repeatable.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written without writing it.")
    args = parser.parse_args()

    source, output = Path(args.source), Path(args.output)
    csv_path = find_metadata_csv(source)
    rows = read_metadata(csv_path)
    print(f"Metadata: {csv_path}  ({len(rows)} rows)")

    unexpected = {r["Plane"].strip() for r in rows} - set(PLANE_TO_CLASS)
    if unexpected:
        sys.exit(f"Unexpected Plane value(s) {sorted(unexpected)}; expected {sorted(PLANE_TO_CLASS)}.")
    excluded = {p.strip() for p in args.exclude_plane}
    if excluded - set(PLANE_TO_CLASS):
        sys.exit(f"--exclude-plane got {sorted(excluded - set(PLANE_TO_CLASS))}; expected {sorted(PLANE_TO_CLASS)}.")

    images = index_images(source)
    print(f"Images:   {len(images)} .png under {source}")

    records, missing = [], []
    for row in rows:
        plane = row["Plane"].strip()
        if plane in excluded:
            continue
        stem = Path(row["Image_name"].strip()).stem
        path = images.get(stem)
        if path is None:
            missing.append(stem)
            continue
        records.append({"path": path, "stem": stem, "patient": row["Patient_num"].strip(), "plane": plane})
    if missing:
        sys.exit(
            f"{len(missing)} CSV row(s) have no image on disk, e.g. {missing[:3]}. "
            "Re-run download_fetal_planes.py, or run verify_fetal_planes.py to see what is wrong."
        )
    if excluded:
        print(f"Excluded: {sorted(excluded)} ({len(rows) - len(records)} images dropped)")

    classes = sorted({PLANE_TO_CLASS[r["plane"]] for r in records})
    class_index = {name: i for i, name in enumerate(classes)}
    patients = {r["patient"] for r in records}
    per_patient = Counter(r["patient"] for r in records)
    print(f"\n{len(records)} images from {len(patients)} patients "
          f"({sum(per_patient.values()) / len(patients):.1f} images/patient, max {max(per_patient.values())})")
    print("Class distribution (images / patients):")
    for name in classes:
        in_class = [r for r in records if PLANE_TO_CLASS[r["plane"]] == name]
        print(f"  {name:20} {len(in_class):>6} / {len({r['patient'] for r in in_class}):>5}")

    # A patient contributes several planes here, so the split is grouped by
    # patient and stratified by class -- not "stratified by the patient's class".
    targets = [class_index[PLANE_TO_CLASS[r["plane"]]] for r in records]
    groups = [r["patient"] for r in records]
    train_idx, test_idx = grouped_stratified_split(targets, groups, args.test_fraction, args.seed)
    train_patients = {groups[i] for i in train_idx}
    test_patients = {groups[i] for i in test_idx}
    if not train_patients.isdisjoint(test_patients):
        sys.exit("Split is not patient-disjoint; refusing to write it.")
    print(f"\nPatient-level split (seed={args.seed}, requested test fraction {args.test_fraction:.0%}): "
          f"Training {len(train_patients)} patients / {len(train_idx)} images | "
          f"Testing {len(test_patients)} patients / {len(test_idx)} images "
          f"({len(test_idx) / len(records):.1%} of images)")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    split_of = {}
    for i in train_idx:
        split_of[i] = "Training"
    for i in test_idx:
        split_of[i] = "Testing"

    patient_ids: dict[str, str] = {}
    per_split_class: dict[tuple[str, str], int] = defaultdict(int)
    for i, record in enumerate(records):
        split = split_of[i]
        cls = PLANE_TO_CLASS[record["plane"]]
        out_dir = output / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{record['stem']}.png"
        save_greyscale(record["path"], out_dir / filename, args.max_side)
        patient_ids[f"{split}/{cls}/{filename}"] = record["patient"]
        per_split_class[(split, cls)] += 1
        if (i + 1) % 2000 == 0:
            print(f"  wrote {i + 1}/{len(records)}")

    (output / "patient_ids.json").write_text(json.dumps(patient_ids, indent=2, sort_keys=True))

    manifest = {
        "source_doi": "10.5281/zenodo.3904280",
        "source_dir": str(source),
        "metadata_csv": str(csv_path),
        "license": "CC BY 4.0",
        "num_images": len(records),
        "num_patients": len(patients),
        "classes": classes,
        "excluded_planes": sorted(excluded),
        "max_side": args.max_side,
        "test_fraction_of_patients": args.test_fraction,
        "seed": args.seed,
        "split_is_patient_disjoint": True,
        "split_method": "grouped_stratified_split (StratifiedGroupKFold), grouped by Patient_num",
        "published_train_column_used": False,
        "patients": {"Training": sorted(train_patients), "Testing": sorted(test_patients)},
        "counts": {f"{s}/{c}": n for (s, c), n in sorted(per_split_class.items())},
    }
    (output / "fetal_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {len(records)} PNGs to {output}")
    for key, n in sorted(manifest["counts"].items()):
        print(f"  {key:36} {n:>6}")
    print(f"  patient_ids.json        {len(patient_ids)} entries")
    print("  fetal_manifest.json")
    print(f"\nNext: point a config's dataset.data_root at {output} (num_classes: {len(classes)}).")


if __name__ == "__main__":
    main()
