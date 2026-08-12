#!/usr/bin/env python
"""Convert the figshare/Cheng brain-tumour dataset into this pipeline's layout.

Source: https://doi.org/10.6084/m9.figshare.1512427 (CC BY 4.0)
  3064 T1-weighted contrast-enhanced slices from 233 patients, acquired at
  Nanfang Hospital and General Hospital of Tianjin Medical University.
  Each .mat holds a `cjdata` struct with:
      cjdata.label       1 = meningioma, 2 = glioma, 3 = pituitary
      cjdata.PID         patient id (MATLAB char array)
      cjdata.image       512x512 slice
      cjdata.tumorBorder / cjdata.tumorMask   (unused here)

Why this dataset: it ships **patient IDs**, so the train/test boundary can be
drawn per patient instead of per image. On the previous Kaggle/SARTAJ data we
measured 73.3% of validation images sharing a near-duplicate group with a
training image, worth 12.2 points of purely illusory validation accuracy.
Patient IDs make that impossible to get wrong rather than merely unlikely.

What this writes:

    <output>/Training/{glioma,meningioma,pituitary}_tumor/*.png
    <output>/Testing/{glioma,meningioma,pituitary}_tumor/*.png
    <output>/patient_ids.json      relative image path -> patient id
    <output>/figshare_manifest.json  provenance + split summary

The Training/Testing split here is **patient-disjoint**, and `patient_ids.json`
then lets `build_dataset_bundle` keep the train/val split patient-disjoint too.

Usage:
    python scripts/prepare_figshare.py --source <dir-of-.mat-files> \
        --output data/figshare_brain_tumor
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# figshare's numeric labels -> folder names. The `_tumor` suffix matches the
# convention the rest of the project already uses.
LABEL_TO_CLASS = {1: "meningioma_tumor", 2: "glioma_tumor", 3: "pituitary_tumor"}


def _decode_matlab_string(raw) -> str:
    """MATLAB char arrays land in HDF5 as uint16 character codes."""
    arr = np.asarray(raw).flatten()
    return "".join(chr(int(c)) for c in arr).strip()


def read_mat_record(path: Path) -> dict:
    """Read one cjdata struct. Tries HDF5 (MATLAB v7.3) first, then falls back
    to scipy for older v7 files -- the figshare archive is v7.3, but the
    fallback costs little and makes this usable on re-saved copies."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            cj = f["cjdata"]
            label = int(np.asarray(cj["label"]).flatten()[0])
            pid = _decode_matlab_string(cj["PID"][:])
            # HDF5 stores MATLAB arrays transposed (column-major -> row-major).
            image = np.asarray(cj["image"]).T
    except OSError:
        from scipy.io import loadmat

        cj = loadmat(str(path), simplify_cells=True)["cjdata"]
        label = int(cj["label"])
        pid = str(cj["PID"]).strip()
        image = np.asarray(cj["image"])

    if label not in LABEL_TO_CLASS:
        raise ValueError(f"{path.name}: unexpected label {label!r}, expected one of {sorted(LABEL_TO_CLASS)}.")
    if not pid:
        raise ValueError(f"{path.name}: empty patient id.")
    if image.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2-D slice, got shape {image.shape}.")
    return {"label": label, "pid": pid, "image": image}


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Per-image min-max to 8-bit.

    The source is 16-bit with a scanner-dependent range, and PNG/ImageFolder
    work in 8-bit. Per-image (rather than global) scaling is consistent with
    the pipeline's `PerImageNormalize`, which already removes absolute
    intensity as a variable. A constant image maps to all-zeros.
    """
    img = image.astype(np.float64)
    low, high = float(img.min()), float(img.max())
    if high <= low:
        return np.zeros(img.shape, dtype=np.uint8)
    return np.round((img - low) / (high - low) * 255.0).astype(np.uint8)


def split_patients(
    patient_labels: dict[str, int], test_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    """Patient-level stratified split. Patients, never images, are the unit --
    that is the entire point of using this dataset."""
    patients = sorted(patient_labels)
    labels = [patient_labels[p] for p in patients]
    train_patients, test_patients = train_test_split(
        patients, test_size=test_fraction, random_state=seed, stratify=labels
    )
    return set(train_patients), set(test_patients)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Directory containing the figshare .mat files (searched recursively).")
    parser.add_argument("--output", default="data/figshare_brain_tumor")
    parser.add_argument("--test-fraction", type=float, default=0.20, help="Fraction of PATIENTS held out as Testing/.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written without writing it.")
    args = parser.parse_args()

    source, output = Path(args.source), Path(args.output)
    mat_files = sorted(source.rglob("*.mat"))
    if not mat_files:
        sys.exit(f"No .mat files found under {source}. Point --source at the extracted figshare archive.")
    print(f"Found {len(mat_files)} .mat files under {source}")

    records, patient_labels = [], {}
    conflicts = []
    for i, path in enumerate(mat_files, 1):
        rec = read_mat_record(path)
        rec["source"] = path
        records.append(rec)
        # A patient must not carry two different tumour types; if one does,
        # the grouping key is not trustworthy and we should not proceed quietly.
        seen = patient_labels.setdefault(rec["pid"], rec["label"])
        if seen != rec["label"]:
            conflicts.append((rec["pid"], seen, rec["label"]))
        if i % 500 == 0:
            print(f"  read {i}/{len(mat_files)}")

    if conflicts:
        sys.exit(
            f"{len(conflicts)} patient(s) carry more than one label, e.g. {conflicts[:3]}. "
            "Patient-level splitting assumes one tumour type per patient -- aborting rather "
            "than producing a split whose grouping key is unreliable."
        )

    slices_per_patient = Counter(r["pid"] for r in records)
    print(
        f"\n{len(records)} slices from {len(patient_labels)} patients "
        f"({np.mean(list(slices_per_patient.values())):.1f} slices/patient, "
        f"max {max(slices_per_patient.values())})"
    )
    print("Class distribution (slices / patients):")
    for label, cls in sorted(LABEL_TO_CLASS.items()):
        n_sl = sum(1 for r in records if r["label"] == label)
        n_pt = sum(1 for v in patient_labels.values() if v == label)
        print(f"  {cls:20} {n_sl:>5} / {n_pt:>4}")

    train_patients, test_patients = split_patients(patient_labels, args.test_fraction, args.seed)
    n_train_sl = sum(1 for r in records if r["pid"] in train_patients)
    print(
        f"\nPatient-level split (seed={args.seed}): "
        f"Training {len(train_patients)} patients / {n_train_sl} slices | "
        f"Testing {len(test_patients)} patients / {len(records) - n_train_sl} slices"
    )
    assert train_patients.isdisjoint(test_patients)

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    patient_ids: dict[str, str] = {}
    per_split_class = defaultdict(int)
    for rec in records:
        split = "Training" if rec["pid"] in train_patients else "Testing"
        cls = LABEL_TO_CLASS[rec["label"]]
        out_dir = output / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_pid = "".join(c if c.isalnum() else "_" for c in rec["pid"])
        filename = f"{safe_pid}__{rec['source'].stem}.png"
        Image.fromarray(to_uint8(rec["image"]), mode="L").save(out_dir / filename)

        patient_ids[f"{split}/{cls}/{filename}"] = rec["pid"]
        per_split_class[(split, cls)] += 1

    (output / "patient_ids.json").write_text(json.dumps(patient_ids, indent=2, sort_keys=True))

    manifest = {
        "source_doi": "10.6084/m9.figshare.1512427",
        "source_dir": str(source),
        "license": "CC BY 4.0",
        "num_slices": len(records),
        "num_patients": len(patient_labels),
        "test_fraction_of_patients": args.test_fraction,
        "seed": args.seed,
        "split_is_patient_disjoint": True,
        "patients": {"Training": sorted(train_patients), "Testing": sorted(test_patients)},
        "counts": {f"{s}/{c}": n for (s, c), n in sorted(per_split_class.items())},
    }
    (output / "figshare_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {len(records)} PNGs to {output}")
    for key, n in sorted(manifest["counts"].items()):
        print(f"  {key:36} {n:>5}")
    print(f"  patient_ids.json        {len(patient_ids)} entries")
    print(f"  figshare_manifest.json")
    print(f"\nNext: point a config's dataset.data_root at {output} (num_classes: 3).")


if __name__ == "__main__":
    main()
