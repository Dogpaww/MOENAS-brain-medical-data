"""FETAL_PLANES_DB preparation: metadata parsing, the patient-level split, and
that the written layout loads through the pipeline."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from brainmri_nas.data.loader import build_dataset_bundle, load_patient_ids

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = _load("prepare_fetal_planes")
verify = _load("verify_fetal_planes")

PLANES = ["Fetal abdomen", "Fetal brain", "Fetal femur", "Fetal thorax", "Maternal cervix", "Other"]


def _write_dataset(source: Path, patients: int = 8, per_plane: int = 2, delimiter: str = ";",
                   trailing_space: bool = True) -> Path:
    """A miniature archive: every patient contributes images to every plane,
    which is what makes this dataset's split different from figshare's.

    Mirrors two quirks of the published CSV: `Patient_num` is a bare integer
    (the patient also appears in the image name), and the header ends
    `US_Machine;Train ` with a trailing space that the values carry too.
    """
    images = source / "Images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for p in range(1, patients + 1):
        for plane_index, plane in enumerate(PLANES, start=1):
            for k in range(per_plane):
                name = f"Patient{p:05d}_Plane{plane_index}_{k + 1}_of_{per_plane}"
                # Non-square on purpose: the prepare step must not assume shape.
                Image.fromarray(rng.integers(0, 255, (40, 60), dtype=np.uint8), mode="L").save(images / f"{name}.png")
                rows.append({
                    "Image_name": name, "Patient_num": str(p), "Plane": plane,
                    "Brain_plane": "Not A Brain", "Operator": "Op1", "US_Machine": "Voluson E6",
                    "Train": "1" if p <= patients - 2 else "0",
                })
    csv_path = source / "FETAL_PLANES_DB_data.csv"
    pad = " " if trailing_space else ""
    with open(csv_path, "w", newline="") as f:
        f.write(delimiter.join(list(rows[0])) + pad + "\n")
        for row in rows:
            f.write(delimiter.join(row.values()) + pad + "\n")
    return csv_path


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "extracted"
    _write_dataset(root)
    return root


def _run(source: Path, output: Path, monkeypatch, *extra: str) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_fetal_planes.py", "--source", str(source),
                                      "--output", str(output), "--test-fraction", "0.25", *extra])
    prepare.main()


@pytest.mark.parametrize("delimiter", [";", ","])
@pytest.mark.parametrize("trailing_space", [True, False])
def test_metadata_is_read_whatever_the_delimiter(tmp_path: Path, delimiter: str, trailing_space: bool):
    """The published file is ';'-separated and its last column is named
    "Train " -- both of which defeated an earlier sniff-and-exact-match read."""
    csv_path = _write_dataset(tmp_path / "extracted", patients=2, per_plane=1,
                              delimiter=delimiter, trailing_space=trailing_space)
    rows = prepare.read_metadata(csv_path)
    assert len(rows) == 12
    assert rows[0]["Patient_num"] == "1"
    assert set(rows[0]) >= {"Image_name", "Patient_num", "Plane", "Train"}
    assert rows[0]["Train"] in {"0", "1"}  # stripped: the raw value is "0 " / "1 "


def test_metadata_without_the_needed_columns_is_rejected(tmp_path: Path):
    path = tmp_path / "other.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing column"):
        prepare.read_metadata(path)


def test_the_metadata_csv_is_found_among_decoys(source: Path):
    (source / "readme_counts.csv").write_text("total,images\n12400,1\n")
    assert prepare.find_metadata_csv(source).name == "FETAL_PLANES_DB_data.csv"


def test_macos_sidecar_files_are_ignored(source: Path):
    (source / "Images" / "._Patient00001_Plane1_1_of_2.png").write_bytes(b"not an image")
    index = prepare.index_images(source)
    assert not any(stem.startswith("._") for stem in index)


def test_greyscale_conversion_caps_the_longer_side(tmp_path: Path):
    colour = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((300, 900, 3), dtype=np.uint8), mode="RGB").save(colour)

    prepare.save_greyscale(colour, tmp_path / "capped.png", max_side=512)
    with Image.open(tmp_path / "capped.png") as img:
        assert img.mode == "L" and max(img.size) == 512 and img.size == (512, 171)

    prepare.save_greyscale(colour, tmp_path / "full.png", max_side=0)
    with Image.open(tmp_path / "full.png") as img:
        assert img.size == (900, 300)


def test_end_to_end_produces_a_loadable_patient_disjoint_dataset(source: Path, tmp_path: Path, monkeypatch):
    output = tmp_path / "prepared"
    _run(source, output, monkeypatch)

    manifest = json.loads((output / "fetal_manifest.json").read_text())
    assert manifest["num_images"] == 96
    assert manifest["classes"] == ["fetal_abdomen", "fetal_brain", "fetal_femur", "fetal_thorax",
                                   "maternal_cervix", "other"]
    assert manifest["published_train_column_used"] is False
    train_patients = set(manifest["patients"]["Training"])
    test_patients = set(manifest["patients"]["Testing"])
    assert train_patients and test_patients and train_patients.isdisjoint(test_patients)

    patient_ids = json.loads((output / "patient_ids.json").read_text())
    assert len(patient_ids) == 96
    # The split recorded in the manifest must match where images actually landed.
    for relative, patient in patient_ids.items():
        expected = "Training" if patient in train_patients else "Testing"
        assert relative.startswith(expected + "/")

    bundle = build_dataset_bundle(output, image_size=96, validation_fraction=0.34, split_seed=42,
                                  batch_size=4, split_indices_path=output / "split_indices.json",
                                  group_aware_split=True)
    assert bundle.num_classes == 6
    samples = bundle.train_loader.dataset.dataset.samples
    ids = load_patient_ids(output)
    in_train = {ids[str(Path(samples[i][0]).relative_to(output))] for i in bundle.train_indices}
    in_val = {ids[str(Path(samples[i][0]).relative_to(output))] for i in bundle.val_indices}
    assert in_train.isdisjoint(in_val)
    x, y = next(iter(bundle.train_loader))
    assert x.shape[1:] == (3, 96, 96) and set(y.tolist()) <= set(range(6))


def test_excluding_a_plane_drops_that_class(source: Path, tmp_path: Path, monkeypatch):
    output = tmp_path / "prepared"
    _run(source, output, monkeypatch, "--exclude-plane", "Other")

    manifest = json.loads((output / "fetal_manifest.json").read_text())
    assert "other" not in manifest["classes"] and len(manifest["classes"]) == 5
    assert manifest["num_images"] == 80
    assert not (output / "Training" / "other").exists()


def test_a_csv_row_without_an_image_aborts(source: Path, tmp_path: Path, monkeypatch):
    (source / "Images" / "Patient00001_Plane1_1_of_2.png").unlink()
    with pytest.raises(SystemExit, match="no image on disk"):
        _run(source, tmp_path / "prepared", monkeypatch)


def test_an_unexpected_plane_name_aborts(source: Path, tmp_path: Path, monkeypatch):
    csv_path = source / "FETAL_PLANES_DB_data.csv"
    rows = list(csv.DictReader(open(csv_path), delimiter=";"))
    rows[0]["Plane"] = "Fetal spine"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SystemExit, match="Unexpected Plane"):
        _run(source, tmp_path / "prepared", monkeypatch)


def test_verify_flags_near_duplicates_that_span_patients():
    """The check that decides whether Patient_num can be trusted as a grouping key."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 1024))
    thumbs = np.repeat(base, 3, axis=0) + rng.normal(scale=1e-3, size=(3, 1024))
    thumbs = (thumbs - thumbs.mean(axis=1, keepdims=True)) / thumbs.std(axis=1, keepdims=True)
    pairs = list(verify.near_duplicate_pairs(thumbs, threshold=0.95))
    assert sorted(pairs) == [(0, 1), (0, 2), (1, 2)]

    unrelated = rng.normal(size=(2, 1024))
    unrelated = (unrelated - unrelated.mean(axis=1, keepdims=True)) / unrelated.std(axis=1, keepdims=True)
    assert list(verify.near_duplicate_pairs(unrelated, threshold=0.95)) == []


def _rewrite_csv(csv_path: Path, mutate) -> None:
    lines = csv_path.read_text().splitlines()
    header, rows = lines[0], [line.split(";") for line in lines[1:]]
    mutate(rows)
    csv_path.write_text("\n".join([header] + [";".join(r) for r in rows]) + "\n")


def test_published_split_uses_the_train_column(source: Path, tmp_path: Path, monkeypatch):
    output = tmp_path / "prepared"
    _run(source, output, monkeypatch, "--use-published-split")

    manifest = json.loads((output / "fetal_manifest.json").read_text())
    assert manifest["published_train_column_used"] is True
    assert manifest["split_method"] == "published Train column"
    # Fraction and seed had no effect here, so the manifest must not imply they did.
    assert manifest["test_fraction_of_patients"] is None and manifest["seed"] is None

    # The fixture marks the last two of eight patients as Train=0.
    assert set(manifest["patients"]["Testing"]) == {"7", "8"}
    assert set(manifest["patients"]["Training"]) == {str(p) for p in range(1, 7)}
    ids = json.loads((output / "patient_ids.json").read_text())
    for relative, patient in ids.items():
        assert relative.startswith(("Testing/" if patient in {"7", "8"} else "Training/"))


def test_published_split_is_refused_when_a_patient_straddles_it(source: Path, tmp_path: Path, monkeypatch):
    """The one property that makes the published column worth using at all."""
    _rewrite_csv(source / "FETAL_PLANES_DB_data.csv",
                 lambda rows: rows[0].__setitem__(-1, "0 "))  # patient 1 now has both flags

    with pytest.raises(SystemExit, match="not patient-disjoint"):
        _run(source, tmp_path / "prepared", monkeypatch, "--use-published-split")


def test_published_split_rejects_unrecognised_flags(source: Path, tmp_path: Path, monkeypatch):
    _rewrite_csv(source / "FETAL_PLANES_DB_data.csv",
                 lambda rows: [row.__setitem__(-1, "maybe ") for row in rows])

    with pytest.raises(SystemExit, match="unrecognised value"):
        _run(source, tmp_path / "prepared", monkeypatch, "--use-published-split")


def test_the_two_split_modes_disagree_about_where_patients_land(source: Path, tmp_path: Path, monkeypatch):
    """A guard against the flag silently doing nothing."""
    ours, published = tmp_path / "ours", tmp_path / "published"
    _run(source, ours, monkeypatch)
    _run(source, published, monkeypatch, "--use-published-split")

    a = json.loads((ours / "fetal_manifest.json").read_text())["patients"]["Testing"]
    b = json.loads((published / "fetal_manifest.json").read_text())["patients"]["Testing"]
    assert set(a) != set(b)
    # Both remain patient-disjoint and cover every image.
    for directory in (ours, published):
        manifest = json.loads((directory / "fetal_manifest.json").read_text())
        assert set(manifest["patients"]["Training"]).isdisjoint(manifest["patients"]["Testing"])
        assert manifest["num_images"] == 96
