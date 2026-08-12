from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from brainmri_nas.data.loader import build_dataset_bundle, load_patient_ids

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_figshare.py"
_spec = importlib.util.spec_from_file_location("prepare_figshare", SCRIPT)
prepare_figshare = importlib.util.module_from_spec(_spec)
sys.modules["prepare_figshare"] = prepare_figshare
_spec.loader.exec_module(prepare_figshare)


def _write_v73_mat(path: Path, label: int, pid: str, image: np.ndarray) -> None:
    """Imitate a figshare cjdata struct as MATLAB v7.3 writes it: char arrays
    as uint16 codes, arrays stored transposed (column-major)."""
    with h5py.File(path, "w") as f:
        group = f.create_group("cjdata")
        group.create_dataset("label", data=np.array([[float(label)]]))
        group.create_dataset("PID", data=np.array([[ord(c)] for c in pid], dtype=np.uint16))
        group.create_dataset("image", data=image.T.astype(np.int16))
        group.create_dataset("tumorMask", data=np.zeros_like(image).T)


@pytest.fixture
def figshare_source(tmp_path: Path) -> Path:
    """9 patients x 3 slices, 3 patients per class -- the minimum that still
    supports a stratified patient-level split."""
    source = tmp_path / "mat"
    source.mkdir()
    rng = np.random.default_rng(0)
    index = 0
    for label in (1, 2, 3):
        for p in range(3):
            pid = f"PT{label}{p:02d}"
            for s in range(3):
                image = (rng.random((32, 24)) * 4000).astype(np.int16)  # non-square on purpose
                _write_v73_mat(source / f"{index}.mat", label, pid, image)
                index += 1
    return source


def test_reads_label_pid_and_image_from_a_v73_mat(tmp_path: Path):
    image = np.arange(6 * 4, dtype=np.int16).reshape(6, 4)
    _write_v73_mat(tmp_path / "x.mat", label=2, pid="ABC123", image=image)

    rec = prepare_figshare.read_mat_record(tmp_path / "x.mat")
    assert rec["label"] == 2
    assert rec["pid"] == "ABC123"
    # Transposition must be undone, or every image would be silently rotated.
    assert rec["image"].shape == (6, 4)
    np.testing.assert_array_equal(rec["image"], image)


def test_to_uint8_rescales_per_image_and_survives_a_constant_image():
    out = prepare_figshare.to_uint8(np.array([[100, 4100]], dtype=np.int16))
    assert out.dtype == np.uint8
    assert (out.min(), out.max()) == (0, 255)
    assert prepare_figshare.to_uint8(np.full((4, 4), 7, dtype=np.int16)).max() == 0


def test_patient_split_is_patient_disjoint_and_stratified():
    patient_labels = {f"p{i}": (i % 3) + 1 for i in range(30)}
    train, test = prepare_figshare.split_patients(patient_labels, test_fraction=0.2, seed=0)
    assert train.isdisjoint(test)
    assert train | test == set(patient_labels)
    assert {patient_labels[p] for p in test} == {1, 2, 3}


def test_end_to_end_produces_a_loadable_patient_disjoint_dataset(figshare_source: Path, tmp_path: Path, monkeypatch):
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        sys, "argv",
        ["prepare_figshare.py", "--source", str(figshare_source), "--output", str(output),
         "--test-fraction", "0.34", "--seed", "0"],
    )
    prepare_figshare.main()

    for split in ("Training", "Testing"):
        for cls in ("glioma_tumor", "meningioma_tumor", "pituitary_tumor"):
            assert (output / split / cls).is_dir(), f"missing {split}/{cls}"

    patient_ids = load_patient_ids(output)
    assert patient_ids is not None
    assert len(patient_ids) == 27  # 9 patients x 3 slices

    # No patient may appear in both Training/ and Testing/.
    by_split = {"Training": set(), "Testing": set()}
    for rel, pid in patient_ids.items():
        by_split[rel.split("/")[0]].add(pid)
    assert by_split["Training"].isdisjoint(by_split["Testing"])

    manifest = json.loads((output / "figshare_manifest.json").read_text())
    assert manifest["num_slices"] == 27
    assert manifest["num_patients"] == 9
    assert manifest["split_is_patient_disjoint"] is True

    # And the prepared folder drops straight into the normal loader, with the
    # train/val split also coming out patient-disjoint.
    bundle = build_dataset_bundle(
        output, image_size=32, validation_fraction=0.34, split_seed=1, batch_size=2
    )
    assert bundle.num_classes == 3
    samples = bundle.train_loader.dataset.dataset.samples
    keys = [patient_ids[Path(p).relative_to(output).as_posix()] for p, _ in samples]
    train_patients = {keys[i] for i in bundle.train_indices}
    val_patients = {keys[i] for i in bundle.val_indices}
    assert train_patients and val_patients
    assert train_patients.isdisjoint(val_patients)


def test_aborts_when_one_patient_carries_two_labels(tmp_path: Path, monkeypatch):
    """The grouping key must be trustworthy; a patient with two tumour types
    means it isn't, and proceeding would produce a meaningless split."""
    source = tmp_path / "mat"
    source.mkdir()
    image = np.ones((8, 8), dtype=np.int16)
    _write_v73_mat(source / "a.mat", label=1, pid="SAME", image=image)
    _write_v73_mat(source / "b.mat", label=2, pid="SAME", image=image)

    monkeypatch.setattr(
        sys, "argv",
        ["prepare_figshare.py", "--source", str(source), "--output", str(tmp_path / "out")],
    )
    with pytest.raises(SystemExit, match="more than one label"):
        prepare_figshare.main()
