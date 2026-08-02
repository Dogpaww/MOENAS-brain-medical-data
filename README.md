# MOENAS Brain MRI

Clean reimplementation of a DARTS-style zero-cost NAS + augmentation-search pipeline for 4-class brain tumor MRI classification.

## Dataset

Uses the [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/versions/2) dataset from Kaggle (version 2).

Run from the repo root:

```bash
python3 -m pip install kagglehub
mkdir -p data/brain_tumor_mri
python3 - <<'PY'
import kagglehub

path = kagglehub.dataset_download(
    "sartajbhuvaji/brain-tumor-classification-mri/versions/2",
    output_dir="data/brain_tumor_mri",
)

print("Downloaded to:", path)
PY
```

This should leave you with:

```
data/brain_tumor_mri/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

`data/` is gitignored, so this needs to be run again on every machine you use (VM, laptop, etc.) — the dataset itself isn't tracked in this repo.
