#!/usr/bin/env python
"""Download and extract the figshare/Cheng brain-tumour dataset.

    https://doi.org/10.6084/m9.figshare.1512427  (CC BY 4.0)
    Cheng, Jun, et al. "Enhanced Performance of Brain Tumor Classification
    via Tumor Region Augmentation and Partition." PLoS One 10.10 (2015).

~880 MB across four .zip archives plus cvind.mat and a README. File list and
MD5s are read from the figshare API rather than hardcoded, so this keeps
working if the article gains a new version -- and every archive is checked
against the MD5 the repository publishes before it is extracted.

Idempotent: an archive already present with the right MD5 is not re-fetched,
so re-running after an interrupted download resumes rather than restarts.

    python scripts/download_figshare.py --output data/figshare_raw

Then:

    python scripts/verify_figshare.py  --source data/figshare_raw/extracted
    python scripts/prepare_figshare.py --source data/figshare_raw/extracted \
                                       --output data/figshare_brain_tumor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ARTICLE_API = "https://api.figshare.com/v2/articles/1512427"
CHUNK = 1 << 20


def md5_of(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - matching figshare's published checksum, not a security control
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_manifest() -> dict:
    with urllib.request.urlopen(ARTICLE_API, timeout=60) as response:  # noqa: S310 - fixed https URL
        return json.load(response)


def download(url: str, destination: Path, expected_size: int) -> None:
    downloaded = 0
    with urllib.request.urlopen(url, timeout=1800) as response, open(destination, "wb") as out:  # noqa: S310
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            out.write(block)
            downloaded += len(block)
            if expected_size:
                pct = 100.0 * downloaded / expected_size
                print(f"\r    {downloaded / 1e6:7.1f} / {expected_size / 1e6:.1f} MB ({pct:5.1f}%)", end="", flush=True)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="data/figshare_raw")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading manifest from {ARTICLE_API} ...")
    manifest = fetch_manifest()
    files = manifest.get("files", [])
    print(f"  '{manifest.get('title')}' v{manifest.get('version')}  "
          f"doi={manifest.get('doi')}  license={(manifest.get('license') or {}).get('name')}")
    print(f"  {len(files)} files, {sum(f['size'] for f in files) / 1e6:.1f} MB total\n")

    for entry in files:
        name, expected_md5 = entry["name"], entry.get("computed_md5")
        target = out_dir / name.replace(" ", "_")

        if target.exists() and expected_md5 and md5_of(target) == expected_md5:
            print(f"  [have] {name}")
            continue

        print(f"  [get ] {name}  ({entry['size'] / 1e6:.1f} MB)")
        download(entry["download_url"], target, entry["size"])

        if expected_md5:
            actual = md5_of(target)
            if actual != expected_md5:
                print(f"    MD5 MISMATCH: expected {expected_md5}, got {actual}", file=sys.stderr)
                target.unlink(missing_ok=True)
                return 1
            print("    md5 verified")

    if args.skip_extract:
        print("\n--skip-extract: archives downloaded, not unpacked.")
        return 0

    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    print(f"\nExtracting into {extract_dir} ...")
    for archive in sorted(out_dir.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)
        print(f"  [ok] {archive.name}")

    count = len([p for p in extract_dir.rglob("*.mat") if p.name != "cvind.mat"])
    print(f"\n{count} .mat files under {extract_dir}  (expected 3064)")
    if count != 3064:
        print("  WARNING: unexpected file count -- run verify_figshare.py before using this.")

    print("\nNext:")
    print(f"  python scripts/verify_figshare.py  --source {extract_dir}")
    print(f"  python scripts/prepare_figshare.py --source {extract_dir} --output data/figshare_brain_tumor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
