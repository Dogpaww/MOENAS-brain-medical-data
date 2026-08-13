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
import time
import urllib.request
import zipfile
from pathlib import Path

ARTICLE_API = "https://api.figshare.com/v2/articles/1512427"
DOWNLOAD_HOST = "https://ndownloader.figshare.com/files"
CHUNK = 1 << 20
USER_AGENT = "brainmri-nas-dataset-fetch/1.0 (research; +https://doi.org/10.6084/m9.figshare.1512427)"
RETRIES = 3

# Pinned snapshot of article version 8, used only when api.figshare.com is
# unreachable -- some hosts (observed on a GCP VM) get HTTP 403 from the API
# while the *download* host stays reachable, since it is a different service.
# Correctness does not rest on this list being current: every file is still
# checked against its MD5 before use, so a stale entry fails loudly rather
# than silently yielding the wrong data.
FALLBACK_VERSION = 8
FALLBACK_FILES = [
    ("brainTumorDataPublic_1-766.zip", 3381290, "74b949ad33f042e6e103523091cd1428", 214401279),
    ("brainTumorDataPublic_767-1532.zip", 3381296, "7e8a875500d2c8a346f270538e29890e", 217848429),
    ("brainTumorDataPublic_1533-2298.zip", 3381293, "8227bf6080cb71f15a88be8d25c79ae7", 215563856),
    ("brainTumorDataPublic_2299-3064.zip", 3381302, "b378a80d6174e5317d59eb28430c6652", 231679762),
    ("cvind.mat", 7005344, "5ea82eb3212c9fcb84290ccf0beb486d", 5736),
    ("README 2024.txt", 51340418, "4188f7c30957e9673c3611613071bd6e", 3303),
]


def md5_of(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - matching figshare's published checksum, not a security control
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _open(url: str, timeout: int):
    """urlopen with an identifying User-Agent and retries.

    The default `Python-urllib/x.y` agent is rejected by some CDN
    configurations, and transient 5xx/timeouts are common on a ~200 MB file.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - fixed https hosts
        except Exception as exc:  # noqa: BLE001 - retry any transport failure
            last = exc
            if attempt < RETRIES:
                delay = 2**attempt
                print(f"    attempt {attempt}/{RETRIES} failed ({exc}); retrying in {delay}s")
                time.sleep(delay)
    raise last  # type: ignore[misc]


def _fallback_manifest() -> dict:
    return {
        "title": "brain tumor dataset",
        "version": FALLBACK_VERSION,
        "doi": "10.6084/m9.figshare.1512427.v8",
        "license": {"name": "CC BY 4.0"},
        "files": [
            {
                "name": name,
                "id": file_id,
                "computed_md5": md5,
                "size": size,
                "download_url": f"{DOWNLOAD_HOST}/{file_id}",
            }
            for name, file_id, md5, size in FALLBACK_FILES
        ],
    }


def fetch_manifest(allow_fallback: bool = True) -> dict:
    try:
        with _open(ARTICLE_API, timeout=60) as response:
            return json.load(response)
    except Exception as exc:  # noqa: BLE001
        if not allow_fallback:
            raise
        print(f"  API unreachable ({exc}).")
        print(f"  Falling back to the pinned v{FALLBACK_VERSION} file list; MD5 checks still apply.")
        return _fallback_manifest()


def download(url: str, destination: Path, expected_size: int) -> None:
    downloaded = 0
    with _open(url, timeout=1800) as response, open(destination, "wb") as out:
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
    ap.add_argument(
        "--no-api",
        action="store_true",
        help="Skip api.figshare.com entirely and use the pinned file list (for hosts the API rejects).",
    )
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_api:
        print(f"--no-api: using the pinned v{FALLBACK_VERSION} file list.")
        manifest = _fallback_manifest()
    else:
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
        try:
            download(entry["download_url"], target, entry["size"])
        except urllib.error.HTTPError as exc:
            target.unlink(missing_ok=True)
            if exc.code != 403:
                raise
            # Observed on a GCP VM: figshare 403s the whole network, both
            # api.figshare.com and ndownloader.figshare.com. Nothing in this
            # script can route around that, so say so instead of retrying
            # three more times and printing a traceback.
            print(
                f"\n  HTTP 403 from {DOWNLOAD_HOST}.\n"
                "  figshare is refusing this host, not this request -- some cloud IP ranges\n"
                "  are blocked outright. Confirm with:\n"
                f"      curl -sI {DOWNLOAD_HOST}/7005344 | head -1\n"
                "  A 403 there means no download client will work from this machine.\n\n"
                "  Fetch the dataset on a machine that can reach figshare, run\n"
                "  prepare_figshare.py there, and copy the prepared folder across\n"
                "  (~305 MB, versus 880 MB of raw archives):\n"
                "      tar -cf figshare_brain_tumor.tar figshare_brain_tumor\n"
                "      scp figshare_brain_tumor.tar <host>:<repo>/data/\n"
                "      ssh <host> 'cd <repo>/data && tar -xf figshare_brain_tumor.tar'\n"
                "  The patient split is seeded, so preparing it elsewhere gives an\n"
                "  identical result to preparing it here.",
                file=sys.stderr,
            )
            return 2

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
