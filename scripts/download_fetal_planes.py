#!/usr/bin/env python
"""Download and extract FETAL_PLANES_DB (common maternal-fetal ultrasound planes).

    https://doi.org/10.5281/zenodo.3904280  (CC BY 4.0)
    Burgos-Artizzu, X.P. et al. "Evaluation of deep convolutional neural
    networks for automatic classification of common maternal fetal ultrasound
    planes." Scientific Reports 10, 10200 (2020).

One ~2.1 GB archive holding 12,400 ultrasound images from 1,792 patients in 6
classes, plus a metadata CSV carrying the patient number, operator and
ultrasound machine per image. The patient column is the reason this dataset is
usable here: it lets the train/test boundary be drawn per patient.

The file list and MD5 come from the Zenodo API, with a pinned fallback for
hosts the API rejects; either way the archive is checked against the published
MD5 before it is extracted.

Resumable: a partial download continues with an HTTP range request rather than
starting the 2.1 GB again, and an archive already present with the right MD5 is
not re-fetched.

    python scripts/download_fetal_planes.py --output data/fetal_raw

Then:

    python scripts/verify_fetal_planes.py  --source data/fetal_raw/extracted
    python scripts/prepare_fetal_planes.py --source data/fetal_raw/extracted \
                                           --output data/fetal_planes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

RECORD_API = "https://zenodo.org/api/records/3904280"
CHUNK = 1 << 20
USER_AGENT = "brainmri-nas-dataset-fetch/1.0 (research; +https://doi.org/10.5281/zenodo.3904280)"
RETRIES = 3

# Pinned snapshot, used only when the Zenodo API is unreachable. Correctness
# does not depend on it being current: the MD5 check below still applies, so a
# stale entry fails loudly rather than silently yielding the wrong data.
FALLBACK_FILE = {
    "key": "FETAL_PLANES_ZENODO.zip",
    "size": 2088522169,
    "checksum": "md5:2a5fcc2cefb789bcc0f6c1f73e0ea43f",
    "links": {"self": "https://zenodo.org/api/records/3904280/files/FETAL_PLANES_ZENODO.zip/content"},
}
EXPECTED_IMAGES = 12400


def md5_of(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - matching Zenodo's published checksum, not a security control
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _open(url: str, timeout: int, headers: dict[str, str] | None = None):
    """urlopen with an identifying User-Agent and retries.

    The default `Python-urllib/x.y` agent is rejected by some CDN
    configurations, and transient 5xx/timeouts are common on a 2 GB file.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - fixed https host
        except Exception as exc:  # noqa: BLE001 - retry any transport failure
            last = exc
            if attempt < RETRIES:
                delay = 2**attempt
                print(f"    attempt {attempt}/{RETRIES} failed ({exc}); retrying in {delay}s")
                time.sleep(delay)
    raise last  # type: ignore[misc]


def fetch_file_entry(allow_fallback: bool = True) -> dict:
    try:
        with _open(RECORD_API, timeout=60) as response:
            record = json.load(response)
    except Exception as exc:  # noqa: BLE001
        if not allow_fallback:
            raise
        print(f"  API unreachable ({exc}).")
        print("  Falling back to the pinned file entry; the MD5 check still applies.")
        return FALLBACK_FILE

    files = record.get("files", [])
    archives = [f for f in files if str(f.get("key", "")).lower().endswith(".zip")]
    if not archives:
        raise SystemExit(f"No .zip file listed on {RECORD_API}; files were {[f.get('key') for f in files]}.")
    if len(archives) > 1:
        print(f"  note: {len(archives)} archives listed, taking {archives[0]['key']}")
    print(f"  '{record.get('title')}'  doi={record.get('doi')}  "
          f"license={((record.get('metadata') or {}).get('license') or {}).get('id')}")
    return archives[0]


def download(url: str, destination: Path, expected_size: int) -> None:
    """Fetch `url` into `destination`, resuming a partial file when possible."""
    existing = destination.stat().st_size if destination.exists() else 0
    if existing > expected_size:
        # Longer than the published size: it is not a prefix of the real file.
        print(f"    {destination.name} is larger than expected; starting over")
        destination.unlink()
        existing = 0

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with _open(url, timeout=1800, headers=headers) as response:
        resuming = existing > 0 and response.status == 206
        if existing and not resuming:
            # Server ignored the range request and is sending the whole file.
            print("    server does not support resuming; downloading from the start")
            existing = 0
        if resuming:
            print(f"    resuming at {existing / 1e6:.1f} MB")
        downloaded = existing
        with open(destination, "ab" if resuming else "wb") as out:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                out.write(block)
                downloaded += len(block)
                if expected_size:
                    pct = 100.0 * downloaded / expected_size
                    print(f"\r    {downloaded / 1e6:7.1f} / {expected_size / 1e6:.1f} MB ({pct:5.1f}%)",
                          end="", flush=True)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="data/fetal_raw")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--no-api", action="store_true",
                    help="Skip the Zenodo API and use the pinned file entry (for hosts the API rejects).")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_api:
        print("--no-api: using the pinned file entry.")
        entry = FALLBACK_FILE
    else:
        print(f"Reading record from {RECORD_API} ...")
        entry = fetch_file_entry()

    name = entry["key"]
    expected_size = int(entry["size"])
    expected_md5 = str(entry.get("checksum", "")).removeprefix("md5:") or None
    url = entry["links"]["self"]
    target = out_dir / name
    print(f"  {name}  {expected_size / 1e9:.2f} GB  md5={expected_md5}\n")

    if target.exists() and target.stat().st_size == expected_size and expected_md5 and md5_of(target) == expected_md5:
        print(f"  [have] {name}")
    else:
        print(f"  [get ] {name}")
        try:
            download(url, target, expected_size)
        except urllib.error.HTTPError as exc:
            print(
                f"\n  HTTP {exc.code} from Zenodo.\n"
                "  If this is 403 or 429, the host is being refused or rate-limited rather than\n"
                "  the request being wrong. Check with:\n"
                f"      curl -sI {url} | head -1\n"
                "  You can also fetch the archive on another machine, run prepare_fetal_planes.py\n"
                "  there, and copy the prepared folder across -- the patient split is seeded, so\n"
                "  preparing it elsewhere gives an identical result.",
                file=sys.stderr,
            )
            return 2

        if expected_md5:
            actual = md5_of(target)
            if actual != expected_md5:
                print(f"    MD5 MISMATCH: expected {expected_md5}, got {actual}", file=sys.stderr)
                print("    The file is incomplete or corrupt; delete it and re-run.", file=sys.stderr)
                return 1
            print("    md5 verified")

    if args.skip_extract:
        print("\n--skip-extract: archive downloaded, not unpacked.")
        return 0

    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    print(f"\nExtracting into {extract_dir} ...")
    with zipfile.ZipFile(target) as zf:
        zf.extractall(extract_dir)
    print(f"  [ok] {name}")

    images = [p for p in extract_dir.rglob("*.png") if not p.name.startswith("._")]
    csvs = sorted(p.name for p in extract_dir.rglob("*.csv"))
    print(f"\n{len(images)} .png files under {extract_dir}  (expected {EXPECTED_IMAGES})")
    print(f"metadata CSV(s): {csvs or 'NONE FOUND'}")
    if len(images) != EXPECTED_IMAGES or not csvs:
        print("  WARNING: unexpected contents -- run verify_fetal_planes.py before using this.")

    print("\nNext:")
    print(f"  python scripts/verify_fetal_planes.py  --source {extract_dir}")
    print(f"  python scripts/prepare_fetal_planes.py --source {extract_dir} --output data/fetal_planes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
