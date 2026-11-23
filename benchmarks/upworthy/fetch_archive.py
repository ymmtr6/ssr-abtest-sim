# SPDX-License-Identifier: Apache-2.0
"""Download the public exploratory Upworthy Research Archive CSV and check its hash.

The file is CC BY 4.0. It is not stored in this repository. The SHA-256 below
is the hash published by OSF for the exploratory packages file (checked
2026-11-23 via the OSF API for node jd64p).

    python benchmarks/upworthy/fetch_archive.py --output data/upworthy
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

# Exploratory sample: 22,666 packages in 4,873 tests.
# https://osf.io/jd64p/files/osfstorage  file guid 3vqmp
DOWNLOAD_URL = "https://osf.io/download/3vqmp/"
FILE_NAME = "upworthy-archive-exploratory-packages-03.12.2020.csv"
SHA256 = "8368313b060f4015a0c6fb34e6d788163cee29554144aba9390b14922eb9d8ce"
BYTE_SIZE = 14_260_949
OSF_PROJECT = "https://doi.org/10.17605/OSF.IO/JD64P"
LICENSE = "CC BY 4.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str = SHA256) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"checksum mismatch for {path}: {actual} != {expected}")


def download(url: str, dest: Path, expected: str = SHA256) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "ssr-abtest-sim"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
    verify(partial, expected)
    partial.replace(dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the exploratory Upworthy archive CSV.")
    parser.add_argument("--output", type=Path, default=Path("data/upworthy"))
    parser.add_argument("--url", default=DOWNLOAD_URL)
    args = parser.parse_args(argv)
    dest = args.output / FILE_NAME
    if dest.is_file():
        verify(dest)
        print(f"Already present and checksum matches: {dest}")
        return 0
    print(f"Downloading {FILE_NAME} ({BYTE_SIZE} bytes) from {args.url}")
    print(f"License: {LICENSE}. Archive: {OSF_PROJECT}")
    download(args.url, dest)
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
