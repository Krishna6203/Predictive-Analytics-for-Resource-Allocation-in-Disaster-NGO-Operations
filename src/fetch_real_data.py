"""Download CivicDataLab's Assam flood ecosystem master variables dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

try:
    from .config import REAL_MASTER_PATH, SOURCE_MANIFEST_PATH, DATA_RAW
except ImportError:
    from config import REAL_MASTER_PATH, SOURCE_MANIFEST_PATH, DATA_RAW

RAW_URL = "https://github.com/CivicDataLab/flood-data-ecosystem-Assam/raw/refs/heads/main/Sources/MASTER_VARIABLES.csv"
REPO_URL = "https://github.com/CivicDataLab/flood-data-ecosystem-Assam"


def fetch(output: Path = REAL_MASTER_PATH, force: bool = False) -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        print(f"Raw data already exists: {output}")
    else:
        print(f"Downloading: {RAW_URL}")
        response = requests.get(RAW_URL, timeout=120)
        response.raise_for_status()
        output.write_bytes(response.content)
        print(f"Saved: {output} ({output.stat().st_size:,} bytes)")
    manifest = {
        "dataset": "CivicDataLab flood-data-ecosystem-Assam MASTER_VARIABLES.csv",
        "repository": REPO_URL,
        "raw_url": RAW_URL,
        "granularity": "Monthly revenue-circle-level data for Assam",
    }
    SOURCE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(REAL_MASTER_PATH))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    fetch(Path(args.output), args.force)


if __name__ == "__main__":
    main()
