"""Populate ./data on a fresh clone (A100 etc).

Tried in order:
1. $SNUAI_DATA_URL   — http(s) URL of a tar.gz/zip containing
                       train.csv/test.csv/sample_submission.csv + train/ test/
2. $SNUAI_KAGGLE_COMP — kaggle competition slug (needs kaggle CLI + creds:
                       $KAGGLE_USERNAME/$KAGGLE_KEY or ~/.kaggle/kaggle.json)
3. fail with a copy-paste rsync command for manual transfer (~4GB).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REQUIRED = ["train.csv", "test.csv", "sample_submission.csv"]


def _extract(archive: Path, dest: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)
    # flatten a single wrapping directory if present
    entries = [p for p in dest.iterdir() if p.name != archive.name]
    if len(entries) == 1 and entries[0].is_dir() and not (dest / "train.csv").exists():
        for child in entries[0].iterdir():
            shutil.move(str(child), dest / child.name)
        entries[0].rmdir()


def main() -> None:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    dest.mkdir(parents=True, exist_ok=True)

    url = os.environ.get("SNUAI_DATA_URL")
    comp = os.environ.get("SNUAI_KAGGLE_COMP")

    if url:
        print(f"[data] fetching {url}")
        with tempfile.TemporaryDirectory() as td:
            name = url.rsplit("/", 1)[-1] or "data.tar.gz"
            archive = Path(td) / name
            urllib.request.urlretrieve(url, archive)
            _extract(archive, dest)
    elif comp:
        print(f"[data] kaggle competitions download -c {comp}")
        subprocess.run(["kaggle", "competitions", "download", "-c", comp, "-p", str(dest)], check=True)
        for z in dest.glob("*.zip"):
            _extract(z, dest)
            z.unlink()
    else:
        raise SystemExit(
            "No data source configured. Either:\n"
            "  export SNUAI_DATA_URL=<https://.../snuai_data.tar.gz>   (see scripts/pack_data.sh)\n"
            "  export SNUAI_KAGGLE_COMP=<competition-slug>  (+ kaggle credentials)\n"
            "or copy manually from the dev box:\n"
            "  rsync -a --copy-links <devbox>:~/SNU-AI-Challenge/Ver11/data/ ./data/"
        )

    missing = [f for f in REQUIRED if not (dest / f).exists()]
    if missing:
        raise SystemExit(f"data incomplete after download, missing: {missing}")
    print(f"[data] ready at {dest}")


if __name__ == "__main__":
    main()
