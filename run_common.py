"""Shared bootstrap for the one-command entry points.

Model resolution order: $SNUAI_MODEL_ID > local prequantized 32B > HF hub id.
Data resolution: ./data with train.csv, else data_download.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOCAL_32B = Path("/home/yhmin/model/hub/Qwen3-VL-32B-Instruct-bnb-4bit")
HUB_32B = "unsloth/Qwen3-VL-32B-Instruct-bnb-4bit"


def bootstrap_path() -> None:
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def resolve_model_id() -> str:
    env = os.environ.get("SNUAI_MODEL_ID")
    if env:
        return env
    if (LOCAL_32B / "config.json").exists():
        return str(LOCAL_32B)
    return HUB_32B


def ensure_data(root: str | Path = "data") -> Path:
    root = Path(root)
    if not root.is_absolute():
        root = REPO / root
    if (root / "train.csv").exists() and (root / "test.csv").exists():
        return root
    print(f"[data] {root} incomplete — running data_download.py")
    subprocess.run([sys.executable, str(REPO / "data_download.py"), str(root)], check=True)
    if not (root / "train.csv").exists():
        raise FileNotFoundError(f"data download did not produce {root}/train.csv")
    return root
