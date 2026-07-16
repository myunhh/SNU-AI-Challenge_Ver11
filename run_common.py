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
    """torchrun DDP에서는 여러 프로세스가 동시에 이걸 부르므로, rank0가 아닌 프로세스는
    다운로드하지 않고 파일이 생기길 기다린다(동시 다운로드 경합 방지). torch.distributed는
    아직 초기화 전(모델 로딩보다 먼저 호출됨)이라 RANK 환경변수만 본다."""
    root = Path(root)
    if not root.is_absolute():
        root = REPO / root
    if (root / "train.csv").exists() and (root / "test.csv").exists():
        return root
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        import time
        print(f"[data][rank{rank}] {root} 대기 중 (rank0 다운로드 완료까지)…")
        for _ in range(600):   # 최대 ~30분 폴링
            if (root / "train.csv").exists() and (root / "test.csv").exists():
                return root
            time.sleep(3)
        raise SystemExit(f"[data][rank{rank}] rank0의 data_download.py가 30분 내 안 끝남 — 확인 필요")
    print(f"[data] {root} incomplete — running data_download.py")
    subprocess.run([sys.executable, str(REPO / "data_download.py"), str(root)], check=True)
    if not (root / "train.csv").exists():
        raise FileNotFoundError(f"data download did not produce {root}/train.csv")
    return root
