#!/usr/bin/env python3
"""GPU VRAM poller — samples nvidia-smi at a fixed interval and writes a
single report-ready JSON (metadata + timeseries + running summary), rather
than the raw --out-dir/vram_log.jsonl the A100 runbook describes (CLAUDE.md
"A100 박스 임무 지시서"). Stops on its own once the watched run's
submission.csv appears, so the file is complete and "closed" (completed:
true) the moment the run finishes — no separate step needed to fold a
jsonl into a peaks table before pasting into a report.

Usage:
  python3 scripts/vram_poll.py --out-dir runs/test_v11_ckpt1200_tta8_motion03 \
      --json-out runs/vram_ckpt1200_tta8_motion03.json --total 819 \
      --label "ckpt1200 + TTA8 balanced + motion_weight=0.3 (serving-matched)" \
      --config '{"tta": 8, "motion_weight": 0.3, "stage2": "always", "keep_ratio": 0.5}'
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

THRESHOLD_23GIB_MIB = 23 * 1024  # project convention: 24GB-card danger line


def sample_gpu(gpu_index: int) -> dict:
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}",
         "--query-gpu=memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    used, total, util = (int(x) for x in out.split(","))
    return {"used_mib": used, "total_mib": total, "util_pct": util}


def count_progress(out_dir: Path) -> int:
    p = out_dir / "progress.jsonl"
    if not p.exists():
        return 0
    with open(p) as f:
        return sum(1 for _ in f)


def summarize(samples: list[dict]) -> dict:
    used = [s["used_mib"] for s in samples]
    peak = max(used)
    total = samples[-1]["total_mib"] if samples else None
    return {
        "n_samples": len(samples),
        "peak_used_mib": peak,
        "peak_used_gib": round(peak / 1024, 2),
        "peak_pct_of_total": round(100 * peak / total, 1) if total else None,
        "mean_used_mib": round(sum(used) / len(used), 1) if used else None,
        "over_23gib_threshold": peak > THRESHOLD_23GIB_MIB,
        "note": (
            f"피크 {peak/1024:.2f}GiB — 24GB 카드(3090/4090) 가용분(~23GiB) "
            + ("초과, OOM 위험 있음" if peak > THRESHOLD_23GIB_MIB else "이내, OOM 여유 있음")
        ),
    }


def write_json(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="run's --out dir (watched for progress.jsonl / submission.csv)")
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--total", type=int, default=819)
    ap.add_argument("--interval", type=int, default=600, help="seconds between samples (default 10min)")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--config", default="{}", help="JSON string of run hyperparams, for the report header")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    json_out = Path(args.json_out)
    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    doc = {
        "meta": {
            "label": args.label,
            "out_dir": str(out_dir),
            "config": json.loads(args.config),
            "interval_s": args.interval,
            "started": started,
            "completed": False,
        },
        "samples": [],
        "summary": {},
    }

    print(f"[vram_poll] watching {out_dir} every {args.interval}s -> {json_out}", flush=True)
    while True:
        g = sample_gpu(args.gpu_index)
        done_n = count_progress(out_dir)
        entry = {
            "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "progress": f"{done_n}/{args.total}",
            **g,
        }
        doc["samples"].append(entry)
        doc["summary"] = summarize(doc["samples"])
        finished = (out_dir / "submission.csv").exists()
        if finished:
            doc["meta"]["completed"] = True
            doc["meta"]["finished"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        write_json(json_out, doc)
        print(f"[vram_poll] {entry['ts']} progress={entry['progress']} "
              f"used={g['used_mib']}MiB util={g['util_pct']}%", flush=True)
        if finished:
            print(f"[vram_poll] submission.csv detected -> done, {json_out} closed", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
