#!/usr/bin/env python3
"""Ver11 motion run — terminal dashboard (stdlib only, no deps).

  python3 scripts/term_dashboard.py           # one snapshot, then exit
  python3 scripts/term_dashboard.py --watch   # auto-refresh every 20s (Ctrl-C to stop)
  python3 scripts/term_dashboard.py --watch 5 # custom interval (seconds)
  python3 scripts/term_dashboard.py --no-color
  python3 scripts/term_dashboard.py --run runs/sft32b_v11_ws8_reheat --total-steps 300 \
      --log runs/sft_v11_ws8_reheat.log   # point at a different run (e.g. a resume/reheat)
  python3 scripts/term_dashboard.py --watch --infer-out runs/test_v11_ckpt1200
      # test/train INFERENCE progress instead of training (progress.jsonl-based)

Reads the same on-disk sources as runs/dashboard.html (train_log.jsonl,
vram_log.jsonl, checkpoints, sweep/test artifacts) — always live, never
stale. Safe on partial data (pre-launch, mid-run, or finished).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/sft32b_v11_ws8"
SFT_LOG = ROOT / "runs/sft_v11_ws8.log"
VRAM_LOG = ROOT / "runs/vram_log.jsonl"
TOTAL_STEPS = 1500
TRIPWIRE_STEP = 200
KST = timezone(timedelta(hours=9))
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BLOCKS = "▁▂▃▄▅▆▇█"  # no blank slot — the minimum value must stay visible


class C:
    use = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    @classmethod
    def set(cls, enabled: bool) -> None:
        cls.use = enabled


def col(code: str) -> str:
    return f"\x1b[{code}m" if C.use else ""


RESET = lambda: col("0")
BOLD = lambda: col("1")
ACCENT = lambda: col("38;5;44")
GOOD = lambda: col("38;5;42")
WARN = lambda: col("38;5;214")
CRIT = lambda: col("1;38;5;203")
MUTED = lambda: col("38;5;244")
INK = lambda: col("38;5;253")


def _char_width(ch: str) -> int:
    # Hangul/CJK render as 2 terminal columns; ambiguous-width block/box
    # glyphs render as 1 in the vast majority of Western terminal fonts.
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def vlen(s: str) -> int:
    return sum(_char_width(c) for c in ANSI_RE.sub("", s))


def pad(s: str, width: int, align: str = "left") -> str:
    n = width - vlen(s)
    if n <= 0:
        return s
    return s + " " * n if align == "left" else " " * n + s


def truncate(s: str, width: int) -> str:
    """Visible-width-aware truncate + ellipsis, for content whose length is
    unbounded (checkpoint lists, grep'd error lines) — never let it stretch
    a box row past the border. Strips ANSI first (plain text only)."""
    plain = ANSI_RE.sub("", s)
    if vlen(plain) <= width or width <= 1:
        return plain[:max(0, width)]
    out, w = [], 0
    for ch in plain:
        cw = _char_width(ch)
        if w + cw > width - 1:
            break
        out.append(ch)
        w += cw
    return "".join(out) + "…"


def read_jsonl(p: Path) -> list[dict]:
    out = []
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def fmt_kst(ts: float) -> str:
    return datetime.fromtimestamp(ts, KST).strftime("%m-%d %H:%M KST")


def fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def sparkline(values: list[float], width: int) -> str:
    """Bucket to `width` points, then scale by the 5th-95th percentile band
    (not raw min/max) so one early outlier (e.g. step-1's miscalibration
    spike) doesn't compress the entire steady-state range into one level."""
    if not values:
        return MUTED() + "·" * width + RESET()
    n = len(values)
    width = min(width, n) if n < width else width
    bucket = n / width
    pts = []
    for i in range(width):
        lo_i = int(i * bucket)
        hi_i = max(lo_i + 1, int((i + 1) * bucket))
        seg = values[lo_i:hi_i]
        pts.append(sum(seg) / len(seg))
    s = sorted(pts)
    m = len(s)
    lo, hi = s[int(m * 0.05)], s[min(m - 1, int(m * 0.95))]
    if hi - lo < 1e-9:
        lo, hi = min(pts), max(pts)
        if hi - lo < 1e-9:
            hi = lo + 1.0
    rng = hi - lo
    chars = []
    for v in pts:
        vv = min(hi, max(lo, v))
        idx = round((vv - lo) / rng * (len(BLOCKS) - 1))
        chars.append(BLOCKS[max(0, min(len(BLOCKS) - 1, idx))])
    return ACCENT() + "".join(chars) + RESET()


def bar(frac: float, width: int) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return ACCENT() + "█" * filled + RESET() + MUTED() + "░" * (width - filled) + RESET()


def trend_arrow(recent: list[float], prior: list[float]) -> str:
    if not recent or not prior:
        return MUTED() + "·" + RESET()
    d = sum(recent) / len(recent) - sum(prior) / len(prior)
    if abs(d) < 1e-3:
        return MUTED() + "→" + RESET()
    return (GOOD() + "↓" if d < 0 else WARN() + "↑") + RESET()


LAUNCH_MARKER_RE = re.compile(r"^\[data\] train \d+ \(", re.MULTILINE)
ERROR_RE = re.compile(r".*(?:Traceback|CUDA error|out of memory|RuntimeError).*")


def has_errors() -> str | None:
    if not SFT_LOG.exists():
        return None
    try:
        text = SFT_LOG.read_text(errors="ignore")
    except Exception:
        return None
    # SFT_LOG is append-only across relaunches (e.g. a resume after a crash
    # rewritten from a different --adapter), so an old crash's traceback can
    # sit before this invocation's own output. Scope the scan to after the
    # most recent launch marker ("[data] train N ...", printed once data
    # loads OK) so a stale failure doesn't get reported as a live one.
    markers = list(LAUNCH_MARKER_RE.finditer(text))
    if markers:
        text = text[markers[-1].start():]
    matches = ERROR_RE.findall(text)
    return matches[-1].strip() if matches else None


def session_alive(proc_pattern: str = "run_fit.py", tag: str | None = None) -> bool:
    try:
        if subprocess.run(
            ["tmux", "has-session", "-t", "snuai11"], capture_output=True, timeout=5
        ).returncode == 0:
            return True
    except Exception:
        pass
    # tmux-less path (e.g. harness-tracked background task): look for a live
    # process (training or inference) writing into this dashboard's target dir.
    try:
        out = subprocess.run(["pgrep", "-af", proc_pattern], capture_output=True, text=True, timeout=5).stdout
        return (tag or RUN.name) in out
    except Exception:
        return False


def box_top(title: str, width: int) -> str:
    label = f" {title} "
    dashes = width - vlen(label) - 2
    return "┌" + label + "─" * max(0, dashes) + "┐"


def box_bottom(width: int) -> str:
    return "└" + "─" * (width - 2) + "┘"


def box_line(content: str, width: int) -> str:
    inner = width - 2
    body = " " + content
    if vlen(body) > inner:  # last-resort guard: any unforeseen overflow gets
        body = " " + truncate(content, inner - 1)  # clipped, not left to break the border
    return "│" + pad(body, inner) + "│"


def box_rule(width: int) -> str:
    return "├" + "─" * (width - 2) + "┤"


def render(width: int) -> str:
    out: list[str] = []
    W = width

    log = read_jsonl(RUN / "train_log.jsonl")
    launch_ts = (RUN / "train_log.jsonl").stat().st_mtime - log[-1]["elapsed_s"] if log else None
    vram_all = read_jsonl(VRAM_LOG)
    # VRAM_LOG is a single session-wide append log shared across runs/resumes;
    # scope it to this run's launch window so "peak" isn't polluted by an
    # earlier (possibly differently-configured) run in the same file.
    vram = [r for r in vram_all if launch_ts is None or r["ts"] >= launch_ts] if vram_all else vram_all
    step = log[-1]["step"] if log else 0
    ce = log[-1].get("ce", log[-1]["loss"]) if log else None
    acc = log[-1]["acc"] if log else None
    err = has_errors()
    alive = session_alive()
    done = (RUN / "adapter_final/adapter/adapter_config.json").exists()
    sweep_dir = ROOT / "runs/sweep_insample"
    sweep = sorted(sweep_dir.glob("*/eval.json")) if sweep_dir.exists() else []
    tests = sorted(ROOT.glob("runs/test_motion_*/submission.csv"))

    if err:
        phase, pcolor = "오류 감지", CRIT
    elif not alive and not done:
        phase, pcolor = "세션 종료(비정상?)", CRIT
    elif not log:
        phase, pcolor = "기동 중", WARN
    elif not done:
        phase, pcolor = "학습 중", ACCENT
    elif tests:
        phase, pcolor = "완료 — 채점 결과 있음", GOOD
    elif sweep:
        phase, pcolor = "인샘플 스윕 중", ACCENT
    else:
        phase, pcolor = "학습 완료 — 스윕 대기", GOOD

    # ---- header -----------------------------------------------------
    title = f"{BOLD()}{ACCENT()}Ver11 모션 블렌드 본런{RESET()}"
    pill = f"{pcolor()}{BOLD()} {phase} {RESET()}"
    out.append(box_top("●", W))
    out.append(box_line(f"{title}   {pill}", W))
    out.append(box_rule(W))

    # ---- stats grid ---------------------------------------------------
    rate_txt, eta_txt = "—", "—"
    if len(log) >= 2:
        a, b = log[max(0, len(log) - 20)], log[-1]
        rate = (b["elapsed_s"] - a["elapsed_s"]) / max(1, b["step"] - a["step"])
        rate_txt = f"{rate:.1f}s/step"
        if not done:
            eta_txt = fmt_kst(time.time() + (TOTAL_STEPS - step) * rate)

    trip_txt, trip_c = "step200 대기", MUTED
    if step >= TRIPWIRE_STEP and ce is not None:
        rec200 = next((r for r in log if r["step"] >= TRIPWIRE_STEP), None)
        c200 = rec200.get("ce", rec200["loss"]) if rec200 else ce
        if c200 < 3.0:
            trip_txt, trip_c = f"PASS (ce {c200:.2f} < 3.0)", GOOD
        else:
            trip_txt, trip_c = f"FAIL! (ce {c200:.2f} >= 3.0)", CRIT

    recent = [r.get("ce", r["loss"]) for r in log[-5:]]
    prior = [r.get("ce", r["loss"]) for r in log[-10:-5]]
    arrow = trend_arrow(recent, prior)

    col1 = [
        ("step", f"{step} / {TOTAL_STEPS}"),
        ("ce (윈도우)", f"{ce:.3f} {arrow}" if ce is not None else "—"),
        ("acc (윈도우)", f"{acc:.3f}" if acc is not None else "—"),
    ]
    col2 = [
        ("속도", rate_txt),
        ("종료 ETA", eta_txt),
        ("트립와이어", f"{trip_c()}{trip_txt}{RESET()}"),
    ]
    half = W // 2 - 2
    if half >= 34:  # two-column grid only when a full value can't overflow it
        for (k1, v1), (k2, v2) in zip(col1, col2):
            left = f"{MUTED()}{pad(k1, 13)}{RESET()}{INK()}{v1}{RESET()}"
            right = f"{MUTED()}{pad(k2, 11)}{RESET()}{INK()}{v2}{RESET()}"
            out.append(box_line(pad(left, half) + " " + right, W))
    else:  # narrow terminal: stack single-column, each line short enough to fit
        for k, v in col1 + col2:
            out.append(box_line(f"{MUTED()}{pad(k, 14)}{RESET()}{INK()}{v}{RESET()}", W))

    out.append(box_line(bar(step / TOTAL_STEPS, W - 14) + f"  {step*100//TOTAL_STEPS:3d}%", W))

    sess_txt = f"{GOOD()}tmux 생존{RESET()}" if alive else f"{CRIT()}tmux 없음{RESET()}"
    err_budget = W - 3 - vlen("tmux 생존") - 3
    err_txt = f"{CRIT()}{truncate(err, err_budget)}{RESET()}" if err else f"{GOOD()}에러 0{RESET()}"
    out.append(box_line(f"{sess_txt}   {err_txt}", W))

    # ---- sparklines -----------------------------------------------------
    if log:
        out.append(box_rule(W))
        sw = W - 14
        ce_series = [r.get("ce", r["loss"]) for r in log]
        acc_series = [r["acc"] for r in log]
        out.append(box_line(f"{MUTED()}ce  {RESET()}" + sparkline(ce_series, sw) +
                             f" {INK()}{ce_series[-1]:.2f}{RESET()}", W))
        out.append(box_line(f"{MUTED()}acc {RESET()}" + sparkline(acc_series, sw) +
                             f" {INK()}{acc_series[-1]:.2f}{RESET()}", W))

    # ---- checkpoints + vram ------------------------------------------
    ckpts = sorted((p.parent.parent.name) for p in RUN.glob("checkpoint-*/adapter/adapter_config.json"))
    if ckpts:
        out.append(box_rule(W))
        ck_text = truncate(", ".join(ckpts), W - 15)
        out.append(box_line(f"{MUTED()}checkpoints{RESET()} {ck_text}", W))

    if vram:
        peak = max(r["mib"] for r in vram) / 1024
        cur = vram[-1]["mib"] / 1024
        vcolor = CRIT if peak > 23 else GOOD
        out.append(box_rule(W))
        out.append(box_line(
            f"{MUTED()}VRAM{RESET()} 현재 {INK()}{cur:.1f}{RESET()} GiB   "
            f"피크 {vcolor()}{peak:.1f}{RESET()} GiB   {MUTED()}(23GiB=3090 기준선){RESET()}",
            W))

    # ---- sweep / test results ------------------------------------------
    if sweep:
        out.append(box_rule(W))
        out.append(box_line(f"{BOLD()}인샘플 스윕{RESET()} (train 300, TTA4+motion0.3)", W))
        for ev in sweep:
            d = json.loads(ev.read_text())
            row = f"  {ev.parent.name:<16} EM {d['em']:.4f}  pw {d['pairwise']:.4f}"
            out.append(box_line(truncate(row, W - 4), W))

    gpath = ROOT / "runs/grade_results.json"
    if gpath.exists():
        g = json.loads(gpath.read_text())
        out.append(box_rule(W))
        out.append(box_line(f"{BOLD()}test 819 채점{RESET()} (key v13 est, 베이스라인 0.9011)", W))
        for k, v in g.items():
            row = f"  {k:<20} EM {v.get('em','—')}  est {v.get('est','—')}"
            out.append(box_line(truncate(row, W - 4), W))

    launch_txt = fmt_kst(launch_ts) if launch_ts is not None else "—"

    out.append(box_rule(W))
    out.append(box_line(f"{MUTED()}갱신 {fmt_kst(time.time())} · launch {launch_txt} · "
                         f"1×A100 · Ctrl-C로 종료{RESET()}", W))
    out.append(box_bottom(W))
    return "\n".join(out)


def render_infer(width: int, infer_dir: Path, total: int) -> str:
    """Test/train inference progress — reads <infer_dir>/progress.jsonl
    (one line appended per completed sample, resumable/idempotent — see
    infer.py). Separate from render() since a serving run has no
    steps/LR/checkpoints, only a sample counter and per-sample cost."""
    out: list[str] = []
    W = width

    recs = read_jsonl(infer_dir / "progress.jsonl")
    done = len(recs)
    launch_ts = (infer_dir / "config.json").stat().st_mtime if (infer_dir / "config.json").exists() else None
    vram_all = read_jsonl(VRAM_LOG)
    vram = [r for r in vram_all if launch_ts is None or r["ts"] >= launch_ts] if vram_all else vram_all
    err = has_errors()
    alive = session_alive(proc_pattern="snuai11.infer", tag=infer_dir.name)
    sub_done = (infer_dir / "submission.csv").exists()

    if err:
        phase, pcolor = "오류 감지", CRIT
    elif sub_done:
        phase, pcolor = "완료 — submission.csv 있음", GOOD
    elif not alive and done < total:
        phase, pcolor = "세션 종료(비정상?)", CRIT
    elif not recs:
        phase, pcolor = "기동 중", WARN
    else:
        phase, pcolor = "추론 중", ACCENT

    title = f"{BOLD()}{ACCENT()}Ver11 test 추론{RESET()} {MUTED()}{infer_dir.name}{RESET()}"
    pill = f"{pcolor()}{BOLD()} {phase} {RESET()}"
    out.append(box_top("●", W))
    out.append(box_line(f"{title}   {pill}", W))
    out.append(box_rule(W))

    rate_txt, eta_txt = "—", "—"
    tail = recs[-20:]
    if tail:
        rate = sum(r.get("elapsed_s", 0.0) for r in tail) / len(tail)
        rate_txt = f"{rate:.1f}s/샘플"
        if not sub_done:
            eta_txt = fmt_kst(time.time() + (total - done) * rate)
    n_esc = sum(1 for r in recs if r.get("escalated"))

    cfg = json.loads((infer_dir / "config.json").read_text()) if (infer_dir / "config.json").exists() else {}
    adapter_path = Path(cfg["adapter"]) if cfg.get("adapter") else None
    # adapter dirs are always named "adapter" (the parent, e.g. checkpoint-1200
    # or adapter_final, is the identifying part) — show that instead.
    adapter_txt = truncate(adapter_path.parent.name, 20) if adapter_path else "—"

    col1 = [
        ("샘플", f"{done} / {total}"),
        ("속도", rate_txt),
        ("이스컬레이션", f"{n_esc} ({n_esc*100//max(1, done)}%)" if done else "—"),
    ]
    col2 = [
        ("종료 ETA", eta_txt),
        ("어댑터", adapter_txt),
        ("TTA", str(cfg.get("tta", "—"))),
    ]
    half = W // 2 - 2
    if half >= 34:
        for (k1, v1), (k2, v2) in zip(col1, col2):
            left = f"{MUTED()}{pad(k1, 13)}{RESET()}{INK()}{v1}{RESET()}"
            right = f"{MUTED()}{pad(k2, 11)}{RESET()}{INK()}{v2}{RESET()}"
            out.append(box_line(pad(left, half) + " " + right, W))
    else:
        for k, v in col1 + col2:
            out.append(box_line(f"{MUTED()}{pad(k, 14)}{RESET()}{INK()}{v}{RESET()}", W))

    out.append(box_line(bar(done / max(1, total), W - 14) + f"  {done*100//max(1, total):3d}%", W))

    sess_txt = f"{GOOD()}세션 생존{RESET()}" if alive else f"{CRIT()}세션 없음{RESET()}"
    err_budget = W - 3 - vlen("세션 생존") - 3
    err_txt = f"{CRIT()}{truncate(err, err_budget)}{RESET()}" if err else f"{GOOD()}에러 0{RESET()}"
    out.append(box_line(f"{sess_txt}   {err_txt}", W))

    if recs:
        out.append(box_rule(W))
        margins = [r.get("margin_final", r.get("margin", 0.0)) for r in recs]
        out.append(box_line(f"{MUTED()}margin{RESET()} " + sparkline(margins, W - 17) +
                             f" {INK()}{margins[-1]:.2f}{RESET()}", W))
        out.append(box_line(f"{MUTED()}최근 샘플{RESET()} " + ", ".join(
            f"{r['id']}({r.get('margin_final', r.get('margin', 0.0)):.2f})" for r in recs[-3:]), W))

    if vram:
        peak = max(r["mib"] for r in vram) / 1024
        cur = vram[-1]["mib"] / 1024
        vcolor = CRIT if peak > 23 else GOOD
        out.append(box_rule(W))
        out.append(box_line(
            f"{MUTED()}VRAM{RESET()} 현재 {INK()}{cur:.1f}{RESET()} GiB   "
            f"피크 {vcolor()}{peak:.1f}{RESET()} GiB   {MUTED()}(23GiB=3090 기준선){RESET()}",
            W))

    launch_txt = fmt_kst(launch_ts) if launch_ts is not None else "—"
    out.append(box_rule(W))
    out.append(box_line(f"{MUTED()}갱신 {fmt_kst(time.time())} · launch {launch_txt} · "
                         f"1×A100 · Ctrl-C로 종료{RESET()}", W))
    out.append(box_bottom(W))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", nargs="?", const=20, type=int, default=None, metavar="SECONDS")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--run", default=None, help="run dir, relative to repo root (default: runs/sft32b_v11_ws8)")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--tripwire-step", type=int, default=None)
    ap.add_argument("--log", default=None, help="sft stdout log, relative to repo root (for the error scan)")
    ap.add_argument("--infer-out", default=None,
                     help="switch to inference-progress mode: a runs/... dir with progress.jsonl "
                          "(e.g. runs/test_v11_ckpt1200), instead of the training dashboard")
    ap.add_argument("--infer-total", type=int, default=819, help="sample count for --infer-out (819 = test split)")
    args = ap.parse_args()
    if args.no_color:
        C.set(False)

    global RUN, SFT_LOG, TOTAL_STEPS, TRIPWIRE_STEP
    if args.run:
        RUN = ROOT / args.run
    if args.total_steps:
        TOTAL_STEPS = args.total_steps
    if args.tripwire_step:
        TRIPWIRE_STEP = args.tripwire_step
    if args.log:
        SFT_LOG = ROOT / args.log

    width = min(max(shutil.get_terminal_size((100, 24)).columns, 64), 96)

    if args.infer_out:
        infer_dir = ROOT / args.infer_out
        if not args.log:  # e.g. runs/test_v11_ckpt1200 -> runs/test_v11_ckpt1200.log
            SFT_LOG = infer_dir.with_suffix(".log")
        render_fn = lambda w: render_infer(w, infer_dir, args.infer_total)  # noqa: E731
    else:
        render_fn = render

    if args.watch is None:
        print(render_fn(width))
        return

    # --watch is an explicit request to redraw in place, independent of
    # whether Python's isatty() auto-detect (C.use) sees a real tty — some
    # execution contexts (piped/captured shells) report isatty()=False even
    # though the bytes do reach a real terminal, which otherwise silently
    # skips every clear code and makes frames stack instead of overwrite.
    # --no-color is the explicit opt-out (e.g. logging to a file) that
    # should still suppress control codes.
    redraw = not args.no_color

    def ctrl(seq: str) -> None:
        if redraw:
            sys.stdout.write(seq)

    ctrl("\x1b[?1049h\x1b[?25l")  # alternate screen + hide cursor (htop/less style)
    try:
        while True:
            ctrl("\x1b[H\x1b[J")  # cursor home, erase to end — redraw in place
            print(render_fn(width))
            sys.stdout.flush()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl("\x1b[?25h\x1b[?1049l")  # always restore the caller's screen
        sys.stdout.flush()


if __name__ == "__main__":
    main()
