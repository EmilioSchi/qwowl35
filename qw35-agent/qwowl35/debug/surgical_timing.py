"""Surgical per-request timing runner for the qwowl35 agent.

A focused variant of ``headless_steps.py``: drives ONE task (e.g. cal_task.md)
through a persistent agent session and captures the server's per-request
``qw35_timings`` for EVERY internal request the tool-calling loop makes. The
goal is to localize a long-session decode slowdown to a phase — prefill
(``prompt_eval_*``), decode (``eval_*``), state reset (``reset_ms``), or a
session-cache miss (``session_path`` flipping away from ``extend``) — as the
conversation context grows.

Usage:
    python qwowl35/debug/surgical_timing.py \
        --task-file benchmark/cal_task.md --timeout 360 \
        --out /tmp/timing.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for _p in (_PKG, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent import Agent  # noqa: E402
from config import load_config  # noqa: E402
from tools_registry import ToolRegistry  # noqa: E402
from headless import (  # noqa: E402
    DebugClient,
    LoggingUI,
    Transcript,
    _make_approver,
    _resolve_workdir,
)


# Columns pulled from each request's usage + qw35_timings.
_FIELDS = [
    "req", "wall_s", "ctx_tokens", "cached_tokens", "session_path",
    "prefill_tokens", "prefill_tps", "prefill_ms", "prefill_path",
    "decode_tokens", "decode_tps", "decode_ms",
    "reset_ms", "lock_ms", "total_ms",
]


class TimingUI(LoggingUI):
    """LoggingUI that also captures one timing row per server request."""

    def __init__(self, transcript: Transcript, t0: float) -> None:
        super().__init__(transcript)
        self.rows: list[dict] = []
        self._t0 = t0

    def set_usage(self, usage, timings) -> None:
        super().set_usage(usage, timings)
        usage = usage or {}
        timings = timings or {}
        details = usage.get("prompt_tokens_details") or {}
        row = {
            "req": len(self.rows) + 1,
            "wall_s": round(time.monotonic() - self._t0, 1),
            "ctx_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens", timings.get("cached_prompt_tokens")),
            "session_path": timings.get("session_path"),
            "prefill_tokens": timings.get("prompt_eval_count"),
            "prefill_tps": _r(timings.get("prompt_eval_tps")),
            "prefill_ms": _r(timings.get("prompt_eval_ms")),
            "prefill_path": timings.get("prefill_path"),
            "decode_tokens": timings.get("eval_count"),
            "decode_tps": _r(timings.get("eval_tps")),
            "decode_ms": _r(timings.get("eval_ms")),
            "reset_ms": _r(timings.get("reset_ms")),
            "lock_ms": _r(timings.get("runtime_lock_ms")),
            "total_ms": _r(timings.get("total_ms")),
        }
        self.rows.append(row)
        print(
            f"  req {row['req']:>3} t={row['wall_s']:>5}s ctx={_s(row['ctx_tokens']):>6} "
            f"cached={_s(row['cached_tokens']):>6} {_s(row['session_path']):<10} "
            f"prefill={_s(row['prefill_tokens']):>5}@{_s(row['prefill_tps']):>6}tps "
            f"decode={_s(row['decode_tokens']):>4}@{_s(row['decode_tps']):>6}tps "
            f"reset={_s(row['reset_ms']):>6}ms total={_s(row['total_ms']):>7}ms",
            flush=True,
        )


def _r(v):
    return round(v, 2) if isinstance(v, (int, float)) else v


def _s(v):
    return "-" if v is None else v


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Surgical per-request timing for qwowl35.")
    p.add_argument("--task-file", required=True, help="markdown task prompt (run as one turn)")
    p.add_argument("--base-url", help="qw35-server base URL (default http://127.0.0.1:8080)")
    p.add_argument("--think", choices=["auto", "on", "off"])
    p.add_argument("--reasoning-effort")
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--workdir")
    p.add_argument("--timeout", type=float, default=360.0, help="max seconds for the task")
    p.add_argument("--out", help="CSV path for the per-request timing rows")
    return p.parse_args(argv)


async def _run(args) -> int:
    task_path = Path(args.task_file)
    if not task_path.exists():
        print(f"error: task file not found: {task_path}", file=sys.stderr)
        return 2
    task = task_path.read_text(encoding="utf-8").strip()

    cfg = load_config(
        base_url=args.base_url, think=args.think,
        reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens,
    )
    run_dir = _resolve_workdir(args.workdir, task_path.stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(run_dir / "transcript.jsonl")
    client = DebugClient(cfg.base_url, cfg.request_timeout, None)
    registry = ToolRegistry(approval=_make_approver(transcript), restricted_bash=False)

    try:
        await client.health()
    except Exception as exc:  # noqa: BLE001
        print(f"error: server not reachable at {cfg.base_url}: {exc}", file=sys.stderr)
        await client.aclose()
        return 2

    t0 = time.monotonic()
    ui = TimingUI(transcript, t0)
    print(f"=== surgical timing: {task_path.name} (timeout {args.timeout}s) ===")
    print(f"  base_url={cfg.base_url} max_tokens={cfg.max_tokens}")
    prev_cwd = os.getcwd()
    os.chdir(run_dir)
    agent = Agent(client, registry, cfg, ui)
    try:
        try:
            await asyncio.wait_for(agent.run_turn(task), timeout=args.timeout)
        except asyncio.TimeoutError:
            print(f"  (task hit the {args.timeout}s cap — analyzing what ran)", flush=True)
    finally:
        os.chdir(prev_cwd)
        await client.aclose()
        transcript.close()

    rows = ui.rows
    if not rows:
        print("no requests captured", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else (run_dir / "timing.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)

    _analyze(rows, out)
    return 0


def _analyze(rows, out) -> None:
    def num(v):
        return v if isinstance(v, (int, float)) else None

    first = rows[0]
    last = rows[-1]
    pre_ms = sum(num(r["prefill_ms"]) or 0 for r in rows)
    dec_ms = sum(num(r["decode_ms"]) or 0 for r in rows)
    rst_ms = sum(num(r["reset_ms"]) or 0 for r in rows)
    tot_ms = sum(num(r["total_ms"]) or 0 for r in rows)
    resets = [r for r in rows if r["session_path"] not in ("extend", "checkpoint", None)]

    print("\n=== PHASE BREAKDOWN (sum across all requests) ===")
    print(f"  requests:        {len(rows)}")
    print(f"  prefill total:   {pre_ms/1000:8.1f}s  ({_pct(pre_ms, tot_ms)})")
    print(f"  decode  total:   {dec_ms/1000:8.1f}s  ({_pct(dec_ms, tot_ms)})")
    print(f"  reset   total:   {rst_ms/1000:8.1f}s  ({_pct(rst_ms, tot_ms)})")
    print(f"  accounted total: {tot_ms/1000:8.1f}s")
    print("\n=== TREND (first vs last request) ===")
    print(f"  ctx:         {_s(first['ctx_tokens'])} -> {_s(last['ctx_tokens'])} tokens")
    print(f"  decode tps:  {_s(first['decode_tps'])} -> {_s(last['decode_tps'])}")
    print(f"  prefill tok: {_s(first['prefill_tokens'])} -> {_s(last['prefill_tokens'])} (per request)")
    print(f"  prefill tps: {_s(first['prefill_tps'])} -> {_s(last['prefill_tps'])}")
    print(f"  session-cache MISSES (full re-prefill): {len(resets)} / {len(rows)} requests")
    if resets:
        print("    miss requests (req#, ctx, prefill_tokens):")
        for r in resets[:20]:
            print(f"      req {r['req']}: ctx={_s(r['ctx_tokens'])} reprefilled={_s(r['prefill_tokens'])}")
    print(f"\n  per-request CSV: {out}")


def _pct(part, whole):
    return f"{100*part/whole:4.1f}%" if whole else "  n/a"


def main(argv=None) -> int:
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
