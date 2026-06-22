#!/usr/bin/env python3
"""Condense a qwowl35 headless transcript.jsonl into a diagnosis-friendly view.

The raw transcript contains `raw_chunk`/`prefill`/`state` spam that floods a
reading context. This prints only the agent-meaningful events (reasoning,
assistant, tool calls + results, errors, verdict) plus aggregate stats: total
wall time, token usage, tool-call counts, and how many edits failed.

Usage:
    python3 summarize_run.py <run_dir_or_transcript.jsonl> [--full-errors] [--max N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _t(s: str, n: int) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"…(+{len(s)-n})"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="run dir or transcript.jsonl")
    ap.add_argument("--max", type=int, default=240, help="truncate field chars (errors always full)")
    ap.add_argument("--full-errors", action="store_true", default=True)
    args = ap.parse_args(argv)

    p = Path(args.path)
    if p.is_dir():
        p = p / "transcript.jsonl"
    if not p.exists():
        print(f"no transcript at {p}", file=sys.stderr)
        return 2

    last_t = 0.0
    tool_calls: Counter = Counter()
    edit_errors = 0
    tool_results = 0
    usage = None
    verdict = None
    files_created = []

    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        k = r.get("kind")
        t = r.get("t", 0.0)
        last_t = max(last_t, t)
        if k in ("raw_chunk", "prefill", "state"):
            continue
        if k == "meta":
            print(f"[meta] timeout={r.get('timeout')}s think={r.get('think')} "
                  f"reasoning_effort={r.get('reasoning_effort')} max_tokens={r.get('max_tokens')}")
        elif k == "reasoning":
            print(f"\n[{t:6.1f} think] {_t(r.get('text',''), args.max)}")
        elif k == "assistant":
            print(f"\n[{t:6.1f} assistant] {_t(r.get('text',''), args.max)}")
        elif k == "tool_call":
            name = r.get("name", "?")
            tool_calls[name] += 1
            print(f"[{t:6.1f} CALL {name}] {_t(r.get('arguments',''), args.max)}")
        elif k == "tool_result":
            tool_results += 1
            err = r.get("is_error")
            if err:
                edit_errors += 1
            tag = "ERR" if err else "ok"
            res = r.get("result", "")
            shown = res if (err and args.full_errors) else _t(res, args.max)
            print(f"[{t:6.1f} {tag:3} {r.get('name','?')}] {shown}")
        elif k == "usage":
            usage = r.get("usage")
        elif k in ("stream_error", "error", "warning", "timeout"):
            print(f"\n[{t:6.1f} {k.upper()}] {r}")
        elif k == "verdict":
            verdict = r.get("verdict")

    print("\n" + "=" * 60)
    print(f"VERDICT       : {verdict}")
    print(f"WALL TIME     : {last_t:.1f}s")
    print(f"TOOL CALLS    : {dict(tool_calls)}  (total {sum(tool_calls.values())})")
    print(f"TOOL RESULTS  : {tool_results}, of which ERRORS: {edit_errors}")
    print(f"USAGE         : {usage}")
    rd = p.parent
    created = sorted(x.name for x in rd.iterdir()
                     if x.name not in {"transcript.jsonl", "messages.json"})
    print(f"FILES CREATED : {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
