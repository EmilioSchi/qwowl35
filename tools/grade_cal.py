#!/usr/bin/env python3
"""Independent oracle for the cal_task benchmark.

The agent writes its own tests, which it could make trivially pass. This grader
ignores the agent's tests entirely: it runs the agent's program and exact-string-
compares its stdout against the system `cal` for the current month.

Goal-reached (binary) = the agent's program, run with no arguments, reproduces
`cal` byte-for-byte (including trailing whitespace and blank lines).

Resolution of the agent's program (in order):
  1. --cmd "<shell command>"  — explicit, most reliable for CI/Stage-3.
  2. auto-detect inside --dir: a python entrypoint named cal*.py / main.py, else
     the most recently modified *.py; run with `python3 <file>`.

Usage:
    python3 grade_cal.py --dir <run_dir>
    python3 grade_cal.py --cmd "python3 src/cal.py" --dir <run_dir>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def system_cal() -> str:
    # No args = current month. Pin LANG so weekday/month names are stable.
    return subprocess.run(
        ["cal"], capture_output=True, text=True, env={"LANG": "C", "TERM": "dumb"}
    ).stdout


def find_entry(run_dir: Path) -> list[str] | None:
    cands = list(run_dir.rglob("*.py"))
    cands = [c for c in cands if "test" not in c.name.lower()]
    if not cands:
        return None
    # prefer cal*.py / main.py, else newest
    pref = [c for c in cands if c.name.lower().startswith("cal") or c.name == "main.py"]
    chosen = (pref or sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True))[0]
    return ["python3", str(chosen)]


def run_candidate(cmd: list[str], cwd: Path) -> tuple[str, str, int]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=20, env={"LANG": "C", "TERM": "dumb"})
        return p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1
    except Exception as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {e}", -1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="agent run dir")
    ap.add_argument("--cmd", help="explicit run command (overrides auto-detect)")
    args = ap.parse_args(argv)

    run_dir = Path(args.dir).absolute()
    if not run_dir.is_dir():
        print(f"no run dir: {run_dir}", file=sys.stderr)
        return 2

    expected = system_cal()
    cmd = args.cmd.split() if args.cmd else find_entry(run_dir)
    if not cmd:
        print("RESULT: FAIL (no runnable program found)")
        return 1

    out, err, rc = run_candidate(cmd, run_dir)
    ok = out == expected
    print(f"program     : {' '.join(cmd)}")
    print(f"exit code   : {rc}")
    print(f"exact match : {ok}")
    if not ok:
        print("--- expected (system cal) ---")
        print(repr(expected))
        print("--- got ---")
        print(repr(out))
        if err.strip():
            print("--- stderr ---")
            print(err[:500])
        # first differing line for a quick read
        el, ol = expected.splitlines(), out.splitlines()
        for i in range(max(len(el), len(ol))):
            e = el[i] if i < len(el) else "<missing>"
            o = ol[i] if i < len(ol) else "<missing>"
            if e != o:
                print(f"first diff @ line {i}: expected {e!r} got {o!r}")
                break
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
