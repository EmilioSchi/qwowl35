"""Multi-step headless runner for the qwowl35 agent.

Unlike ``headless.py`` (one task, one turn), this drives several *steps* through
ONE persistent agent session — same conversation and same working directory — so
we can test incremental editing: step 1 creates a file, later
steps must MODIFY it with the edit tool rather than rewriting it.

After each step it records which file tool calls were made (and whether they were
``edit`` vs ``write``), snapshots the target file, and at the end prints a STEP
SUMMARY plus a GOAL verdict (steps after the first must use edit, not a full
rewrite, and the final script must run).

Usage:
    python qwowl35/debug/headless_steps.py \
        --steps-file benchmark/solve_real_root_steps.md \
        --target solve_real_root.py --timeout 360
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # the qwowl35 package dir
for _p in (_PKG, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent import Agent  # noqa: E402
from config import load_config  # noqa: E402
from tools_registry import ToolRegistry  # noqa: E402

# Reuse the plumbing from the single-turn runner.
from headless import (  # noqa: E402
    DebugClient,
    LoggingUI,
    Transcript,
    _make_approver,
    _resolve_workdir,
)


def _split_steps(text: str) -> list[str]:
    """Steps are separated by a line containing only '---'."""
    steps: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if line.strip() == "---":
            steps.append("\n".join(buf).strip())
            buf = []
        else:
            buf.append(line)
    if buf:
        steps.append("\n".join(buf).strip())
    return [s for s in steps if s]


def _tool_calls_in(messages: list[dict], start: int) -> list[dict]:
    """Extract (name, args, result_excerpt) for tool calls since index ``start``."""
    calls: list[dict] = []
    pending: dict[str, dict] = {}
    for msg in messages[start:]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                rec = {"id": tc.get("id"), "name": fn.get("name"), "args": args, "result": ""}
                calls.append(rec)
                if tc.get("id"):
                    pending[tc["id"]] = rec
        elif msg.get("role") == "tool":
            rec = pending.get(msg.get("tool_call_id"))
            if rec is not None:
                content = msg.get("content")
                rec["result"] = content if isinstance(content, str) else str(content)
    return calls


EDIT_TOOLS = {
    "edit",
    "insert",
    "delete",
}
_EDIT_OK_PREFIXES = ("Edited", "Inserted", "Deleted")


def _classify(calls: list[dict], target: str) -> dict:
    """Summarize file-tool usage for one step relative to the target file."""
    edits_ok = writes_ok = edits_fail = 0
    used_write_on_target = used_edit_on_target = bash_override = False
    for c in calls:
        name = c["name"]
        res = c["result"] or ""
        first = res.splitlines()[0] if res else ""
        if name == "bash":
            cmd = str(c["args"].get("command") or "")
            if target in cmd and ("rm " in cmd or ">" in cmd or "tee " in cmd or "mv " in cmd):
                used_write_on_target = True
                writes_ok += 1
            continue
        path = c["args"].get("file") or c["args"].get("path") or ""
        on_target = (path == target) or (target in path) or path == ""  # path may be omitted (last-file)
        if name in EDIT_TOOLS:
            if first.startswith(_EDIT_OK_PREFIXES):
                edits_ok += 1
                used_edit_on_target = used_edit_on_target or on_target
            else:
                edits_fail += 1
    return {
        "edits_ok": edits_ok,
        "edits_fail": edits_fail,
        "writes_ok": writes_ok,
        "used_edit_on_target": used_edit_on_target,
        "used_write_on_target": used_write_on_target,
        "bash_override": bash_override,
        "total_tool_calls": len(calls),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-step headless runner for qwowl35.")
    p.add_argument("--steps-file", required=True, help="markdown file; steps separated by a '---' line")
    p.add_argument("--target", required=True, help="the file the steps build up (e.g. solve_real_root.py)")
    p.add_argument("--base-url", help="qw35-server base URL (default http://127.0.0.1:8080)")
    p.add_argument("--think", choices=["auto", "on", "off"], help="thinking mode")
    p.add_argument("--reasoning-effort", help="reasoning effort hint")
    p.add_argument("--max-tokens", type=int, help="per-request completion token cap")
    p.add_argument("--workdir", help="scratch dir (default qw35-debug-runs/<steps>-NNN)")
    p.add_argument("--timeout", type=float, default=360.0, help="max seconds PER STEP (default 360)")
    p.add_argument("--run-cmd", help="shell command to run the target at the end (default: python <target>)")
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    steps_path = Path(args.steps_file)
    if not steps_path.exists():
        print(f"error: steps file not found: {steps_path}", file=sys.stderr)
        return 2
    steps = _split_steps(steps_path.read_text(encoding="utf-8"))
    if not steps:
        print("error: no steps found", file=sys.stderr)
        return 2

    cfg = load_config(
        base_url=args.base_url,
        think=args.think,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
    )
    run_dir = _resolve_workdir(args.workdir, steps_path.stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(run_dir / "transcript.jsonl")
    transcript.record(
        "meta",
        steps=steps,
        target=args.target,
        workdir=str(run_dir),
        base_url=cfg.base_url,
        max_tokens=cfg.max_tokens,
        timeout=args.timeout,
    )

    raw_sink = lambda data: transcript.record("raw_chunk", data=data)  # noqa: E731
    client = DebugClient(cfg.base_url, cfg.request_timeout, raw_sink)
    registry = ToolRegistry(approval=_make_approver(transcript), restricted_bash=False)
    ui = LoggingUI(transcript)

    try:
        await client.health()
    except Exception as exc:  # noqa: BLE001
        print(f"error: server not reachable at {cfg.base_url}: {exc}", file=sys.stderr)
        await client.aclose()
        return 2

    print("=== qwowl35 multi-step run ===")
    prev_cwd = os.getcwd()
    os.chdir(run_dir)
    agent = Agent(client, registry, cfg, ui)
    target = args.target
    step_reports: list[dict] = []
    try:
        for i, step in enumerate(steps, 1):
            print(f"\n----- STEP {i}/{len(steps)}: {step[:80]} -----")
            transcript.record("step_begin", index=i, prompt=step)
            start = len(agent.messages)
            verdict = "finished"
            try:
                ok = await asyncio.wait_for(agent.run_turn(step), timeout=args.timeout)
                if not ok:
                    verdict = "stream-error"
            except asyncio.TimeoutError:
                verdict = "timed-out"
                transcript.record("timeout", step=i, seconds=args.timeout)
            calls = _tool_calls_in(agent.messages, start)
            summary = _classify(calls, target)
            summary["step"] = i
            summary["verdict"] = verdict
            # Snapshot the target after this step.
            snap = Path(target)
            if snap.exists():
                (run_dir / f"{Path(target).stem}.step{i}{Path(target).suffix}").write_text(snap.read_text(), encoding="utf-8")
                summary["target_lines"] = len(snap.read_text().splitlines())
            else:
                summary["target_lines"] = None
            transcript.record("step_summary", **summary)
            step_reports.append(summary)
            print(f"  tool calls: {summary['total_tool_calls']}  edits_ok={summary['edits_ok']} "
                  f"edits_fail={summary['edits_fail']} writes_ok={summary['writes_ok']} "
                  f"target_lines={summary['target_lines']} verdict={verdict}")
    finally:
        run_output = ""
        target_exists = Path(target).exists()
        if target_exists:
            cmd = args.run_cmd or f"{sys.executable} {target}"
            try:
                proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
                run_output = f"exit={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            except Exception as exc:  # noqa: BLE001
                run_output = f"run failed: {exc}"
        pngs = sorted(p.name for p in Path(".").glob("*.png")) if target_exists else []
        src = Path(target).read_text(encoding="utf-8") if target_exists else ""
        os.chdir(prev_cwd)
        (run_dir / "messages.json").write_text(json.dumps(agent.messages, indent=2, ensure_ascii=False), encoding="utf-8")
        await client.aclose()
        transcript.record("final", run_output=run_output, pngs=pngs)
        transcript.close()

    # ---- GOAL verdict -------------------------------------------------------
    incremental_ok = all(
        r["used_edit_on_target"] and not r["used_write_on_target"] and not r.get("bash_override")
        for r in step_reports[1:]  # steps after the first must be edits, not rewrites
    ) and len(step_reports) >= 3
    roots_ok = ("-9" in run_output and "11" in run_output)
    png_ok = bool(pngs)
    s = src.lower()
    green_ok = any(tok in s for tok in ("green", "'go'", '"go"', "color='g'", 'color="g"', "='g'", '="g"'))
    gray_ok = ("gray" in s) or ("grey" in s)
    colors_ok = green_ok and gray_ok

    print("\n=== STEP SUMMARY ===")
    for r in step_reports:
        kind = "EDIT" if r["used_edit_on_target"] else ("WRITE" if r["used_write_on_target"] else "-")
        print(f"  step {r['step']}: {kind}  edits_ok={r['edits_ok']} edits_fail={r['edits_fail']} "
              f"writes_ok={r['writes_ok']} lines={r['target_lines']} ({r['verdict']})")
    print("\n=== GOAL CHECKS ===")
    print(f"  steps 2+ used incremental edit (not rewrite): {incremental_ok}")
    print(f"  script runs and prints roots -9 and 11:        {roots_ok}")
    print(f"  png generated:                                 {png_ok} {pngs}")
    print(f"  colors applied (green points, gray bg):        {colors_ok}")
    goal = incremental_ok and roots_ok and png_ok and colors_ok
    print(f"\n=== GOAL REACHED: {goal} ===")
    print(f"run dir: {run_dir}")
    print("--- final run output ---")
    print(run_output[:1500])
    return 0 if goal else 1


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
