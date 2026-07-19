#!/usr/bin/env python3
"""Headless (non-TUI) debug runner for the qwowl35 coding agent.

This drives the *exact same* routines as the Textual app — the :class:`Agent`
loop, :class:`Qw35Client`, :class:`ToolRegistry`, the ``file``/``bash`` tools,
:class:`Config`, and the system prompt — but non-interactively, so we can see
what actually happens during a session and find out why a small model fails to
create and edit files.

It runs one task to completion (or a timeout), inside an isolated scratch dir,
and writes a full debug transcript:

* ``transcript.jsonl`` — one timestamped record per event: the raw SSE chunks the
  model emitted (``raw_chunk``), reasoning, assistant text, every tool call with
  its arguments, and every tool result with its *full, untruncated* text
  (including edit-tool error messages) and ``is_error`` flag.
* ``messages.json`` — the exact conversation the model saw (``agent.messages``),
  which confirms whether tool errors were fed back to inference and how they read.

Run::

    python qwowl35/debug/headless.py --task benchmark/cal_task.md --timeout 300
    python qwowl35/debug/headless.py --prompt "write hello.py" --restricted-bash

The package modules import each other by bare name (the flat sys.path quirk), so
this file (living in ``qwowl35/debug/``) puts the ``qwowl35`` package dir — its
parent — on ``sys.path`` before importing them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # the qwowl35 package dir
for _p in (_PKG, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent import Agent  # noqa: E402
from approval import ApprovalDecision  # noqa: E402
from client import Qw35Client  # noqa: E402
from config import load_config  # noqa: E402
from sessions.transcript import TranscriptWriter  # noqa: E402
from tools_registry import ToolRegistry  # noqa: E402

# Default task: the cal.py benchmark shipped alongside the repo (qw35-agent/benchmark).
DEFAULT_TASK = Path(_PKG).parent / "benchmark" / "cal_task.md"

# stdout echo truncates long fields; the JSONL file always keeps them in full.
ECHO_TRUNCATE = 2000
# These record kinds are written to the file but not echoed to stdout (too noisy).
_QUIET_KINDS = {"raw_chunk", "prefill", "state", "request"}


def _truncate(text: str, limit: int = ECHO_TRUNCATE) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"… (+{len(text) - limit} chars, see transcript)"


class Transcript(TranscriptWriter):
    """The shared session JSONL writer, mirrored to stdout in human-readable
    form for live debug runs."""

    def __init__(self, path: Path, echo: bool = True) -> None:
        super().__init__(path)
        self._echo = echo

    def record(self, kind: str, **fields) -> None:
        super().record(kind, **fields)
        if self._echo and kind not in _QUIET_KINDS:
            self._echo_human(kind, fields)

    def _echo_human(self, kind: str, fields: dict) -> None:
        if kind == "meta":
            print(f"  task: {_truncate(fields.get('task', ''), 400)}")
            print(f"  workdir: {fields.get('workdir')}")
            print(
                f"  base_url={fields.get('base_url')} think={fields.get('think')} "
                f"reasoning_effort={fields.get('reasoning_effort')} "
                f"restricted_bash={fields.get('restricted_bash')} timeout={fields.get('timeout')}s"
            )
        elif kind == "user":
            print(f"\n>>> user: {_truncate(fields.get('text', ''))}")
        elif kind == "reasoning":
            print(f"\n[think]\n{_truncate(fields.get('text', ''))}")
        elif kind == "assistant":
            print(f"\n[assistant]\n{_truncate(fields.get('text', ''))}")
        elif kind == "tool_call":
            print(f"\n[tool_call #{fields.get('index')}] {fields.get('name')} "
                  f"{_truncate(fields.get('arguments', ''))}")
        elif kind == "tool_result":
            tag = "ERR" if fields.get("is_error") else "ok"
            print(f"[tool_result #{fields.get('index')} {fields.get('name')} :: {tag}]\n"
                  f"{_truncate(fields.get('result', ''))}")
        elif kind == "bash_approval":
            print(f"[bash approved] {_truncate(fields.get('command', ''), 400)}")
        elif kind == "stream_error":
            print(f"\n!! stream error {fields.get('code')}: {fields.get('message')}")
        elif kind == "warning":
            print(f"\n!! warning: {_truncate(fields.get('text', ''))}")
        elif kind == "error":
            print(f"\n!! {_truncate(fields.get('text', ''))}")
        elif kind == "timeout":
            print(f"\n!! TIMEOUT after {fields.get('seconds')}s")
        elif kind == "gate":
            mark = "USEFUL" if fields.get("kept") else "NOT USEFUL"
            subject = fields.get("subject") or ""
            label = f" [{subject}]" if subject else ""
            print(f"[{mark}]{label} {_truncate(fields.get('meaning', ''), 200)}")
        elif kind == "findings":
            rows = fields.get("findings") or []
            print(f"[FINDINGS] {len(rows)} kept, {fields.get('dropped', 0)} discarded")
            for subject, meaning in rows:
                print(f"  + {subject} — {_truncate(meaning, 160)}")
        elif kind == "verdict":
            print(f"\n=== verdict: {fields.get('verdict')} ===")


class LoggingChat:
    """Implements the ``chat`` half of the AgentUI contract by logging events.

    Reasoning/assistant deltas are buffered and emitted whole on ``flush_*``;
    tool-call argument fragments are accumulated per index during the stream and
    paired with the result when the loop reports it.
    """

    def __init__(self, transcript: Transcript) -> None:
        self.t = transcript
        self._reasoning: list[str] = []
        self._assistant: list[str] = []
        self._tool_calls: dict[int, dict] = {}

    def add_user(self, text: str) -> None:
        self.t.record("user", text=text)

    def add_reasoning_chunk(self, text: str) -> None:
        self._reasoning.append(text)

    def flush_reasoning(self) -> None:
        if self._reasoning:
            self.t.record("reasoning", text="".join(self._reasoning))
            self._reasoning = []

    def add_assistant_chunk(self, text: str) -> None:
        self._assistant.append(text)

    def flush_assistant(self) -> None:
        if self._assistant:
            self.t.record("assistant", text="".join(self._assistant))
            self._assistant = []

    def begin_tool_call(self, index: int, name: str) -> None:
        self._tool_calls[index] = {"name": name, "args": []}

    def update_tool_call(self, index: int, fragment: str) -> None:
        entry = self._tool_calls.setdefault(index, {"name": "", "args": []})
        entry["args"].append(fragment)

    def name_tool_call(self, index: int, name: str) -> None:
        entry = self._tool_calls.setdefault(index, {"name": "", "args": []})
        entry["name"] = name

    def finalize_tool_call(self, index: int, arguments: str) -> None:
        # Authoritative parsed JSON replaces the streamed raw XML fragments.
        entry = self._tool_calls.setdefault(index, {"name": "", "args": []})
        entry["args"] = [arguments]

    def demote_tool_call(self, index: int) -> None:
        self._tool_calls.pop(index, None)

    def add_tool_result(self, index: int, name: str, result: str, is_error: bool = False) -> None:
        entry = self._tool_calls.get(index, {})
        arguments = "".join(entry.get("args", []))
        self.t.record("tool_call", index=index, name=name, arguments=arguments)
        self.t.record("tool_result", index=index, name=name, is_error=is_error, result=result)

    def add_warning(self, text: str) -> None:
        self.t.record("warning", text=text)

    def add_error(self, text: str) -> None:
        self.t.record("error", text=text)

    def add_system(self, text: str) -> None:
        self.t.record("system_note", text=text)


class LoggingUI:
    """Implements the top-level AgentUI contract (state + errors)."""

    def __init__(self, transcript: Transcript) -> None:
        self.t = transcript
        self.chat = LoggingChat(transcript)
        self._last_state: str | None = None

    def begin_generation(self) -> None:
        self.t.record("begin_generation")

    def set_state(self, state) -> None:
        name = getattr(state, "name", str(state))
        if name != self._last_state:
            self.t.record("state", state=name)
            self._last_state = name

    def set_prefill(
        self,
        percent: float,
        processed: int | None = None,
        total: int | None = None,
        session_ctx: int | None = None,
    ) -> None:
        self.t.record(
            "prefill", percent=percent, processed=processed, total=total, session_ctx=session_ctx
        )

    def add_reasoning_delta(self, text: str) -> None:
        # Reasoning text is already captured via chat.add_reasoning_chunk; no-op here.
        pass

    def set_usage(self, usage, timings) -> None:
        self.t.record("usage", usage=usage, timings=timings)

    def set_error(self, code: str, message: str) -> None:
        self.t.record("stream_error", code=code, message=message)

    def set_warning(self, message: str) -> None:
        self.t.record("warning", message=message)

    def set_info(self, message: str) -> None:
        self.t.record("info", message=message)

    def set_mode(self, mode) -> None:
        self.t.record("mode", mode=getattr(mode, "value", str(mode)))


def _make_approver(transcript: Transcript):
    """Auto-accept every flagged bash command (unattended debug run)."""

    async def _approve(command: str, warnings: list[str], allowlist_info: str) -> ApprovalDecision:
        transcript.record("bash_approval", command=command, warnings=warnings, decision="accept")
        return ApprovalDecision(kind="accept")

    return _approve


def _build_agent(client, cfg, ui, transcript: Transcript, args, artifacts_dir: Path):
    """The orchestrator with scripted (unattended) gates: plans auto-approve,
    planner questions answer themselves with the first option. Run artifacts
    land in the debug artifacts dir (never the user cache, and never the
    agent's workspace)."""
    from orchestrator import Orchestrator
    from sessions.store import SessionStore
    from tools.plan import PlanDecision

    async def auto_answer(questions: list[dict]) -> dict:
        answers = {}
        for question in questions:
            options = question.get("options") or []
            label = str(options[0].get("label", "")) if options else "proceed as you see fit"
            answers[str(question.get("question", ""))] = label
        transcript.record("auto_answer", answers=answers)
        return answers

    async def auto_approve(plan: str) -> PlanDecision:
        transcript.record("plan_approval", plan=plan, decision="approve")
        return PlanDecision(kind="approve")

    # Unattended operation is EXPLICIT here: real callbacks that auto-decide
    # and log it, bound at construction like every interactive callback.
    return Orchestrator(
        client,
        cfg,
        ui,
        approval=_make_approver(transcript),
        restricted_bash=args.restricted_bash,
        session_store=SessionStore(root=artifacts_dir / "sessions"),
        question_callback=auto_answer,
        plan_callback=auto_approve,
    )


def _resolve_workdir(workdir: str | None, task_label: str) -> Path:
    if workdir:
        return Path(workdir).absolute()
    base = Path.cwd() / "qw35-debug-runs"
    n = 1
    while True:
        cand = base / f"{task_label}-{n:03d}"
        if not cand.exists():
            return cand.absolute()
        n += 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="headless",
        description="Headless qwowl35 agent runner for debugging sessions.",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--task", help="path to a task prompt file (default: benchmark/cal_task.md)")
    src.add_argument("--prompt", help="literal task prompt text")
    p.add_argument("--base-url", help="qw35-server base URL (default http://127.0.0.1:8080)")
    p.add_argument("--think", choices=["auto", "on", "off"], help="thinking mode (default auto)")
    p.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        help="thinking budget when --think on",
    )
    p.add_argument("--workdir", help="scratch dir to run in (default qw35-debug-runs/<task>-NNN)")
    p.add_argument("--timeout", type=float, default=300.0, help="max seconds before giving up (default 300)")
    p.add_argument("--max-tokens", type=int, help="per-request completion token cap")
    p.add_argument(
        "--restricted-bash",
        action="store_true",
        help="run the agent's shell under 'bash -r' (restricted: no cd, no redirection, "
        "no /-qualified paths) for safer unattended runs",
    )
    p.add_argument("--no-raw", action="store_true", help="do not record raw SSE chunks")
    p.add_argument(
        "--no-compress",
        dest="compress",
        action="store_false",
        default=None,
        help="disable tool-output compression (full raw tool results)",
    )
    p.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        default=None,
        help="disable the query-aware semantic rerank of web results",
    )
    p.add_argument(
        "--rerank-scorer",
        choices=["cross-encoder", "bm25"],
        help="rerank scorer: cross-encoder (default; server /v1/rerank, "
        "degrades to bm25 without a server reranker), bm25 (lexical only)",
    )
    p.add_argument(
        "--no-lsp",
        dest="lsp",
        action="store_false",
        default=None,
        help="disable LSP semantic diagnostics on read/edit results "
        "(tree-sitter syntax checks only)",
    )
    p.add_argument(
        "--mode",
        choices=["normal", "plan", "web", "chat"],
        default="normal",
        help="debug-only: which TUI mode the turn runs under (default normal); "
        "in plan mode approvals are scripted to auto-approve and planner "
        "questions answer themselves with the first option",
    )
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    # Resolve the task text and a short label for the run dir.
    if args.prompt:
        task_text = args.prompt
        task_label = "prompt"
    else:
        task_path = Path(args.task) if args.task else DEFAULT_TASK
        if not task_path.exists():
            print(f"error: task file not found: {task_path}", file=sys.stderr)
            return 2
        task_text = task_path.read_text(encoding="utf-8")
        task_label = task_path.stem

    cfg = load_config(
        base_url=args.base_url,
        think=args.think,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        compress=args.compress,
        rerank=args.rerank,
        rerank_scorer=args.rerank_scorer,
        lsp=args.lsp,
    )
    try:  # LSP validation is optional; a broken install must not stop the run.
        from tools.lsp import configure as _configure_lsp

        _configure_lsp(cfg.lsp)
    except Exception:  # noqa: BLE001 - degrades to tree-sitter checks
        pass

    run_dir = _resolve_workdir(args.workdir, task_label)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Debug artifacts live OUTSIDE the agent's workspace: an in-workdir
    # transcript is workspace contamination — the explorer once found the
    # growing transcript.jsonl of its own session, inspected it, and blew the
    # server context (50K tokens in one result).
    artifacts_dir = run_dir.with_name(run_dir.name + "-artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(artifacts_dir / "transcript.jsonl")

    print(f"=== qwowl35 headless run ===")
    transcript.record(
        "meta",
        task=task_text,
        workdir=str(run_dir),
        base_url=cfg.base_url,
        think=cfg.think,
        reasoning_effort=cfg.reasoning_effort,
        max_tokens=cfg.max_tokens,
        restricted_bash=args.restricted_bash,
        timeout=args.timeout,
    )

    raw_sink = None if args.no_raw else (lambda data: transcript.record("raw_chunk", data=data))
    request_sink = lambda payload: transcript.record("request", **payload)  # noqa: E731
    client = Qw35Client(
        cfg.base_url,
        timeout=cfg.request_timeout,
        raw_sink=raw_sink,
        request_sink=request_sink,
    )
    registry = ToolRegistry(
        approval=_make_approver(transcript),
        restricted_bash=args.restricted_bash,
        compress=cfg.compress,
    )
    ui = LoggingUI(transcript)

    # Pre-flight: a clear message beats a wall of connection errors mid-stream.
    try:
        await client.health()
    except Exception as exc:  # noqa: BLE001
        print(f"error: server not reachable at {cfg.base_url}: {exc}", file=sys.stderr)
        transcript.record("error", text=f"server unreachable: {exc}")
        transcript.close()
        await client.aclose()
        return 2

    # chdir into the scratch dir BEFORE building the Agent: build_system_message()
    # reads os.getcwd() to ground the model, and the file/bash tools resolve paths
    # against the process cwd. Both must see the scratch dir, not the launch dir.
    prev_cwd = os.getcwd()
    os.chdir(run_dir)
    # Every run drives the orchestrator; --mode picks the dispatch (normal =
    # the direct freestyle executor, plan = planner pipeline, web, chat).
    agent = _build_agent(client, cfg, ui, transcript, args, artifacts_dir)
    transcript.record("cwd", path=str(run_dir))
    verdict = "finished"
    try:
        from modes import Mode

        ok = await asyncio.wait_for(
            agent.run_turn(task_text, Mode(args.mode)), timeout=args.timeout
        )
        verdict = "finished" if ok else "stream-error"
    except asyncio.TimeoutError:
        transcript.record("timeout", seconds=args.timeout)
        verdict = "timed-out"
    finally:
        os.chdir(prev_cwd)
        (artifacts_dir / "messages.json").write_text(
            json.dumps(agent.messages, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        try:  # stop any language servers started during the run.
            from tools.lsp import shutdown_all as _shutdown_lsp

            _shutdown_lsp()
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass
        await client.aclose()
        transcript.record("verdict", verdict=verdict)
        transcript.close()

    created = sorted(p.name for p in run_dir.iterdir())
    print(f"\nrun dir: {run_dir}")
    print(f"files the agent created: {created or '(none)'}")
    print(f"transcript: {artifacts_dir / 'transcript.jsonl'}")
    print(f"messages:   {artifacts_dir / 'messages.json'}")
    return 0 if verdict == "finished" else 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
