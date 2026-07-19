"""Render a saved session's transcript as a human-readable Markdown log.

The on-disk ``transcript.jsonl`` holds only the display-restore essentials
(``user`` / ``assistant`` / ``tool_result`` records — see
``sessions.transcript``). ``sessions.restore.load_session`` already surfaces
exactly those as ``RestoredTurn.display_records`` (plus each turn's goal, mode
and outcome), so this module is a pure renderer over that structure — no new
reader is needed. Tool calls are paired to their results by call ``id`` so each
invocation reads as one block.

Runnable as a script::

    python -m sessions.export                 # most recent session -> stdout
    python -m sessions.export <hash> -o log.md
    python -m sessions.export --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Flat sys.path quirk (see tests/conftest.py): make the package dir importable
# when this file is run directly as ``python qwowl35/sessions/export.py``. A
# no-op when imported normally.
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from sessions.restore import (  # noqa: E402
    RestoredTurn,
    list_session_summaries,
    load_session,
)
from sessions.store import default_sessions_root  # noqa: E402

ARGS_MAX_CHARS = 400
RESULT_MAX_CHARS = 2000


def render_session_markdown(turns: list[RestoredTurn], *, title: str = "") -> str:
    """A whole session rendered as Markdown, one section per turn."""
    lines: list[str] = [f"# {title or 'qwowl35 session'}", "", f"_{len(turns)} turn(s)_", ""]
    for index, turn in enumerate(turns, start=1):
        lines.extend(_render_turn(turn, index))
    return "\n".join(lines).rstrip() + "\n"


def _render_turn(turn: RestoredTurn, index: int) -> list[str]:
    lines: list[str] = [f"## Turn {index} · {turn.mode or 'normal'}", ""]
    if turn.goal:
        lines += ["**User**", "", *_quote(turn.goal), ""]

    # Pair tool results to the assistant tool_call that issued them by id, so
    # each call renders together with its result; leftover results (no matching
    # call) fall through as standalone blocks.
    results = {
        rec.get("id"): rec
        for rec in turn.display_records
        if rec.get("kind") == "tool_result" and rec.get("id")
    }
    consumed: set = set()

    for rec in turn.display_records:
        kind = rec.get("kind")
        if kind == "user":
            text = str(rec.get("text", "")).strip()
            if text:
                lines += ["**User**", "", *_quote(text), ""]
        elif kind == "assistant":
            content = str(rec.get("content", "")).strip()
            if content:
                lines += ["**Assistant**", "", content, ""]
            for call in rec.get("tool_calls") or []:
                lines += _render_call(call, results, consumed)
        elif kind == "tool_result":
            if rec.get("id") not in consumed:
                lines += _render_result(rec)
        elif kind == "system_note":
            text = str(rec.get("text", "")).strip()
            if text:
                lines += [f"> _{text}_", ""]

    outcome = (turn.outcome or "").strip()
    if outcome:
        lines += [f"**Outcome:** {outcome}", ""]
    lines += ["---", ""]
    return lines


def _render_call(call: dict, results: dict, consumed: set) -> list[str]:
    name = str(call.get("name", "tool"))
    call_id = call.get("id")
    result = results.get(call_id)
    status = "ok"
    if result is not None:
        status = "error" if result.get("is_error") else "ok"
        consumed.add(call_id)
    lines = [f"**⚙ `{name}`** → {status}", ""]
    args = _compact_args(call.get("arguments"))
    if args:
        lines += _code_block(args, lang="json") + [""]
    if result is not None:
        lines += _code_block(_ellipsize(str(result.get("result", "")), RESULT_MAX_CHARS)) + [""]
    return lines


def _render_result(rec: dict) -> list[str]:
    name = str(rec.get("name", "tool"))
    status = "error" if rec.get("is_error") else "ok"
    body = _ellipsize(str(rec.get("result", "")), RESULT_MAX_CHARS)
    return [f"**⚙ `{name}`** → {status}", "", *_code_block(body), ""]


def _compact_args(arguments) -> str:
    if not arguments:
        return ""
    try:
        text = json.dumps(arguments, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(arguments)
    return _ellipsize(text, ARGS_MAX_CHARS)


def _ellipsize(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"… [+{len(text) - limit} chars]"


def _code_block(text: str, *, lang: str = "") -> list[str]:
    fence = "```"
    while fence in text:  # never let content close the fence early
        fence += "`"
    return [f"{fence}{lang}", text, fence]


def _quote(text: str) -> list[str]:
    lines = [("> " + line if line else ">") for line in text.splitlines()]
    return lines or ["> "]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sessions.export",
        description="Render a saved qwowl35 session as a Markdown log.",
    )
    parser.add_argument("session", nargs="?", help="session hash (default: most recent)")
    parser.add_argument("-l", "--list", action="store_true", help="list saved sessions and exit")
    parser.add_argument("-o", "--output", help="write Markdown here (default: stdout)")
    parser.add_argument("--root", help="sessions cache root (default: platform cache dir)")
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else default_sessions_root()
    summaries = list_session_summaries(root)

    if args.list:
        if not summaries:
            print(f"no sessions under {root}", file=sys.stderr)
            return 1
        for summary in summaries:
            print(
                f"{summary.session_hash}  {summary.last_active or '?'}  "
                f"{summary.turn_count} turn(s)  {summary.first_goal}"
            )
        return 0

    session_hash = args.session
    if session_hash is None:
        if not summaries:
            print(f"no sessions under {root}", file=sys.stderr)
            return 1
        session_hash = summaries[0].session_hash

    turns = load_session(root, session_hash)
    if not turns:
        print(f"no restorable turns for session {session_hash!r} under {root}", file=sys.stderr)
        return 1

    markdown = render_session_markdown(turns, title=f"qwowl35 session · {session_hash}")
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
