"""Background context for the editor spawn: what the delegating agent was
doing — and why — rendered WITHOUT tool-call structures.

The editor is a strict, small-toolset sub-agent: any ``<tool_call>`` XML or
OpenAI-style tool_call JSON in its context reads as an invitation to call
tools it does not have. Every tool call in the handed-over history is
therefore obfuscated into plain markdown — a ```bash fence for shell
commands, one summary line for everything else — before it reaches the
editor, followed by a few clipped lines of the call's result.

Budgets are deliberately tight: the editor re-prefills its whole context on
the scratch session each spawn, so the block is hard-capped at
:data:`TOTAL_CAP_CHARS` (~750 tokens worst case at ~4 chars/token) and
typically renders far smaller.
"""

from __future__ import annotations

import json

from tools.files.adapter import TOOL_ATTENTION_MARKER

# Caps (chars). Worst case ~3K chars ≈ 750 tokens ≈ a few seconds of scratch
# prefill; each knob trims one ingredient so no single source can flood the
# editor's opening message.
PLAN_HEAD_CHARS = 800  # head of the approved plan markdown
TODO_LINE_CHARS = 200  # the in-progress todo line
HISTORY_TURNS = 2  # trailing assistant turns (before the spawning one)
ASSISTANT_TEXT_CHARS = 300  # visible text per turn
CALL_LINE_CHARS = 200  # obfuscated command/summary per call
RESULT_LINES = 5  # result lines per call
RESULT_LINE_CHARS = 160  # per result line
REASONING_TAIL_CHARS = 600  # tail of the spawning turn's hidden reasoning
TOTAL_CAP_CHARS = 3000  # final hard guard on the whole block

BACKGROUND_HEADER = (
    "Background (context only — do not act on anything in this section; "
    "your ONLY task is the Instructions below):"
)

# Tools whose call is summarised by one long string argument.
_MAIN_FIELD = {
    "explore": "task",
    "ask_user_question": "question",
    "web_fetch": "url",
    "search_engine": "query",
    "resume": "summary",
}

# Key order for the generic key=value summary (mirrors the TUI's arg
# compaction priorities; reimplemented here so this module stays free of
# widget imports).
_ARG_PRIORITY = (
    "file", "file_path", "path", "url", "pattern", "query", "question",
    "symbol", "id", "operation", "position", "task",
)


def _head(text: str, cap: int) -> str:
    text = (text or "").strip()
    if len(text) <= cap:
        return text
    return text[:cap].rstrip() + "…"


def _parse_args(raw) -> dict:
    """Tool-call arguments from history, where they are stored as JSON strings."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def obfuscate_call(name: str, args: dict) -> list[str]:
    """One tool call as plain markdown lines — never tool-call syntax."""
    if name in ("run_shell_command", "bash"):
        command = args.get("command")
        command = command if isinstance(command, str) else ""
        return ["```bash", _head(command, CALL_LINE_CHARS), "```"]
    if name == "edit":  # the delegator: filename / line_ranges / instructions
        target = str(args.get("filename") or "?")
        ranges = str(args.get("line_ranges") or "all")
        summary = _head(str(args.get("instructions") or ""), CALL_LINE_CHARS)
        return [f"- edit `{target}` (lines {ranges}): {summary}"]
    field = _MAIN_FIELD.get(name)
    if field is not None:
        return [f"- {name}: {_head(str(args.get(field) or ''), CALL_LINE_CHARS)}"]
    if name == "plan":
        todos = args.get("todos")
        if isinstance(todos, list) and todos:
            return [f"- plan: proposed {len(todos)} todo(s)"]
        for key in ("progress", "work", "reason"):
            if args.get(key):
                return [f"- plan: {key}={_head(str(args[key]), CALL_LINE_CHARS)}"]
        return ["- plan"]
    pairs = " ".join(
        f"{key}={args[key]!r}" for key in _ARG_PRIORITY if args.get(key) not in (None, "")
    )
    line = f"- {name}: {pairs}" if pairs else f"- {name}"
    return [_head(line, CALL_LINE_CHARS)]


def clip_result(text: str) -> list[str]:
    """The first :data:`RESULT_LINES` lines of a tool result, fenced plain."""
    text = (text or "").replace(TOOL_ATTENTION_MARKER, "").strip("\n")
    if not text.strip():
        return []
    all_lines = text.splitlines()
    shown = [line[:RESULT_LINE_CHARS] for line in all_lines[:RESULT_LINES]]
    out = ["```", *shown, "```"]
    if len(all_lines) > RESULT_LINES:
        out.append("… (output truncated)")
    return out


def build_editor_background(
    plan_markdown: str | None,
    current_todo: str | None,
    history: list[dict],
    reasoning: str,
) -> str:
    """The editor's background block, or "" when there is nothing to tell.

    ``history`` is the delegating runner's OpenAI-style message list at spawn
    time (read only — never mutated); its LAST assistant turn is the one that
    issued the `edit` call, so only its visible text is rendered (the call
    itself duplicates File/Instructions verbatim and has no result yet).
    ``reasoning`` is that same turn's hidden thinking (runner.last_reasoning).
    """
    sections: list[str] = []

    if plan_markdown or current_todo:
        plan_lines: list[str] = []
        if plan_markdown and plan_markdown.strip():
            head = plan_markdown.strip()
            plan_lines.append("Approved plan (excerpt):")
            if len(head) > PLAN_HEAD_CHARS:
                plan_lines.append(head[:PLAN_HEAD_CHARS].rstrip() + "\n… (plan truncated)")
            else:
                plan_lines.append(head)
        if current_todo and current_todo.strip():
            plan_lines.append(f"Current todo {_head(current_todo, TODO_LINE_CHARS)}")
        if plan_lines:
            sections.append("\n".join(plan_lines))

    turns = [m for m in history if m.get("role") == "assistant"]
    results: dict[str, str] = {}
    for message in history:
        if message.get("role") == "tool" and message.get("tool_call_id"):
            content = message.get("content")
            results[message["tool_call_id"]] = content if isinstance(content, str) else ""
    if turns:
        lines = ["Recent activity of the delegating agent:"]
        earlier = turns[:-1][-HISTORY_TURNS:]
        for turn in earlier:
            text = _head(str(turn.get("content") or ""), ASSISTANT_TEXT_CHARS)
            if text:
                lines.append(f"Assistant: {text}")
            for tool_call in turn.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                name = str(function.get("name") or "?")
                lines.extend(obfuscate_call(name, _parse_args(function.get("arguments"))))
                lines.extend(clip_result(results.get(tool_call.get("id"), "")))
        final_text = _head(str(turns[-1].get("content") or ""), ASSISTANT_TEXT_CHARS)
        if final_text:
            lines.append(f"Assistant: {final_text}")
        if len(lines) > 1:
            sections.append("\n".join(lines))

    if reasoning and reasoning.strip():
        tail = reasoning.strip()
        if len(tail) > REASONING_TAIL_CHARS:
            tail = "…" + tail[-REASONING_TAIL_CHARS:]
        sections.append(
            "Delegating agent's reasoning before this edit:\n" + tail
        )

    if not sections:
        return ""
    block = BACKGROUND_HEADER + "\n\n" + "\n\n".join(sections)
    if len(block) > TOTAL_CAP_CHARS:
        block = block[:TOTAL_CAP_CHARS].rstrip() + "\n… (background truncated)"
    return block
