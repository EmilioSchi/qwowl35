"""Display-side replay of a restored session into the chat view.

Deliberately partial fidelity: restored turns render as static messages
through the chat view's existing append API — user text, assistant prose,
and one plain system line per tool result. No fabricated live tool blocks
(their reveal timers and per-turn indices exist for streaming, not history)
and no reasoning (it was never part of the model-visible conversation).
The model-visible state is restored exactly elsewhere (Orchestrator.restore).
"""

from __future__ import annotations

TOOL_RESULT_PREVIEW_CHARS = 100


def replay_into_chat(chat, turns) -> None:
    chat.add_system(f"restored session · {len(turns)} turn(s)")
    for turn in turns:
        chat.add_user(turn.goal)
        for record in turn.display_records:
            _replay_record(chat, record)


def _replay_record(chat, record: dict) -> None:
    kind = record.get("kind")
    if kind == "user":
        text = str(record.get("text", ""))
        if text:
            chat.add_user(text)
    elif kind == "assistant":
        content = str(record.get("content", ""))
        if content.strip():
            chat.add_assistant_chunk(content)
            chat.flush_assistant()
    elif kind == "tool_result":
        chat.add_system(_tool_result_line(record))
    elif kind == "system_note":
        text = str(record.get("text", ""))
        if text:
            chat.add_system(text)


def _tool_result_line(record: dict) -> str:
    name = str(record.get("name", "tool"))
    tag = "error" if record.get("is_error") else "ok"
    preview = str(record.get("result", "")).strip().splitlines()
    first_line = preview[0] if preview else ""
    if len(first_line) > TOOL_RESULT_PREVIEW_CHARS:
        first_line = first_line[:TOOL_RESULT_PREVIEW_CHARS] + "…"
    if first_line:
        return f"[{name}] {tag} · {first_line}"
    return f"[{name}] {tag}"
