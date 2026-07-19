"""Read side of the session cache: list past sessions and load one for
restore.

``list_session_summaries`` feeds the /sessions selector rows;
``load_session`` walks a session's ``turns/`` in order and yields what the
app needs to rehydrate the conversation — the (goal, outcome) pairs for the
orchestrator's turn log, the verbatim CHAT message deltas, and the display
records the chat view replays. Everything is best-effort tolerant: corrupt
metadata falls back to filesystem facts, malformed transcript lines are
skipped, incomplete turn dirs (a crash mid-turn) are ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sessions.store import _SESSION_DIR_PATTERN, _TURN_DIR_PATTERN

_DISPLAY_RECORD_KINDS = frozenset(
    {"user", "assistant", "tool_result", "system_note"}
)


@dataclass
class SessionSummary:
    session_hash: str
    created: str
    last_active: str
    last_active_ts: float
    turn_count: int
    first_goal: str
    last_mode: str


@dataclass
class RestoredTurn:
    goal: str
    mode: str
    outcome: str
    ok: bool
    turn_dir: str = ""
    chat_messages: list = field(default_factory=list)
    display_records: list = field(default_factory=list)


def list_session_summaries(
    root: Path, exclude: str | None = None
) -> list[SessionSummary]:
    """Summaries of restorable sessions under ``root``, newest first.

    Sessions with no recorded turn are skipped; the ``exclude`` hash (the
    app's own live session) never appears.
    """
    summaries: list[SessionSummary] = []
    try:
        entries = [
            entry
            for entry in root.iterdir()
            if entry.is_dir()
            and _SESSION_DIR_PATTERN.fullmatch(entry.name)
            and entry.name != exclude
        ]
    except OSError:
        return []
    for entry in entries:
        summary = _summarize(entry)
        if summary is not None and summary.turn_count > 0:
            summaries.append(summary)
    summaries.sort(key=lambda item: item.last_active_ts, reverse=True)
    return summaries


def load_session(root: Path, session_hash: str) -> list[RestoredTurn]:
    """The session's completed turns in conversation order."""
    turns: list[RestoredTurn] = []
    turns_dir = root / session_hash / "turns"
    try:
        entries = sorted(
            entry
            for entry in turns_dir.iterdir()
            if entry.is_dir() and _TURN_DIR_PATTERN.fullmatch(entry.name)
        )
    except OSError:
        return []
    for entry in entries:
        turn = _load_turn(entry)
        if turn is not None:
            turns.append(turn)
    return turns


def _summarize(session_dir: Path) -> SessionSummary | None:
    meta = _read_json(session_dir / "session.json")
    if meta is None:
        meta = _fallback_session_meta(session_dir)
        if meta is None:
            return None
    try:
        mtime = session_dir.stat().st_mtime
    except OSError:
        mtime = 0.0
    last_active_ts = meta.get("last_active_ts")
    if not isinstance(last_active_ts, (int, float)):
        last_active_ts = mtime
    return SessionSummary(
        session_hash=session_dir.name,
        created=str(meta.get("created", "")),
        last_active=str(meta.get("last_active", "")),
        last_active_ts=float(last_active_ts),
        turn_count=_coerce_turn_count(meta, session_dir),
        first_goal=str(meta.get("first_goal", "")),
        last_mode=str(meta.get("last_mode", "")),
    )


def _fallback_session_meta(session_dir: Path) -> dict | None:
    """Reconstruct enough of session.json from the first turn's meta."""
    first_turn = _read_json(session_dir / "turns" / "0001" / "meta.json")
    if first_turn is None:
        return None
    return {
        "created": first_turn.get("started", ""),
        "last_active": first_turn.get("started", ""),
        "first_goal": first_turn.get("goal", ""),
        "last_mode": first_turn.get("mode", ""),
    }


def _coerce_turn_count(meta: dict, session_dir: Path) -> int:
    count = meta.get("turn_count")
    if isinstance(count, int) and count >= 0:
        return count
    try:
        return sum(
            1 for _ in (session_dir / "turns").glob("*/meta.json")
        )
    except OSError:
        return 0


def _load_turn(turn_dir: Path) -> RestoredTurn | None:
    meta = _read_json(turn_dir / "meta.json")
    if meta is None:
        return None
    goal = meta.get("goal")
    if not isinstance(goal, str) or not goal:
        return None
    chat_messages = meta.get("chat_messages")
    if not isinstance(chat_messages, list):
        chat_messages = []
    return RestoredTurn(
        goal=goal,
        mode=str(meta.get("mode", "")),
        outcome=str(meta.get("outcome", "") or ""),
        ok=bool(meta.get("ok", False)),
        turn_dir=turn_dir.name,
        chat_messages=chat_messages,
        display_records=_load_display_records(turn_dir / "transcript.jsonl"),
    )


def _load_display_records(transcript_path: Path) -> list[dict]:
    records: list[dict] = []
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("kind") in _DISPLAY_RECORD_KINDS:
            records.append(record)
    return records


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None
