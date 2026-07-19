"""Conversation cache: persisted sessions, raw-I/O transcripts, restore.

Storage lives under ``platformdirs.user_cache_dir("qwowl35")/sessions/`` —
see ``sessions.store`` for the on-disk layout, ``sessions.transcript`` for
the JSONL record schema, and ``sessions.restore``/``sessions.replay`` for
the read side used by the ``/sessions`` command.
"""

from sessions.replay import replay_into_chat
from sessions.restore import (
    RestoredTurn,
    SessionSummary,
    list_session_summaries,
    load_session,
)
from sessions.store import (
    MAX_AGE_DAYS,
    MAX_SESSIONS,
    SessionStore,
    TurnDir,
    default_sessions_root,
    generate_session_hash,
)
from sessions.transcript import TranscriptWriter

__all__ = [
    "MAX_AGE_DAYS",
    "MAX_SESSIONS",
    "RestoredTurn",
    "SessionStore",
    "SessionSummary",
    "TranscriptWriter",
    "TurnDir",
    "default_sessions_root",
    "generate_session_hash",
    "list_session_summaries",
    "load_session",
    "replay_into_chat",
]
