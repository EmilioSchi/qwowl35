"""Append-only JSONL transcript of a turn's conversation.

Each record is one line: ``{"t": <seconds since the writer opened>, "kind":
..., **fields}``. The transcript holds only the display-restore essentials —
``user``, ``assistant`` (content + tool_calls) and ``tool_result`` records —
which is what the /sessions replay renders and what the Markdown export
(``sessions/export.py``) reads. It is deliberately not a full-fidelity trace:
model-visible resume state lives in ``meta.json`` (``chat_messages``), and the
exact outgoing payloads / raw pre-parse SSE are captured only by the headless
debug harness.

All disk I/O is best-effort: a writer that fails to open or write disables
itself silently. Transcript persistence must never break the agent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class TranscriptWriter:
    def __init__(self, path: Path) -> None:
        self._t0 = time.monotonic()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")
        except OSError:
            self._fh = None

    def record(self, kind: str, **fields) -> None:
        if self._fh is None:
            return
        rec = {"t": round(time.monotonic() - self._t0, 3), "kind": kind, **fields}
        try:
            self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        except (OSError, TypeError, ValueError):
            self.close()

    def close(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.close()
        except OSError:
            pass
        self._fh = None
