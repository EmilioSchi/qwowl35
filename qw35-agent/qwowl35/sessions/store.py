"""Session cache: every conversation persisted under the shared qwowl35
cache dir, one directory per session.

Layout::

    {user_cache_dir}/sessions/{session_hash}/
        session.json            session-level metadata (upserted per turn)
        turns/0001/             one directory per turn, in conversation order
            meta.json           goal, mode, outcome, session-path tallies
            transcript.jsonl    raw + parsed model I/O (see transcript.py)
            plan.md, todos.json, explore-N.md, ...   stage artifacts

The zero-padded turn index makes lexicographic order the replay order, which
is what session restore walks. This module absorbs the former ``runs.py``
(per-turn artifacts now live inside their session) and keeps its discipline:
all writes are best-effort and must never break the agent. The GPU KV/SSM
checkpoints stay in server memory — restoring a session re-primes them via a
normal full prefill on the first request.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import platformdirs

from sessions.transcript import TranscriptWriter

MAX_AGE_DAYS = 30
MAX_SESSIONS = 50
ORPHAN_GRACE_SECONDS = 3600

_SESSION_DIR_PATTERN = re.compile(r"[0-9a-f]{64}")
_TURN_DIR_PATTERN = re.compile(r"\d{4}")


def generate_session_hash() -> str:
    current_timestamp_ns = str(time.time_ns())
    random_entropy = uuid.uuid4().hex
    source_string = f"{current_timestamp_ns}-{random_entropy}"
    return hashlib.sha256(source_string.encode()).hexdigest()


def default_sessions_root() -> Path:
    return Path(platformdirs.user_cache_dir("qwowl35")) / "sessions"


def _read_turn_meta(turn_dir: Path) -> dict:
    try:
        payload = json.loads((turn_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


@dataclass
class TurnDir:
    """One turn's artifact directory. All writes are best-effort: artifact
    persistence must never break the agent."""

    path: Path
    meta: dict = field(default_factory=dict)
    transcript: TranscriptWriter | None = None

    def save(self, name: str, text: str) -> None:
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            (self.path / name).write_text(text, encoding="utf-8")
        except OSError:
            pass

    def save_json(self, name: str, payload: dict) -> None:
        try:
            self.save(name, json.dumps(payload, indent=2, ensure_ascii=False))
        except (TypeError, ValueError):
            pass

    def record(self, kind: str, **fields) -> None:
        if self.transcript is not None:
            self.transcript.record(kind, **fields)

    def record_timings(self, timings: dict | None) -> None:
        """Tally a request's session path (reset/extend/checkpoint) into meta."""
        if not isinstance(timings, dict):
            return
        path = timings.get("session_path")
        if isinstance(path, str) and path:
            counts = self.meta.setdefault("session_paths", {})
            counts[path] = counts.get(path, 0) + 1
        for key in ("prompt_eval_count", "cached_prompt_tokens"):
            value = timings.get(key)
            if isinstance(value, (int, float)):
                self.meta[f"total_{key}"] = self.meta.get(f"total_{key}", 0) + int(value)

    def flush_meta(self) -> None:
        self.save_json("meta.json", self.meta)

    def close(self) -> None:
        if self.transcript is not None:
            self.transcript.close()


class SessionStore:
    """One store per app process, identified by a fresh session hash.

    ``attach`` re-points the store at a restored session so new turns append
    to it, continuing the turn numbering.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root if root is not None else default_sessions_root()
        self.session_hash = generate_session_hash()
        self._active_turn: TurnDir | None = None
        self._next_turn_index: int | None = None

    @property
    def session_dir(self) -> Path:
        return self.root / self.session_hash

    def attach(self, session_hash: str) -> None:
        self.session_hash = session_hash
        self._active_turn = None
        self._next_turn_index = None

    def rotate(self) -> str:
        """Start a fresh session (a cleared conversation): new hash, turn
        numbering restarts. The previous session dir is left behind as a
        restorable past session."""
        self.attach(generate_session_hash())
        return self.session_hash

    def fork(self, source_hash: str, turn_names: list[str]) -> str:
        """Branch a new session from a prefix of an existing one (a partial
        restore): the named turn dirs are COPIED so the fork is fully
        self-contained and restorable on its own, provenance is stamped in
        its session.json, and the store attaches to it. The source session
        is never modified — copy failures leave gaps, never damage."""
        fork_hash = generate_session_hash()
        source_turns = self.root / source_hash / "turns"
        fork_turns = self.root / fork_hash / "turns"
        copied: list[str] = []
        for name in turn_names:
            if not _TURN_DIR_PATTERN.fullmatch(name):
                continue
            source_dir = source_turns / name
            try:
                if source_dir.is_dir():
                    shutil.copytree(source_dir, fork_turns / name)
                    copied.append(name)
            except OSError:
                continue
        self.attach(fork_hash)
        self._upsert_session_meta(
            first_goal=self._turn_goal(fork_turns, copied[0]) if copied else None,
            last_mode=self._turn_mode(fork_turns, copied[-1]) if copied else None,
            turn_count=len(copied),
            extra={
                "restored_from": {"session": source_hash, "turns": len(copied)}
            },
        )
        return fork_hash

    @staticmethod
    def _turn_goal(turns_dir: Path, name: str) -> str | None:
        meta = _read_turn_meta(turns_dir / name)
        goal = meta.get("goal")
        return goal if isinstance(goal, str) and goal else None

    @staticmethod
    def _turn_mode(turns_dir: Path, name: str) -> str | None:
        meta = _read_turn_meta(turns_dir / name)
        mode = meta.get("mode")
        return mode if isinstance(mode, str) and mode else None

    def begin_turn(self, goal: str) -> TurnDir:
        index = self._allocate_turn_index()
        turn = TurnDir(path=self.session_dir / "turns" / f"{index:04d}")
        turn.meta["goal"] = goal
        turn.meta["started"] = time.strftime("%Y%m%d-%H%M%S")
        try:
            turn.path.mkdir(parents=True, exist_ok=True)
            turn.transcript = TranscriptWriter(turn.path / "transcript.jsonl")
        except OSError:
            turn.transcript = None
        # Flushed immediately so a turn aborted mid-stream (quit, crash,
        # cancelled worker) still counts as a recorded turn: the goal is
        # known, the outcome stays empty, and restore tolerates both.
        turn.flush_meta()
        self._active_turn = turn
        self._upsert_session_meta(first_goal=goal)
        return turn

    def end_turn(self, turn: TurnDir, ok: bool) -> None:
        turn.close()
        try:
            turn_count = int(turn.path.name)
        except ValueError:
            turn_count = None
        self._upsert_session_meta(
            last_mode=turn.meta.get("mode"), turn_count=turn_count
        )
        if self._active_turn is turn:
            self._active_turn = None

    def cleanup(
        self, max_age_days: int = MAX_AGE_DAYS, max_sessions: int = MAX_SESSIONS
    ) -> int:
        """Delete stale session dirs: past the age cutoff, beyond the count
        cap, or orphaned (crashed before the first turn completed).

        Called at app startup and exit. Best-effort: unrecognized names are
        left alone, the attached session is always spared, errors are
        swallowed. Also sweeps the retired legacy ``runs/`` tree next to the
        sessions root. Returns how many entries were removed.
        """
        removed = self._sweep_legacy_runs()
        try:
            candidates = [
                entry
                for entry in self.root.iterdir()
                if entry.is_dir()
                and _SESSION_DIR_PATTERN.fullmatch(entry.name)
                and entry.name != self.session_hash
            ]
        except OSError:
            return removed
        now = time.time()
        cutoff = now - max_age_days * 86400
        survivors: list[tuple[float, Path]] = []
        for entry in candidates:
            last_active = self._last_active_ts(entry, now)
            if self._is_orphan(entry):
                if now - last_active > ORPHAN_GRACE_SECONDS:
                    removed += self._remove(entry)
                continue
            if last_active < cutoff:
                removed += self._remove(entry)
            else:
                survivors.append((last_active, entry))
        survivors.sort(key=lambda item: item[0], reverse=True)
        for _, entry in survivors[max_sessions:]:
            removed += self._remove(entry)
        return removed

    def _allocate_turn_index(self) -> int:
        if self._next_turn_index is None:
            highest = 0
            try:
                for entry in (self.session_dir / "turns").iterdir():
                    if entry.is_dir() and _TURN_DIR_PATTERN.fullmatch(entry.name):
                        highest = max(highest, int(entry.name))
            except OSError:
                pass
            self._next_turn_index = highest + 1
        index = self._next_turn_index
        self._next_turn_index += 1
        return index

    def _upsert_session_meta(
        self,
        first_goal: str | None = None,
        last_mode: str | None = None,
        turn_count: int | None = None,
        extra: dict | None = None,
    ) -> None:
        meta = self._load_session_meta()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        meta.setdefault("hash", self.session_hash)
        meta.setdefault("created", stamp)
        if first_goal is not None:
            meta.setdefault("first_goal", first_goal)
        if isinstance(last_mode, str) and last_mode:
            meta["last_mode"] = last_mode
        if turn_count is not None:
            meta["turn_count"] = turn_count
        if extra:
            meta.update(extra)
        meta["last_active"] = stamp
        meta["last_active_ts"] = time.time()
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            (self.session_dir / "session.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass

    def _load_session_meta(self) -> dict:
        try:
            payload = json.loads(
                (self.session_dir / "session.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _sweep_legacy_runs(self) -> int:
        legacy = self.root.parent / "runs"
        if not legacy.is_dir():
            return 0
        return self._remove(legacy)

    @staticmethod
    def _last_active_ts(session_dir: Path, now: float) -> float:
        try:
            payload = json.loads(
                (session_dir / "session.json").read_text(encoding="utf-8")
            )
            value = payload.get("last_active_ts")
            if isinstance(value, (int, float)):
                return float(value)
        except (OSError, ValueError, AttributeError):
            pass
        try:
            return session_dir.stat().st_mtime
        except OSError:
            return now

    @staticmethod
    def _is_orphan(session_dir: Path) -> bool:
        if (session_dir / "session.json").exists():
            return False
        try:
            return not any((session_dir / "turns").glob("*/meta.json"))
        except OSError:
            return False

    @staticmethod
    def _remove(path: Path) -> int:
        try:
            shutil.rmtree(path)
            return 1
        except OSError:
            return 0
