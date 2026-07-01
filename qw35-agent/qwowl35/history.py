"""Persistent prompt-message history, in a self-contained container.

This is the message-history store the ``PromptInput`` widget recalls with Up/Down.
It used to live as loose constants in ``config.py`` plus module functions and
navigation fields baked into the widget; it's gathered here into two cohesive
containers (inspired by Go readline's ``opHistory`` struct):

- ``HistoryConfig`` — the *where* and *how much*: file path, cap, on/off switch.
- ``MessageHistory`` — the entries plus the Up/Down navigation cursor, with a
  concurrency-safe ``append`` that survives several qwowl35 processes writing at
  once.

Storage is JSON-lines (one JSON string per physical line) under the OS cache dir
(``platformdirs.user_cache_dir("qwowl35")`` — e.g. ``~/Library/Caches/qwowl35`` on
macOS, ``~/.cache/qwowl35`` on Linux), so multiline submissions round-trip.

Persistence is best-effort: every disk touch is wrapped so history never crashes
the app. Concurrent writers are serialized with a ``fcntl.flock`` advisory lock on
a sibling ``history.lock`` file, and each write re-reads the file to *merge* other
instances' entries before replacing it atomically (``os.replace``) — so two
processes writing at once don't clobber each other.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import platformdirs

# Advisory locking is POSIX-only. On Windows there's no fcntl; we degrade to a
# lock-free best-effort write (still atomic via os.replace, just no cross-process
# mutual exclusion) rather than failing to import.
try:
    import fcntl
except ImportError:  # pragma: no cover - platform-dependent
    fcntl = None


@dataclass(frozen=True)
class HistoryConfig:
    """Where history lives and how much of it we keep."""

    file: Path
    max_entries: int = 100
    # When false, history stays in memory and never touches disk (useful for
    # tests and for opting a session out of persistence).
    enabled: bool = True

    @classmethod
    def default(cls) -> "HistoryConfig":
        base = Path(platformdirs.user_cache_dir("qwowl35"))
        return cls(file=base / "history")

    @property
    def lock_file(self) -> Path:
        return self.file.with_name(self.file.name + ".lock")

    def _cap(self, entries: list[str]) -> list[str]:
        """Trim to the last ``max_entries`` (``max_entries <= 0`` keeps all)."""
        if self.max_entries <= 0:
            return entries
        return entries[-self.max_entries:]


class MessageHistory:
    """The submitted-message list plus its Up/Down navigation cursor.

    Construction is read-only: it loads whatever is on disk (nothing if the file
    is absent) but never creates the directory or file — that happens lazily on
    the first ``append``, matching the old ``save_history`` behavior so merely
    building the widget has no filesystem side effects.
    """

    def __init__(self, config: HistoryConfig | None = None) -> None:
        self._cfg = config or HistoryConfig.default()
        self._entries: list[str] = self._load()
        # Navigation state: None means "live draft" (not currently browsing);
        # otherwise an index into ``self._entries``.
        self._hidx: int | None = None
        self._draft: str = ""

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> list[str]:
        """Read history (JSON-lines; skip blank/malformed lines, tail the cap)."""
        try:
            raw = self._cfg.file.read_text(encoding="utf-8")
        except (OSError, FileNotFoundError):
            return []
        out: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return self._cfg._cap(out)

    def append(self, text: str) -> None:
        """Record a submitted message: dedup, then persist under a file lock.

        Skips empty submissions and consecutive duplicates (``a, b, a`` is kept as
        ``a, b, a`` — only a repeat of the *current last* entry is dropped). Always
        resets navigation, so the next Up starts from the newest entry.
        """
        text = text.rstrip("\n")
        self.reset_navigation()
        if not text:
            return

        if not self._cfg.enabled:
            if self._entries and self._entries[-1] == text:
                return
            self._entries.append(text)
            self._entries = self._cfg._cap(self._entries)
            return

        # Persistence is best-effort: never let a disk error break the app.
        try:
            self._cfg.file.parent.mkdir(parents=True, exist_ok=True)
            with _file_lock(self._cfg.lock_file):
                # Re-read under the lock so concurrent instances' entries are
                # merged in rather than clobbered by our write.
                merged = self._load()
                if merged and merged[-1] == text:
                    self._entries = merged
                    return
                merged.append(text)
                merged = self._cfg._cap(merged)
                _write_atomic(self._cfg.file, merged)
                self._entries = merged
        except OSError:
            # Fall back to an in-memory update so recall still works this session.
            if not (self._entries and self._entries[-1] == text):
                self._entries.append(text)
                self._entries = self._cfg._cap(self._entries)

    # ------------------------------------------------------------------ #
    # Navigation (moved out of the widget)
    # ------------------------------------------------------------------ #
    def prev(self, current_text: str) -> str | None:
        """Step to the previous (older) entry; return it, or None to do nothing.

        On the first step it captures ``current_text`` as the draft to restore
        later. Stepping past the oldest entry is idempotent (stays on it).
        """
        if not self._entries:
            return None
        if self._hidx is None:
            self._draft = current_text
            self._hidx = len(self._entries) - 1
        elif self._hidx > 0:
            self._hidx -= 1
        return self._entries[self._hidx]

    def next(self) -> str | None:
        """Step to the next (newer) entry; return it, or None to do nothing.

        Returns None when not currently browsing. Stepping past the newest entry
        restores the saved draft (a ``str``, possibly ``""`` — distinct from None,
        so an empty draft is still loaded).
        """
        if self._hidx is None:
            return None
        if self._hidx < len(self._entries) - 1:
            self._hidx += 1
            return self._entries[self._hidx]
        self._hidx = None
        return self._draft

    def reset_navigation(self) -> None:
        """Drop back to live-draft mode (called on submit / clear)."""
        self._hidx = None
        self._draft = ""

    @property
    def entries(self) -> list[str]:
        return list(self._entries)


class _file_lock:
    """Context manager: hold an exclusive advisory lock on ``lock_path``.

    We lock a dedicated sibling ``*.lock`` file rather than the data file itself.
    ``os.replace`` swaps the data file's inode, so a lock taken on the data file
    would end up pinned to the old, now-unlinked inode while another writer locks
    the fresh one — breaking mutual exclusion exactly during the write. The lock
    file is never renamed, so its inode is stable and every process serializes on
    it. On non-POSIX platforms (no fcntl) this is a no-op.
    """

    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._fd: int | None = None

    def __enter__(self) -> "_file_lock":
        if fcntl is None:
            return self
        self._fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


def _write_atomic(file: Path, entries: list[str]) -> None:
    """Write JSON-lines to a temp file in the same dir, then atomically replace."""
    payload = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
    fd, tmp = tempfile.mkstemp(dir=str(file.parent), prefix=".history.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, file)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
