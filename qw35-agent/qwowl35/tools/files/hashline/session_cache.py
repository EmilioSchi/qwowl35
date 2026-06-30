"""Session document cache, mirroring hashline's ``session_cache.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .document import Document, FileStats


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0


@dataclass
class CacheEntry:
    doc_value: Document

    def stats(self) -> FileStats:
        return self.doc_value.compute_stats()

    def doc(self) -> Document:
        return self.doc_value

    def doc_mut(self) -> Document:
        return self.doc_value


class SessionCache:
    def __init__(self, max_entries: int = 64) -> None:
        self.max_entries = max_entries
        self.entries: dict[Path, CacheEntry] = {}
        self.cache_stats = CacheStats()
        self.no_cache = False

    @classmethod
    def new(cls, max_entries: int) -> "SessionCache":
        return cls(max_entries)

    def get_or_load(self, path: str | Path) -> CacheEntry:
        key = Path(path)
        if not self.no_cache and key in self.entries:
            self.cache_stats.hits += 1
            return self.entries[key]
        self.cache_stats.misses += 1
        entry = CacheEntry(Document.load(key))
        if not self.no_cache:
            if len(self.entries) >= self.max_entries:
                self.entries.pop(next(iter(self.entries)))
            self.entries[key] = entry
        return entry

    def invalidate(self, path: str | Path) -> None:
        self.entries.pop(Path(path), None)

    def after_mutation(self, path: str | Path, doc: Document) -> None:
        self.entries[Path(path)] = CacheEntry(doc)

    def peek(self, path: str | Path) -> Document | None:
        entry = self.entries.get(Path(path))
        return entry.doc_value if entry else None

    def stats(self) -> CacheStats:
        return self.cache_stats

    def clear(self) -> None:
        self.entries.clear()

    def set_no_cache(self, enabled: bool) -> None:
        self.no_cache = enabled
