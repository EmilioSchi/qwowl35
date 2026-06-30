"""Hash sidecar helpers, mirroring hashline's ``hash_cache.rs``."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class HashSidecar:
    mtime_secs: int
    size: int
    content_hash: int
    short_hashes: list[int]

    @staticmethod
    def store_path(root: str | Path, source: str | Path) -> Path:
        source_path = Path(source)
        safe = str(source_path.resolve()).replace("/", "_").replace(":", "_")
        return Path(root) / f"{safe}.json"

    @staticmethod
    def ensure_dir(root: str | Path, source: str | Path | None = None) -> None:
        Path(root).mkdir(parents=True, exist_ok=True)

    def write(self, root: str | Path, source: str | Path) -> None:
        self.ensure_dir(root, source)
        self.store_path(root, source).write_text(json.dumps(self.__dict__), encoding="utf-8")

    @classmethod
    def read(cls, root: str | Path, source: str | Path) -> "HashSidecar":
        data = json.loads(cls.store_path(root, source).read_text(encoding="utf-8"))
        return cls(**data)

    @classmethod
    def exists(cls, root: str | Path, source: str | Path) -> bool:
        return cls.store_path(root, source).exists()

    @classmethod
    def invalidate(cls, root: str | Path, source: str | Path) -> None:
        try:
            cls.store_path(root, source).unlink()
        except FileNotFoundError:
            pass


def discover_sidecar_root(path: str | Path) -> Path:
    return Path(path).resolve().parent / ".hashline"
