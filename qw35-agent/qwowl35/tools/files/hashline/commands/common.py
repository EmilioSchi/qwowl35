"""Shared command helpers, mirroring hashline's ``commands/common.rs``."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from ..document import Document
from ..error import HashlineError


def interpret_escapes(input: str) -> str:
    out: list[str] = []
    chars = iter(range(len(input)))
    idx = 0
    while idx < len(input):
        ch = input[idx]
        if ch != "\\":
            out.append(ch)
            idx += 1
            continue
        idx += 1
        if idx >= len(input):
            out.append("\\")
            break
        escaped = input[idx]
        idx += 1
        if escaped == "n":
            out.append("\n")
        elif escaped == "r":
            out.append("\r")
        elif escaped == "t":
            out.append("\t")
        elif escaped == "0":
            out.append("\0")
        elif escaped == "\\":
            out.append("\\")
        elif escaped == '"':
            out.append('"')
        elif escaped == "'":
            out.append("'")
        else:
            out.append("\\")
            out.append(escaped)
    return "".join(out)


def check_guard(
    doc: Document,
    expect_mtime: int | None = None,
    expect_inode: int | None = None,
) -> None:
    meta = doc.file_meta
    if meta is None:
        return
    if (expect_mtime is not None and expect_mtime != meta.mtime_secs) or (
        expect_inode is not None and expect_inode != meta.inode
    ):
        raise HashlineError(f"stale file: {doc.path}")


def atomic_write(path: Path, content: str) -> None:
    atomic_write_with(path, lambda handle: handle.write(content))


def atomic_write_document(path: Path, doc: Document) -> None:
    atomic_write_with(path, lambda handle: handle.write(doc.render()))


def atomic_write_with(path: Path, write_contents) -> None:
    parent = path.parent if path.parent != Path("") else Path(".")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            write_contents(handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            permissions = path.stat().st_mode
            os.chmod(tmp_name, permissions)
        except OSError:
            pass
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    sync_parent_directory(parent)


def persist_with_retry(tmp_name: str, path: Path) -> None:
    os.replace(tmp_name, path)


def sync_parent_directory(parent: Path) -> None:
    try:
        fd = os.open(parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
