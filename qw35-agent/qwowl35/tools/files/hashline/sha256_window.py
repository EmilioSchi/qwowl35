"""Window hashing helpers, mirroring hashline's ``sha256_window.rs``."""

from __future__ import annotations

import hashlib

from .error import HashlineError


def hash_window(content: str, start_line: int, end_line: int) -> str:
    lines = content.splitlines()
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        raise HashlineError("invalid hash window")
    window = "\n".join(lines[start_line - 1 : end_line])
    return hashlib.sha256(window.encode("utf-8")).hexdigest()


def verify_anchor(content: str, start_line: int, end_line: int, expected_hash: str) -> bool:
    return hash_window(content, start_line, end_line) == expected_hash


def apply_edit_within_window(content: str, start_line: int, end_line: int, replacement: str) -> str:
    lines = content.splitlines()
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        raise HashlineError("invalid edit window")
    new_lines = replacement.splitlines() or [""]
    rendered = lines[: start_line - 1] + new_lines + lines[end_line:]
    suffix = "\n" if content.endswith("\n") else ""
    return "\n".join(rendered) + suffix
