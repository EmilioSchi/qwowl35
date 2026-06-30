"""Unified diff apply command, mirroring hashline's ``commands/diff_apply.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from ..commands.common import atomic_write
from ..error import HashlineError


@dataclass(frozen=True)
class Conflict:
    reason: str


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


@dataclass(frozen=True)
class DiffReceipt:
    applied: int
    conflicts: list[Conflict]


def parse_range_pair(text: str) -> tuple[int, int]:
    if "," in text:
        left, right = text.split(",", 1)
        return int(left), int(right)
    return int(text), 1


def parse_hunk_header(header: str) -> tuple[int, int, int, int]:
    match = re.search(r"@@ -([0-9,]+) \\+([0-9,]+) @@", header)
    if not match:
        raise HashlineError(f"invalid hunk header: {header}")
    old_start, old_count = parse_range_pair(match.group(1))
    new_start, new_count = parse_range_pair(match.group(2))
    return old_start, old_count, new_start, new_count


def parse_unified_diff(diff_content: str) -> list[Hunk]:
    hunks: list[Hunk] = []
    current: Hunk | None = None
    for line in diff_content.splitlines():
        if line.startswith("@@"):
            if current is not None:
                hunks.append(current)
            old_start, old_count, new_start, new_count = parse_hunk_header(line)
            current = Hunk(old_start, old_count, new_start, new_count, [])
        elif current is not None:
            current.lines.append(line)
    if current is not None:
        hunks.append(current)
    return hunks


def find_context_in_lines(lines: list[str], context: list[str]) -> int | None:
    for idx in range(0, len(lines) - len(context) + 1):
        if lines[idx : idx + len(context)] == context:
            return idx
    return None


def apply_diff(path: str | Path, diff_content: str) -> DiffReceipt:
    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8").splitlines()
    offset = 0
    applied = 0
    for hunk in parse_unified_diff(diff_content):
        idx = hunk.old_start - 1 + offset
        replacement: list[str] = []
        consumed = 0
        for line in hunk.lines:
            if line.startswith("+"):
                replacement.append(line[1:])
            elif line.startswith("-"):
                consumed += 1
            elif line.startswith(" "):
                replacement.append(line[1:])
                consumed += 1
        lines[idx : idx + consumed] = replacement
        offset += len(replacement) - consumed
        applied += 1
    atomic_write(file_path, "\n".join(lines) + "\n")
    return DiffReceipt(applied, [])


def run(cmd) -> str:
    receipt = apply_diff(cmd.file, cmd.diff)
    return f"Applied {receipt.applied} hunk(s).\n"
