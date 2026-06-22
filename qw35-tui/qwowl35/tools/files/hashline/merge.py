"""Three-way text merge helpers, mirroring hashline's ``merge.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SegKind(Enum):
    Base = "base"
    Target = "target"
    Current = "current"
    Conflict = "conflict"


class DiffOp(Enum):
    Equal = "equal"
    Insert = "insert"
    Delete = "delete"


@dataclass(frozen=True)
class Segment:
    kind: SegKind
    lines: list[str]


@dataclass(frozen=True)
class MergeResult:
    text: str
    conflicts: int


def merge_texts(base: str, target: str, current: str) -> MergeResult:
    if current == base:
        return MergeResult(target, 0)
    if target == base or target == current:
        return MergeResult(current, 0)
    merged = [
        "<<<<<<< current",
        *current.splitlines(),
        "=======",
        *target.splitlines(),
        ">>>>>>> target",
    ]
    suffix = "\n" if current.endswith("\n") or target.endswith("\n") else ""
    return MergeResult("\n".join(merged) + suffix, 1)


def lcs_diff(left: list[str], right: list[str]) -> list[tuple[DiffOp, str]]:
    import difflib

    ops: list[tuple[DiffOp, str]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=left, b=right).get_opcodes():
        if op == "equal":
            ops.extend((DiffOp.Equal, line) for line in left[i1:i2])
        elif op == "delete":
            ops.extend((DiffOp.Delete, line) for line in left[i1:i2])
        elif op == "insert":
            ops.extend((DiffOp.Insert, line) for line in right[j1:j2])
        else:
            ops.extend((DiffOp.Delete, line) for line in left[i1:i2])
            ops.extend((DiffOp.Insert, line) for line in right[j1:j2])
    return ops


def diff_ops(left: list[str], right: list[str]) -> list[tuple[DiffOp, str]]:
    return lcs_diff(left, right)


def merge_segments(segments: list[Segment]) -> str:
    return "\n".join(line for segment in segments for line in segment.lines)


def group_segments(ops: list[tuple[DiffOp, str]]) -> list[Segment]:
    segments: list[Segment] = []
    current_kind: SegKind | None = None
    current_lines: list[str] = []
    for op, line in ops:
        kind = SegKind.Base if op == DiffOp.Equal else SegKind.Target
        if current_kind is not None and kind != current_kind:
            segments.append(Segment(current_kind, current_lines))
            current_lines = []
        current_kind = kind
        current_lines.append(line)
    if current_kind is not None:
        segments.append(Segment(current_kind, current_lines))
    return segments


def merge_adjacent(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for segment in segments:
        if merged and merged[-1].kind == segment.kind:
            merged[-1].lines.extend(segment.lines)
        else:
            merged.append(Segment(segment.kind, list(segment.lines)))
    return merged


def merge_one(base: str, target: str, current: str) -> MergeResult:
    return merge_texts(base, target, current)


def emit_current(lines: list[str]) -> Segment:
    return Segment(SegKind.Current, lines)


def emit_target(lines: list[str]) -> Segment:
    return Segment(SegKind.Target, lines)


def push_conflict(segments: list[Segment], current: list[str], target: list[str]) -> None:
    segments.append(Segment(SegKind.Conflict, ["<<<<<<< current", *current, "=======", *target, ">>>>>>> target"]))
