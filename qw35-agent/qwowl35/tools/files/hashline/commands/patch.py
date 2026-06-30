"""Hashline patch command, mirroring hashline's ``commands/patch.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

from .delete import DeleteCmd, run as run_delete
from .edit import EditCmd, run as run_edit
from .insert import InsertCmd, run as run_insert


class OpType(Enum):
    Edit = "edit"
    Insert = "insert"
    Delete = "delete"


@dataclass
class EditOp:
    anchor: str
    content: str


@dataclass
class InsertOp:
    anchor: str
    content: str
    before: bool = False


@dataclass
class DeleteOp:
    anchor: str


@dataclass
class PatchOp:
    op: str
    anchor: str
    content: str = ""
    before: bool = False


@dataclass
class PatchFile:
    ops: list[PatchOp]


@dataclass
class PlannedOp:
    op: PatchOp


@dataclass
class PendingOp:
    op: PatchOp


@dataclass
class FastOp:
    op: PatchOp


@dataclass
class Occupancy:
    indexes: set[int]


@dataclass
class PatchResult:
    applied: int


@dataclass
class PatchSummary:
    applied: int


@dataclass
class PatchCmd:
    file: Path
    patch: str


def parse_hashline_patch(content: str) -> PatchFile:
    data = json.loads(content)
    ops = data.get("ops", data if isinstance(data, list) else [])
    return PatchFile([PatchOp(**op) for op in ops])


def read_patch(path_or_content: str) -> PatchFile:
    path = Path(path_or_content)
    if path.exists():
        return parse_hashline_patch(path.read_text(encoding="utf-8"))
    return parse_hashline_patch(path_or_content)


def build_lines(content: str) -> list[str]:
    return content.splitlines()


def build_plan(patch: PatchFile) -> list[PlannedOp]:
    return [PlannedOp(op) for op in patch.ops]


def mark_occupied(occupancy: Occupancy, index: int) -> None:
    occupancy.indexes.add(index)


def validate_patch_target(op: PatchOp) -> None:
    if not op.anchor:
        raise ValueError("patch op anchor is required")


def resolve_edit(op: PatchOp) -> PatchOp:
    validate_patch_target(op)
    return op


def resolve_insert(op: PatchOp) -> PatchOp:
    validate_patch_target(op)
    return op


def resolve_delete(op: PatchOp) -> PatchOp:
    validate_patch_target(op)
    return op


def finalize_op(op: PatchOp) -> PatchOp:
    return op


def apply_plan(file: Path, plan: list[PlannedOp]) -> PatchResult:
    applied = 0
    for planned in plan:
        op = planned.op
        if op.op == "edit":
            run_edit(EditCmd(file, op.anchor, op.content))
        elif op.op == "insert":
            run_insert(InsertCmd(file, op.anchor, op.content, before=op.before))
        elif op.op == "delete":
            run_delete(DeleteCmd(file, op.anchor))
        applied += 1
    return PatchResult(applied)


def apply_fast_op(file: Path, op: PatchOp) -> None:
    apply_plan(file, [PlannedOp(op)])


def run_fast_patch(cmd: PatchCmd) -> str:
    return run(cmd)


def patch_error(message: str) -> str:
    return f"patch failed: {message}"


def plural_suffix(count: int) -> str:
    return "" if count == 1 else "s"


def write_dry_run(cmd: PatchCmd, summary: PatchSummary) -> str:
    return f"Would apply {summary.applied} op{plural_suffix(summary.applied)}.\nNo file was written.\n"


def run(cmd: PatchCmd) -> str:
    patch = read_patch(cmd.patch)
    result = apply_plan(cmd.file, build_plan(patch))
    return f"Applied {result.applied} op{plural_suffix(result.applied)}.\n"
