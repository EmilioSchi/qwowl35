"""Batch command, mirroring hashline's ``commands/batch.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .delete import DeleteCmd, run as run_delete
from .edit import EditCmd, run as run_edit
from .insert import InsertCmd, run as run_insert


@dataclass
class EditOp:
    op: str
    anchor: str
    content: str = ""


@dataclass
class ResolvedOp:
    op: EditOp


@dataclass
class BatchReceipt:
    applied: int


@dataclass
class BatchCmd:
    file: Path
    edits: list[EditOp]


def resolve_all_ops(cmd: BatchCmd) -> list[ResolvedOp]:
    return [ResolvedOp(op) for op in cmd.edits]


def batch_edit(cmd: BatchCmd) -> BatchReceipt:
    applied = 0
    for edit in cmd.edits:
        if edit.op == "edit":
            run_edit(EditCmd(cmd.file, edit.anchor, edit.content))
        elif edit.op == "insert":
            run_insert(InsertCmd(cmd.file, edit.anchor, edit.content))
        elif edit.op == "delete":
            run_delete(DeleteCmd(cmd.file, edit.anchor))
        applied += 1
    return BatchReceipt(applied)


def run(cmd: BatchCmd) -> str:
    receipt = batch_edit(cmd)
    return f"Applied {receipt.applied} edit(s).\n"
