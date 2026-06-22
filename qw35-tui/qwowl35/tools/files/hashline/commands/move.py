"""Move command, mirroring hashline's ``commands/move.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..anchor import parse_anchor, resolve
from ..commands.common import atomic_write_document
from ..document import Document
from ..mutation import move_line
from ..output import write_success_line


@dataclass(frozen=True)
class MoveSummary:
    source_line: int
    target_line: int
    inserted_line: int


@dataclass
class MoveCmd:
    file: Path
    anchor: str
    target: str
    direction: str = "after"


def run(cmd: MoveCmd) -> str:
    doc = Document.load(cmd.file)
    index = doc.build_index()
    source = resolve(parse_anchor(cmd.anchor), doc, index)
    target = resolve(parse_anchor(cmd.target), doc, index)
    inserted = move_line(doc, source.index, target.index, cmd.direction == "before")
    atomic_write_document(cmd.file, doc)
    return write_success_line(f"Moved line {source.line_no} to line {inserted + 1}.")


def write_dry_run(cmd: MoveCmd, summary: MoveSummary) -> str:
    return write_success_line(f"Would move line {summary.source_line} to line {summary.inserted_line}.") + "No file was written.\n"
