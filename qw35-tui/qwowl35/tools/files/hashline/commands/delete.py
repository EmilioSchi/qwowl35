"""Delete command, mirroring hashline's ``commands/delete.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..anchor import (
    looks_like_range_anchor,
    parse_anchor,
    parse_range,
    resolve,
    resolve_query_region,
    resolve_range,
)
from ..document import Document
from ..mutation import delete_line, delete_range
from ..output import write_success_line
from .common import atomic_write_document


@dataclass
class DeleteCmd:
    file: Path
    anchor: str
    start_query: str | None = None
    end_query: str | None = None


def run(cmd: DeleteCmd) -> str:
    doc = Document.load(cmd.file)
    index = doc.build_index()
    if cmd.start_query is not None:
        region = resolve_query_region(doc, cmd.start_query, cmd.end_query)
        assert region is not None
        start_idx = region.start_line - 1
        end_idx = region.end_line - 1
        delete_range(doc, start_idx, end_idx)
        summary = DeleteSummary.range(region.start_line, region.end_line)
    elif looks_like_range_anchor(cmd.anchor):
        range_anchor = parse_range(cmd.anchor)
        start, end = resolve_range(range_anchor, doc, index)
        delete_range(doc, start.index, end.index)
        summary = DeleteSummary.range(start.line_no, end.line_no)
    else:
        anchor = parse_anchor(cmd.anchor)
        resolved = resolve(anchor, doc, index)
        delete_line(doc, resolved.index)
        summary = DeleteSummary.single(resolved.line_no)

    atomic_write_document(cmd.file, doc)
    return write_success_line(summary.success_message())


@dataclass(frozen=True)
class DeleteSummary:
    start_line: int
    end_line: int
    single: bool

    @classmethod
    def single(cls, line_no: int) -> "DeleteSummary":
        return cls(line_no, line_no, True)

    @classmethod
    def range(cls, start_line: int, end_line: int) -> "DeleteSummary":
        return cls(start_line, end_line, False)

    def success_message(self) -> str:
        if self.single:
            return f"Deleted line {self.start_line}."
        return f"Deleted lines {self.start_line}-{self.end_line}."


class DeleteSummaryKind(Enum):
    Single = "single"
    Range = "range"


def write_dry_run(cmd: DeleteCmd, summary: DeleteSummary) -> str:
    if summary.single:
        return write_success_line(f"Would delete line {summary.start_line}.") + "No file was written.\n"
    return write_success_line(f"Would delete lines {summary.start_line}-{summary.end_line}.") + "No file was written.\n"
