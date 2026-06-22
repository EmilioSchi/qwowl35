"""Insert command, mirroring hashline's ``commands/insert.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..anchor import parse_anchor, resolve, resolve_query_region
from ..document import Document
from ..mutation import insert_line
from ..output import write_success_line
from .common import atomic_write_document, interpret_escapes


@dataclass
class InsertCmd:
    file: Path
    anchor: str
    content: str
    before: bool = False
    start_query: str | None = None
    end_query: str | None = None
    interpret_escapes: bool = False


def run(cmd: InsertCmd) -> str:
    doc = Document.load(cmd.file)
    content = interpret_escapes(cmd.content) if cmd.interpret_escapes else cmd.content
    index = doc.build_index()
    if cmd.start_query is not None:
        region = resolve_query_region(doc, cmd.start_query, cmd.end_query)
        assert region is not None
        idx = region.start_line - 1
        anchor_line = region.start_line
        insert_at = idx if cmd.before else idx + 1
    else:
        anchor = parse_anchor(cmd.anchor)
        resolved = resolve(anchor, doc, index)
        anchor_line = resolved.line_no
        insert_at = resolved.index if cmd.before else resolved.index + 1
    insert_line(doc, insert_at, content)
    atomic_write_document(cmd.file, doc)
    summary = InsertSummary(anchor_line, insert_at + 1, content, cmd.before)
    return write_success_line(summary.success_message())


@dataclass(frozen=True)
class InsertSummary:
    anchor_line: int
    inserted_line: int
    content: str
    before: bool

    def success_message(self) -> str:
        return f"Inserted line {self.inserted_line}."


def write_dry_run(cmd: InsertCmd, summary: InsertSummary) -> str:
    relation = "before" if summary.before else "after"
    return (
        write_success_line(
            f"Would insert line {summary.inserted_line} {relation} line {summary.anchor_line}:"
        )
        + f"  + {summary.content!r}\n"
        + "No file was written.\n"
    )
