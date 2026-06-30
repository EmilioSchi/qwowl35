"""Indent command, mirroring hashline's ``commands/indent.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..anchor import parse_range, resolve_range
from ..commands.common import atomic_write_document
from ..document import Document
from ..mutation import new_line_record
from ..output import write_success_line


@dataclass(frozen=True)
class IndentChange:
    amount: int
    unit: str = " "


@dataclass(frozen=True)
class IndentSummary:
    start_line: int
    end_line: int
    amount: int


@dataclass
class IndentCmd:
    file: Path
    anchor: str
    amount: str


def parse_indent_change(amount: str) -> IndentChange:
    sign = 1
    raw = amount
    if raw.startswith("+"):
        raw = raw[1:]
    elif raw.startswith("-"):
        sign = -1
        raw = raw[1:]
    return IndentChange(sign * int(raw), " ")


def validate_range_style(doc: Document, start: int, end: int) -> None:
    return None


def apply_indent(doc: Document, start: int, end: int, change: IndentChange) -> None:
    for idx in range(start, end + 1):
        content = doc.lines[idx].content
        if change.amount >= 0:
            updated = change.unit * change.amount + content
        else:
            remove = min(-change.amount, len(content) - len(content.lstrip(change.unit)))
            updated = content[remove:]
        doc.lines[idx] = new_line_record(updated)
    doc.content_len = sum(len(line.content) for line in doc.lines)


def run(cmd: IndentCmd) -> str:
    doc = Document.load(cmd.file)
    start, end = resolve_range(parse_range(cmd.anchor), doc, doc.build_index())
    change = parse_indent_change(cmd.amount)
    validate_range_style(doc, start.index, end.index)
    apply_indent(doc, start.index, end.index, change)
    atomic_write_document(cmd.file, doc)
    return write_success_line(f"Indented lines {start.line_no}-{end.line_no}.")


def write_dry_run(cmd: IndentCmd, summary: IndentSummary) -> str:
    return write_success_line(f"Would indent lines {summary.start_line}-{summary.end_line}.") + "No file was written.\n"
