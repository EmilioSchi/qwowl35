"""Edit command, mirroring hashline's ``commands/edit.rs``."""

from __future__ import annotations

from dataclasses import dataclass
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
from ..hash import format_short_hash, short_hash_value
from ..mutation import replace_line, replace_range, split_content_lines, stream_replace_line
from ..output import write_post_edit_snippet, write_success_line
from .common import atomic_write_document, interpret_escapes


@dataclass
class EditCmd:
    file: Path
    anchor: str
    content: str
    start_query: str | None = None
    end_query: str | None = None
    interpret_escapes: bool = False


def run(cmd: EditCmd) -> str:
    doc = Document.load(cmd.file)
    content = interpret_escapes(cmd.content) if cmd.interpret_escapes else cmd.content
    index = doc.build_index()

    if cmd.start_query is not None:
        region = resolve_query_region(doc, cmd.start_query, cmd.end_query)
        assert region is not None
        start_idx = region.start_line - 1
        end_idx = region.end_line - 1
        after = split_content_lines(content)
        replace_range(doc, start_idx, end_idx, content)
        summary = EditSummary.range(region.start_line, region.end_line, after)
    elif looks_like_range_anchor(cmd.anchor):
        range_anchor = parse_range(cmd.anchor)
        start, end = resolve_range(range_anchor, doc, index)
        after = split_content_lines(content)
        replace_range(doc, start.index, end.index, content)
        summary = EditSummary.range(start.line_no, end.line_no, after)
    else:
        anchor = parse_anchor(cmd.anchor)
        resolved = resolve(anchor, doc, index)
        if "\n" in content or "\r" in content:
            after = split_content_lines(content)
            replace_range(doc, resolved.index, resolved.index, content)
            summary = EditSummary.range(resolved.line_no, resolved.line_no, after)
        else:
            replace_line(doc, resolved.index, content)
            summary = EditSummary.single(resolved.line_no)

    atomic_write_document(cmd.file, doc)
    first, last = summary.changed_window()
    return write_success_line(summary.success_message()) + write_post_edit_snippet(doc, first, last)


def run_streaming(cmd: EditCmd) -> str:
    doc = Document.load(cmd.file)
    anchor = parse_anchor(cmd.anchor)
    if anchor.line is None or anchor.short is None:
        raise ValueError("streaming edit requires a qualified line:hash anchor")
    content = interpret_escapes(cmd.content) if cmd.interpret_escapes else cmd.content
    stream_replace_line(cmd.file, anchor.line - 1, content, anchor.short, doc.newline, doc.trailing_newline)
    return write_success_line(f"Edited line {anchor.line}.")


def atomic_write_single_line_edit(
    path: Path,
    doc: Document,
    target_line: int,
    before: str,
    after: str,
) -> bool:
    if "\n" in before or "\r" in before or "\n" in after or "\r" in after:
        return False
    if target_line < 0 or target_line >= len(doc.lines):
        return False
    expected_hash = short_hash_value(before)
    actual_hash = doc.lines[target_line].short_hash
    if actual_hash != expected_hash:
        return False
    stream_replace_line(path, target_line, after, expected_hash, doc.newline, doc.trailing_newline)
    return True


def original_line_byte_span(content: str, target_line: int) -> tuple[int, int] | None:
    if target_line < 0:
        return None
    start = 0
    for line_no in range(target_line):
        next_newline = content.find("\n", start)
        if next_newline == -1:
            return None
        start = next_newline + 1
    end = content.find("\n", start)
    if end == -1:
        end = len(content)
    if end > start and content[end - 1] == "\r":
        end -= 1
    return start, end


def write_dry_run(cmd: EditCmd, summary: "EditSummary") -> str:
    if summary.single:
        return write_success_line(f"Would edit line {summary.start_line}.") + "No file was written.\n"
    return write_success_line(f"Would edit lines {summary.start_line}-{summary.end_line}.") + "No file was written.\n"


@dataclass(frozen=True)
class EditSummary:
    start_line: int
    end_line: int
    after_count: int
    single: bool

    @classmethod
    def single(cls, line_no: int) -> "EditSummary":
        return cls(line_no, line_no, 1, True)

    @classmethod
    def range(cls, start_line: int, end_line: int, after: list[str]) -> "EditSummary":
        return cls(start_line, end_line, max(1, len(after)), False)

    def success_message(self) -> str:
        if self.single:
            return f"Edited line {self.start_line}."
        return f"Edited lines {self.start_line}-{self.end_line}."

    def changed_window(self) -> tuple[int, int]:
        last = self.start_line + self.after_count - 1
        return self.start_line, max(self.start_line, last)
