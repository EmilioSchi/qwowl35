"""Grep command, mirroring hashline's ``commands/grep.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from ..document import Document, SearchDocument
from ..output import print_grep, print_line_views


@dataclass
class GrepCmd:
    file: Path
    pattern: str
    invert: bool = False
    case_insensitive: bool = False
    regex: bool = False


def grep_lines_regex(doc: Document, pattern: str, invert: bool = False, case_insensitive: bool = False) -> list[int]:
    flags = re.IGNORECASE if case_insensitive else 0
    compiled = re.compile(pattern, flags)
    matches = [idx for idx, line in enumerate(doc.lines) if bool(compiled.search(line.content)) != invert]
    return matches


def run(cmd: GrepCmd) -> str:
    if cmd.regex:
        doc = Document.load(cmd.file)
        return print_grep(doc, grep_lines_regex(doc, cmd.pattern, cmd.invert, cmd.case_insensitive))
    pattern = cmd.pattern.lower() if cmd.case_insensitive else cmd.pattern
    search = SearchDocument.load(cmd.file)
    if cmd.case_insensitive:
        lines = [line for line in search.grep_lines("", False) if (pattern in line.content.lower()) != cmd.invert]
    else:
        lines = search.grep_lines(cmd.pattern, cmd.invert)
    return print_line_views(lines)
