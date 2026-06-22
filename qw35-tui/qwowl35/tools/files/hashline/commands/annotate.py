"""Annotate command, mirroring hashline's ``commands/annotate.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from ..document import Document
from ..output import print_grep


@dataclass
class AnnotateCmd:
    file: Path
    query: str
    regex: bool = False
    expect_one: bool = False


def search_literal(doc: Document, query: str) -> list[int]:
    return [idx for idx, line in enumerate(doc.lines) if query in line.content]


def search_regex(doc: Document, query: str) -> list[int]:
    compiled = re.compile(query)
    return [idx for idx, line in enumerate(doc.lines) if compiled.search(line.content)]


def run(cmd: AnnotateCmd) -> str:
    doc = Document.load(cmd.file)
    indexes = search_regex(doc, cmd.query) if cmd.regex else search_literal(doc, cmd.query)
    if cmd.expect_one and len(indexes) != 1:
        raise ValueError(f"expected one match, found {len(indexes)}")
    return print_grep(doc, indexes)
