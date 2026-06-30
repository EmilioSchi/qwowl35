"""Read command, mirroring hashline's ``commands/read.rs``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..anchor import looks_like_range_anchor, parse_anchor, parse_range, resolve, resolve_range
from ..document import Document
from ..output import print_read, print_read_context


@dataclass
class ReadCmd:
    file: Path
    anchor: list[str] = field(default_factory=list)
    context: int = 5
    compact: bool = False


def run(cmd: ReadCmd) -> str:
    doc = Document.load(cmd.file)
    if not cmd.anchor:
        return print_read(doc)

    index = doc.build_index()
    ranges: list[tuple[int, int]] = []
    for raw_anchor in cmd.anchor:
        if looks_like_range_anchor(raw_anchor):
            range_anchor = parse_range(raw_anchor)
            start, end = resolve_range(range_anchor, doc, index)
            ranges.append((start.index, end.index))
        else:
            resolved = resolve(parse_anchor(raw_anchor), doc, index)
            ranges.append((resolved.index, resolved.index))
    return print_read_context(doc, ranges, cmd.context)
