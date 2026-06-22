"""Swap command, mirroring hashline's ``commands/swap.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..anchor import parse_anchor, resolve
from ..commands.common import atomic_write_document
from ..document import Document
from ..mutation import swap_lines
from ..output import write_success_line


@dataclass(frozen=True)
class SwapSummary:
    left_line: int
    right_line: int


@dataclass
class SwapCmd:
    file: Path
    anchor_a: str
    anchor_b: str


def run(cmd: SwapCmd) -> str:
    doc = Document.load(cmd.file)
    index = doc.build_index()
    left = resolve(parse_anchor(cmd.anchor_a), doc, index)
    right = resolve(parse_anchor(cmd.anchor_b), doc, index)
    swap_lines(doc, left.index, right.index)
    atomic_write_document(cmd.file, doc)
    return write_success_line(f"Swapped lines {left.line_no} and {right.line_no}.")


def write_dry_run(cmd: SwapCmd, summary: SwapSummary) -> str:
    return write_success_line(f"Would swap lines {summary.left_line} and {summary.right_line}.") + "No file was written.\n"
