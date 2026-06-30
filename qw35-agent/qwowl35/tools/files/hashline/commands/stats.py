"""Stats command, mirroring hashline's ``commands/stats.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..document import Document
from ..output import print_stats


@dataclass
class StatsCmd:
    file: Path


def run(cmd: StatsCmd) -> str:
    return print_stats(Document.load(cmd.file).compute_stats())
