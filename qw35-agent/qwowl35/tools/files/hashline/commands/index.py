"""Index command, mirroring hashline's ``commands/index.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..document import Document
from ..output import print_index


@dataclass
class IndexCmd:
    file: Path
    compact: bool = False


def run(cmd: IndexCmd) -> str:
    return print_index(Document.load(cmd.file), cmd.compact)
