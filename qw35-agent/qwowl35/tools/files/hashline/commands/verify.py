"""Verify command, mirroring hashline's ``commands/verify.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..document import Document
from ..orchestration import verify_report
from ..output import serialize_json


@dataclass
class VerifyCmd:
    file: Path
    anchors: list[str]


def run(cmd: VerifyCmd) -> str:
    return serialize_json(verify_report(Document.load(cmd.file), cmd.anchors))
