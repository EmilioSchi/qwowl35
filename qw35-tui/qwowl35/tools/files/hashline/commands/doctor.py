"""Doctor command, mirroring hashline's ``commands/doctor.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..document import Document
from ..orchestration import doctor_payload
from ..output import serialize_json


@dataclass
class DoctorCmd:
    file: Path


def run(cmd: DoctorCmd) -> str:
    doc = Document.load(cmd.file)
    return serialize_json(doctor_payload(cmd.file, doc.compute_stats()))
