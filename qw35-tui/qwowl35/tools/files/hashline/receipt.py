"""Mutation receipts, mirroring hashline's ``receipt.rs``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path


class ChangeKind(Enum):
    Edit = "edit"
    Insert = "insert"
    Delete = "delete"


@dataclass(frozen=True)
class LineChange:
    kind: ChangeKind | str
    line: int
    before: str | None = None
    after: str | None = None


@dataclass(frozen=True)
class Receipt:
    command: str
    file: str
    changes: list[LineChange]
    before_hash: str
    after_hash: str


@dataclass(frozen=True)
class DryRunReceipt:
    command: str
    file: str
    summary: str
    changes: list[LineChange]


def build_receipt(command: str, file: str | Path, changes: list[LineChange], before: bytes | str, after: bytes | str) -> Receipt:
    import hashlib

    before_bytes = before.encode("utf-8") if isinstance(before, str) else before
    after_bytes = after.encode("utf-8") if isinstance(after, str) else after
    return Receipt(
        command=command,
        file=str(file),
        changes=changes,
        before_hash=hashlib.sha256(before_bytes).hexdigest(),
        after_hash=hashlib.sha256(after_bytes).hexdigest(),
    )


def build_dry_run_receipt(command: str, file: str | Path, summary: str, changes: list[LineChange]) -> DryRunReceipt:
    return DryRunReceipt(command=command, file=str(file), summary=summary, changes=changes)


def write_receipt(receipt: Receipt) -> str:
    return json.dumps(asdict(receipt), default=_json_default, ensure_ascii=False) + "\n"


def write_dry_run_receipt(receipt: DryRunReceipt) -> str:
    return json.dumps(asdict(receipt), default=_json_default, ensure_ascii=False) + "\n"


def append_to_audit_log(receipt: Receipt, log_path: str | Path) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(write_receipt(receipt))


def write_audit_warning(log_path: str | Path, error: Exception) -> str:
    return f"Warning: could not write audit log {log_path}: {error}\n"


def _json_default(value):
    if isinstance(value, Enum):
        return value.value
    raise TypeError(type(value).__name__)
