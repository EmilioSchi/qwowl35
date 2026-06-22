"""Command orchestration payloads, mirroring hashline's ``orchestration.rs``."""

from __future__ import annotations

from dataclasses import dataclass

from .anchor import ResolvedLine, parse_anchor, resolve
from .document import Document, FileStats, LineView
from .hash import format_short_hash


@dataclass(frozen=True)
class ReadPayload:
    file: str
    lines: list[LineView]


@dataclass(frozen=True)
class IndexLineView:
    n: int
    hash: str


@dataclass(frozen=True)
class IndexPayload:
    file: str
    lines: list[IndexLineView]
    compact: bool = False


@dataclass(frozen=True)
class VerifyResult:
    anchor: str
    ok: bool
    line: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class VerifyReport:
    file: str
    results: list[VerifyResult]


@dataclass(frozen=True)
class DoctorPayload:
    file: str
    stats: FileStats
    next_commands: list[str]


def command_name(command) -> str:
    return command if isinstance(command, str) else command.__class__.__name__.removesuffix("Cmd").lower()


def resolve_read_anchors(doc: Document, anchors: list[str]) -> list[ResolvedLine]:
    index = doc.build_index()
    return [resolve(parse_anchor(anchor), doc, index) for anchor in anchors]


def read_payload(doc: Document, anchors: list[str] | None = None, context: int = 5, compact: bool = False) -> ReadPayload:
    if not anchors:
        indexes = range(len(doc.lines))
    else:
        resolved = resolve_read_anchors(doc, anchors)
        indexes = collect_context_indexes(doc, resolved, context)
    lines = [
        LineView(idx + 1, "" if compact else format_short_hash(doc.lines[idx].short_hash), doc.lines[idx].content)
        for idx in indexes
    ]
    return ReadPayload(str(doc.path), lines)


def index_payload(doc: Document) -> IndexPayload:
    return index_payload_with_compact(doc, False)


def index_payload_with_compact(doc: Document, compact: bool) -> IndexPayload:
    return IndexPayload(
        str(doc.path),
        [IndexLineView(idx + 1, "" if compact else format_short_hash(line.short_hash)) for idx, line in enumerate(doc.lines)],
        compact,
    )


def verify_report(doc: Document, anchors: list[str]) -> VerifyReport:
    results: list[VerifyResult] = []
    for anchor in anchors:
        try:
            resolved = resolve(parse_anchor(anchor), doc, doc.build_index())
            results.append(VerifyResult(anchor, True, resolved.line_no, None))
        except Exception as error:  # noqa: BLE001
            results.append(VerifyResult(anchor, False, None, str(error)))
    return VerifyReport(str(doc.path), results)


def doctor_payload(path, stats: FileStats) -> DoctorPayload:
    return DoctorPayload(str(path), stats, doctor_next_commands(stats))


def doctor_next_commands(stats: FileStats) -> list[str]:
    commands = ["hashline read"]
    if stats.collision_count:
        commands.insert(0, "hashline stats")
    if stats.estimated_read_tokens > 8000:
        commands.insert(0, "hashline index")
    return commands


def run(command) -> str:
    return run_command(command)


def run_command(command) -> str:
    from .commands import read, edit, insert, delete

    name = command_name(command)
    if name == "read":
        return read.run(command)
    if name == "edit":
        return edit.run(command)
    if name == "insert":
        return insert.run(command)
    if name == "delete":
        return delete.run(command)
    raise ValueError(f"unsupported command {name}")


def verify_line_no_for_error(error: Exception) -> int | None:
    return None


def verify_status_for_error(error: Exception) -> str:
    return "error"


def newline_name(newline) -> str:
    return "lf" if str(newline.value if hasattr(newline, "value") else newline) == "\n" else "crlf"


def collect_context_indexes(doc: Document, anchors: list[ResolvedLine], context: int) -> list[int]:
    included: set[int] = set()
    for anchor in anchors:
        start = max(0, anchor.index - context)
        end = min(len(doc.lines) - 1, anchor.index + context)
        included.update(range(start, end + 1))
    return sorted(included)
