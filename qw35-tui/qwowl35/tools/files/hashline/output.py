"""Pretty output helpers, mirroring hashline's ``output.rs``."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
import difflib
import json
from typing import Iterable

from .anchor import ResolvedLine
from .document import Document, FileStats, LineRecord, LineView, NewlineStyle
from .hash import format_short_hash, write_short_hash_bytes

MAX_DIFF_CHARS = 12_000


class JsonStyle:
    Compact = "compact"
    Pretty = "pretty"

    @staticmethod
    def from_pretty(pretty: bool) -> str:
        return JsonStyle.Pretty if pretty else JsonStyle.Compact


@dataclass(frozen=True)
class ErrorPayload:
    error: str


@dataclass(frozen=True)
class LineViewRef:
    n: int
    hash: str
    content: str


def serialize_json(value, style: str = JsonStyle.Compact) -> str:
    payload = _jsonable(value)
    if style == JsonStyle.Pretty:
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_json_string_fast(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def has_control_byte(bytes_: bytes) -> bool:
    return any(byte < 0x20 for byte in bytes_)


def write_success_line(line: str) -> str:
    return f"{line}\n"


def write_post_edit_snippet(doc: Document, first_changed: int, last_changed: int) -> str:
    context = 2
    max_output_lines = 12
    if not doc.lines:
        return ""
    lo = max(1, first_changed - context)
    hi = min(len(doc.lines), last_changed + context)
    if hi < lo:
        return ""
    if hi - lo + 1 > max_output_lines:
        return "(snippet omitted: changed range too large; use read with anchor LINE:HASH to inspect)\n"
    return "\n".join(line_view(i, doc.lines[i - 1]) for i in range(lo, hi + 1)) + "\n"


def print_read(doc: Document) -> str:
    return render_lines(doc, range(len(doc.lines))) + ("\n" if doc.lines else "")


def print_compact_read(doc: Document, anchors: list[str] | None = None, context: int = 5) -> str:
    return "\n".join(f"{idx + 1}:{format_short_hash(line.short_hash)}" for idx, line in enumerate(doc.lines)) + ("\n" if doc.lines else "")


def print_read_json(doc: Document, style: str = JsonStyle.Compact, compact: bool = False) -> str:
    payload = {
        "file": str(doc.path),
        "lines": [
            {"n": idx + 1, "hash": "" if compact else format_short_hash(line.short_hash), "content": line.content}
            for idx, line in enumerate(doc.lines)
        ],
    }
    return serialize_json(payload, style)


def print_read_json_streaming(doc: Document, style: str = JsonStyle.Compact, compact: bool = False) -> str:
    return print_read_json(doc, style, compact)


def print_read_ndjson_streaming(doc: Document, compact: bool = False) -> str:
    rows = [{"type": "header", "file": str(doc.path)}]
    rows.extend(
        {"n": idx + 1, "hash": "" if compact else format_short_hash(line.short_hash), "content": line.content}
        for idx, line in enumerate(doc.lines)
    )
    return "".join(serialize_json(row) for row in rows)


def print_read_context(
    doc: Document,
    ranges: Iterable[tuple[int, int]],
    context: int,
) -> str:
    indexes: set[int] = set()
    for start, end in ranges:
        lo = max(0, start - context)
        hi = min(len(doc.lines) - 1, end + context)
        indexes.update(range(lo, hi + 1))
    return render_lines(doc, sorted(indexes)) + ("\n" if indexes else "")


def print_index(doc: Document, compact: bool = False) -> str:
    if compact:
        return "\n".join(str(idx + 1) for idx, _line in enumerate(doc.lines)) + ("\n" if doc.lines else "")
    return "\n".join(f"{idx + 1}:{format_short_hash(line.short_hash)}" for idx, line in enumerate(doc.lines)) + ("\n" if doc.lines else "")


def print_index_json(doc: Document, compact: bool = False, style: str = JsonStyle.Compact) -> str:
    payload = {
        "file": str(doc.path),
        "index": [
            {"n": idx + 1, "hash": "" if compact else format_short_hash(line.short_hash)}
            for idx, line in enumerate(doc.lines)
        ],
    }
    return serialize_json(payload, style)


def print_read_ndjson(payload) -> str:
    if isinstance(payload, Document):
        return print_read_ndjson_streaming(payload)
    return serialize_json(payload)


def print_index_ndjson(payload) -> str:
    return serialize_json(payload)


def print_line_views_ndjson(lines: list[LineView]) -> str:
    return "".join(serialize_json(line) for line in lines)


def print_stats(stats: FileStats) -> str:
    lines = [
        f"Lines: {stats.line_count}",
        f"Unique hashes (2-char): {stats.unique_hashes}",
        f"Collisions: {stats.collision_count}",
        f"Collision pairs: {stats.collision_pair_count}",
        f"Est. read tokens: ~{stats.estimated_read_tokens}",
        f"Hash length advice: {stats.hash_length_advice}-char recommended",
        f"Suggested --context: {stats.suggested_context_n}",
        f"Recommended read mode: {stats.recommended_read_mode}",
        f"Recommended anchor mode: {stats.recommended_anchor_mode}",
        f"Recommended workflow: {stats.recommended_workflow}",
    ]
    if stats.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in stats.warnings)
    else:
        lines.append("Warnings: none")
    lines.append("Note: v1 anchors still use fixed 2-char hashes.")
    return "\n".join(lines) + "\n"


def print_stats_json(stats: FileStats, style: str = JsonStyle.Compact) -> str:
    return serialize_json(stats, style)


def print_grep(doc: Document, indexes: list[int]) -> str:
    return render_lines(doc, indexes) + ("\n" if indexes else "")


def print_grep_pretty_streaming(search_doc, pattern: str, invert: bool = False) -> str:
    return print_line_views(search_doc.grep_lines(pattern, invert))


def print_line_views(lines: list[LineView]) -> str:
    return "\n".join(f"{line.n}:{line.hash}|{line.content}" for line in lines) + ("\n" if lines else "")


def write_grep_json(lines: list[LineView], style: str = JsonStyle.Compact) -> str:
    return serialize_json(lines, style)


def print_grep_json_streaming(search_doc, pattern: str, invert: bool = False, style: str = JsonStyle.Compact) -> str:
    return write_grep_json(search_doc.grep_lines(pattern, invert), style)


def print_grep_ndjson_streaming(search_doc, pattern: str, invert: bool = False) -> str:
    return print_line_views_ndjson(search_doc.grep_lines(pattern, invert))


def write_error(message: str) -> str:
    return serialize_json({"error": message})


def write_json_success(value, style: str = JsonStyle.Compact) -> str:
    return serialize_json(value, style)


def line_view(line_no: int, line: LineRecord) -> str:
    return f"{line_no}:{format_short_hash(line.short_hash)}|{line.content}"


def render_lines(doc: Document, indexes: Iterable[int]) -> str:
    return "\n".join(line_view(idx + 1, doc.lines[idx]) for idx in indexes)


def write_short_hash(buf: bytearray, short: int) -> None:
    write_short_hash_bytes(buf, short)


def unified_diff(path: str, before: str, after: str) -> str:
    if before == after:
        return "(no text changes)"
    diff = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
            n=3,
        )
    )
    if len(diff) > MAX_DIFF_CHARS:
        return diff[:MAX_DIFF_CHARS] + "\n... (diff truncated)"
    return diff


def line_number_width(doc: Document) -> int:
    return max(1, len(str(len(doc.lines))))


def newline_name(newline: NewlineStyle) -> str:
    return "lf" if newline == NewlineStyle.Lf else "crlf"


def collect_context_indexes(doc: Document, anchors: list[ResolvedLine], context: int) -> list[int]:
    included: set[int] = set()
    for anchor in anchors:
        start = max(0, anchor.index - context)
        end = min(len(doc.lines) - 1, anchor.index + context)
        included.update(range(start, end + 1))
    return sorted(included)


def _jsonable(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
