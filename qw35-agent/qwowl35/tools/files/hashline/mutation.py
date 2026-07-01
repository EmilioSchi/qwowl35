"""Document mutations, mirroring hashline's ``mutation.rs``."""

from __future__ import annotations

from pathlib import Path

from .commands.common import atomic_write
from .document import Document, LineRecord, byte_len
from .error import HashlineError
from .hash import ShortHash, format_line_ref, format_short_hash, full_hash, short_from_full, short_hash_value
from .document import NewlineStyle


def validate_single_line_content(content: str) -> None:
    if "\n" in content or "\r" in content:
        raise HashlineError("multi-line content unsupported")


def split_content_lines(content: str) -> list[str]:
    if content == "":
        return [""]
    lines = content.splitlines()
    return lines if lines else [""]


def replace_line(doc: Document, index: int, content: str) -> None:
    validate_single_line_content(content)
    ensure_index(doc, index)
    old_len = byte_len(doc.lines[index].content)
    if doc.lines[index].content != content:
        doc.lines[index] = new_line_record(content)
    doc.content_len = doc.content_len + byte_len(doc.lines[index].content) - old_len


def replace_range_with_line(doc: Document, start: int, end: int, content: str) -> None:
    validate_single_line_content(content)
    replace_range(doc, start, end, content)


def replace_range(doc: Document, start: int, end: int, content: str) -> None:
    ensure_range(doc, start, end)
    replacement = split_content_lines(content)
    removed_len = sum(byte_len(line.content) for line in doc.lines[start : end + 1])
    inserted_len = sum(byte_len(line) for line in replacement)
    doc.lines[start : end + 1] = [new_line_record(line) for line in replacement]
    doc.content_len = doc.content_len + inserted_len - removed_len


def insert_line(doc: Document, index: int, content: str) -> None:
    ensure_insert_index(doc, index)
    lines = split_content_lines(content)
    total_len = sum(byte_len(line) for line in lines)
    for i, line in enumerate(lines):
        insert_at = index + i
        doc.lines.insert(insert_at, new_line_record(line))
    doc.content_len += total_len


def delete_line(doc: Document, index: int) -> None:
    ensure_index(doc, index)
    removed_len = byte_len(doc.lines[index].content)
    doc.lines.pop(index)
    doc.content_len -= removed_len


def delete_range(doc: Document, start: int, end: int) -> None:
    ensure_range(doc, start, end)
    removed_len = sum(byte_len(line.content) for line in doc.lines[start : end + 1])
    del doc.lines[start : end + 1]
    doc.content_len -= removed_len


def delete_adjacent_duplicates(doc: Document, should_drop) -> list[int]:
    """Drop each line that ``should_drop(prev_kept, current)`` flags as a duplicate
    of the previous *kept* line, collapsing a run of N identical lines to one.

    ``should_drop`` receives the two line contents and returns True to remove the
    current one. Returns the 1-based original line numbers that were removed (in
    the file's pre-removal numbering); empty when nothing matched. Mirrors
    :func:`delete_line`'s ``content_len`` accounting.
    """
    kept: list[LineRecord] = []
    removed: list[int] = []
    prev: LineRecord | None = None
    for index, line in enumerate(doc.lines):
        if prev is not None and should_drop(prev.content, line.content):
            removed.append(index + 1)
            doc.content_len -= byte_len(line.content)
            continue
        kept.append(line)
        prev = line
    if removed:
        doc.lines = kept
    return removed


def swap_lines(doc: Document, left: int, right: int) -> None:
    ensure_index(doc, left)
    ensure_index(doc, right)
    if left == right:
        raise HashlineError("source and target must resolve to different lines")
    doc.lines[left], doc.lines[right] = doc.lines[right], doc.lines[left]


def move_line(doc: Document, source: int, target: int, place_before: bool) -> int:
    ensure_index(doc, source)
    ensure_index(doc, target)
    if source == target:
        raise HashlineError("source and target must resolve to different lines")
    line = doc.lines.pop(source)
    adjusted_target = target - 1 if source < target else target
    insert_at = adjusted_target if place_before else adjusted_target + 1
    doc.lines.insert(insert_at, line)
    return insert_at


def refresh_line_metadata(line: LineRecord) -> LineRecord:
    return LineRecord(content=line.content, short_hash=short_from_full(full_hash(line.content)))


def new_line_record(content: str) -> LineRecord:
    full = full_hash(content)
    return LineRecord(content=content, short_hash=short_from_full(full))


def ensure_index(doc: Document, index: int) -> None:
    if 0 <= index < len(doc.lines):
        return
    raise HashlineError(f"mutation index out of bounds: index={index}, len={len(doc.lines)}")


def ensure_insert_index(doc: Document, index: int) -> None:
    if 0 <= index <= len(doc.lines):
        return
    raise HashlineError(f"mutation index out of bounds: index={index}, len={len(doc.lines)}")


def ensure_range(doc: Document, start: int, end: int) -> None:
    if 0 <= start <= end < len(doc.lines):
        return
    raise HashlineError(f"invalid mutation range: start={start}, end={end}, len={len(doc.lines)}")


def stream_replace_line(
    path: str | Path,
    target_line: int,
    new_content: str,
    expected_hash: ShortHash,
    newline: NewlineStyle,
    trailing_newline: bool,
) -> None:
    validate_single_line_content(new_content)
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8")
    separator = newline.separator()
    lines = raw.splitlines()
    if target_line < 0 or target_line >= len(lines):
        raise HashlineError(f"mutation index out of bounds: index={target_line}, len={len(lines)}")
    actual_hash = short_hash_value(lines[target_line])
    if actual_hash != expected_hash:
        raise HashlineError(
            f"stale anchor {format_line_ref(target_line + 1, expected_hash)} in {file_path}; "
            f"current line hash is {format_short_hash(actual_hash)}."
        )
    lines[target_line] = new_content
    rendered = separator.join(lines)
    if trailing_newline and lines:
        rendered += separator
    atomic_write(file_path, rendered)
