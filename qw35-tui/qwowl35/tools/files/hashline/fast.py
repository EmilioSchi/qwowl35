"""Fast-path helpers, mirroring hashline's ``fast.rs``."""

from __future__ import annotations

from pathlib import Path

from .anchor import parse_anchor, resolve, try_parse_line_anchor
from .commands.common import atomic_write, interpret_escapes as _interpret_escapes
from .document import Document
from .error import HashlineError
from .hash import ShortHash, short_hash_value
from .mutation import delete_range, insert_line, move_line, replace_range, swap_lines


def read_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def interpret_escapes(input: str) -> str:
    return _interpret_escapes(input)


def get_line_content(content: str, line: int) -> str | None:
    lines = content.splitlines()
    return lines[line] if 0 <= line < len(lines) else None


def get_line_range(content: str, start: int, end: int) -> list[str]:
    return content.splitlines()[start:end]


def atomic_write(path: str | Path, content: str) -> None:  # type: ignore[no-redef]
    from .commands.common import atomic_write as _atomic_write

    _atomic_write(Path(path), content)


def fast_from_hash(content: str, hash: ShortHash) -> int:
    matches = [idx for idx, line in enumerate(content.splitlines()) if short_hash_value(line) == hash]
    if not matches:
        raise HashlineError(f"hash {hash:02x} not found")
    if len(matches) > 1:
        raise HashlineError(f"hash {hash:02x} ambiguous")
    return matches[0]


def fast_fuzzy_resolve(content: str, line_no: int, hash: ShortHash) -> int | None:
    lines = content.splitlines()
    idx = line_no
    if 0 <= idx < len(lines) and short_hash_value(lines[idx]) == hash:
        return idx
    candidates = [i for i, line in enumerate(lines) if short_hash_value(line) == hash]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        closest = min(candidates, key=lambda candidate: abs(candidate - idx))
        if abs(closest - idx) <= 3:
            return closest
    return None


def fast_find_query(content: str, query: str) -> int:
    matches = [idx for idx, line in enumerate(content.splitlines()) if query in line]
    if not matches:
        raise HashlineError(f"query {query!r} not found")
    if len(matches) > 1:
        raise HashlineError(f"query {query!r} ambiguous")
    return matches[0]


def fast_replace_line(path: str | Path, line_no: int, hash: ShortHash, content: str, *args, **kwargs) -> str:
    return run_fast_edit(None, path, line_no, hash, content, *args, **kwargs)


def fast_replace_range(path: str | Path, start_line: int, start_hash: ShortHash, end_line: int, end_hash: ShortHash, content: str, *args, **kwargs) -> str:
    return run_fast_range_edit(None, path, start_line, start_hash, end_line, end_hash, content, *args, **kwargs)


def fast_insert_line(path: str | Path, line_no: int, hash: ShortHash, content: str, *args, **kwargs) -> str:
    return run_fast_insert(None, path, line_no, hash, content, *args, **kwargs)


def fast_insert_line_before(path: str | Path, line_no: int, hash: ShortHash, content: str, *args, **kwargs) -> str:
    return run_fast_insert(None, path, line_no, hash, content, before=True)


def fast_delete_lines(path: str | Path, start_line: int, end_line: int, hash: ShortHash, *args, **kwargs) -> str:
    return run_fast_delete(None, path, start_line, end_line, hash, *args, **kwargs)


def fast_swap_lines(path: str | Path, left: int, right: int) -> str:
    doc = Document.load(path)
    swap_lines(doc, left, right)
    atomic_write(path, doc.render())
    return "Swapped lines."


def fast_move_line(path: str | Path, source: int, target: int, place_before: bool) -> str:
    doc = Document.load(path)
    move_line(doc, source, target, place_before)
    atomic_write(path, doc.render())
    return "Moved line."


def fast_indent_lines(path: str | Path, start: int, end: int, prefix: str) -> str:
    doc = Document.load(path)
    for idx in range(start, end + 1):
        doc.lines[idx] = type(doc.lines[idx])(prefix + doc.lines[idx].content, short_hash_value(prefix + doc.lines[idx].content))
    atomic_write(path, doc.render())
    return "Indented lines."


def run_fast_edit(ctx, file, line_no, hash, content, *args, **kwargs) -> str:
    doc = Document.load(file)
    idx = fast_fuzzy_resolve(doc.render(), line_no, hash)
    if idx is None:
        raise HashlineError("stale anchor")
    replace_range(doc, idx, idx, content)
    atomic_write(file, doc.render())
    return f"Edited line {idx + 1}."


def run_fast_range_edit(ctx, file, s_line, s_hash, e_line, e_hash, content, *args, **kwargs) -> str:
    doc = Document.load(file)
    start = fast_fuzzy_resolve(doc.render(), s_line, s_hash)
    end = fast_fuzzy_resolve(doc.render(), e_line, e_hash)
    if start is None or end is None:
        raise HashlineError("stale anchor")
    replace_range(doc, start, end, content)
    atomic_write(file, doc.render())
    return f"Edited lines {start + 1}-{end + 1}."


def run_fast_insert(ctx, file, line_no, hash, content, *args, before: bool = False, **kwargs) -> str:
    doc = Document.load(file)
    idx = fast_fuzzy_resolve(doc.render(), line_no, hash)
    if idx is None:
        raise HashlineError("stale anchor")
    insert_at = idx if before else idx + 1
    insert_line(doc, insert_at, content)
    atomic_write(file, doc.render())
    return f"Inserted line {insert_at + 1}."


def run_fast_delete(ctx, file, start_line, end_line, hash, *args, **kwargs) -> str:
    doc = Document.load(file)
    start = fast_fuzzy_resolve(doc.render(), start_line, hash)
    if start is None:
        raise HashlineError("stale anchor")
    delete_range(doc, start, end_line)
    atomic_write(file, doc.render())
    return f"Deleted lines {start + 1}-{end_line + 1}."


def run_fast_swap(ctx, file, left, right, *args, **kwargs) -> str:
    return fast_swap_lines(file, left, right)


def run_fast_move(ctx, file, source, target, place_before, *args, **kwargs) -> str:
    return fast_move_line(file, source, target, place_before)


def run_fast_indent(ctx, file, start, end, prefix, *args, **kwargs) -> str:
    return fast_indent_lines(file, start, end, prefix)


def run_fast_query_edit(ctx, file, query, content, *args, **kwargs) -> str:
    line_no = fast_find_query(read_file(file), query)
    hash = short_hash_value(get_line_content(read_file(file), line_no) or "")
    return run_fast_edit(ctx, file, line_no, hash, content, *args, **kwargs)


def try_fuzzy_recover(content: str, line_no: int, hash: ShortHash) -> int | None:
    return fast_fuzzy_resolve(content, line_no, hash)


def try_recover_edit(content: str, anchor: str) -> int | None:
    parsed = try_parse_line_anchor(anchor)
    if parsed is None:
        return None
    return fast_fuzzy_resolve(content, parsed[0], parsed[1])


def repair_replacement(old_lines: list[str], new_lines: list[str]) -> list[str]:
    return new_lines


def make_range_changes(start: int, end: int, replacement: list[str]) -> list[tuple[int, str]]:
    return [(start + idx, line) for idx, line in enumerate(replacement)]


def find_line_span_inner(content: str, line_no: int) -> tuple[int, int] | None:
    start = 0
    for _ in range(line_no):
        pos = content.find("\n", start)
        if pos == -1:
            return None
        start = pos + 1
    end = content.find("\n", start)
    return start, len(content) if end == -1 else end


def handle_receipt(*args, **kwargs) -> str:
    return ""


def check_guards(*args, **kwargs) -> None:
    return None
