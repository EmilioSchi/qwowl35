"""Anchor parsing and resolution, mirroring hashline's ``anchor.rs``."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .document import Document, ShortHashIndex
from .error import HashlineError
from .hash import ShortHash, format_line_ref, format_short_hash

FUZZY_RELOCATE_RADIUS = 3

# A qualified locator is line digits + a fixed 2-hex content hash with no
# separator (e.g. "12af"). The former ":" was dropped for token efficiency; it
# is still accepted as an OPTIONAL separator so a stray "12:af" from the model
# still parses. The hash is always the final two hex chars, so the split is
# unambiguous (line = the leading digits).
_QUALIFIED_ANCHOR = re.compile(r"(\d+):?([0-9a-f]{2})")


@dataclass(frozen=True)
class Anchor:
    short: ShortHash | None = None
    line: int | None = None
    block_line: int | None = None
    # Positional keyword anchor: "start" (first line) or "end" (last line).
    keyword: str | None = None


@dataclass(frozen=True)
class RangeAnchor:
    start: Anchor
    end: Anchor


@dataclass(frozen=True)
class ResolvedLine:
    index: int
    line_no: int
    short_hash: str


@dataclass(frozen=True)
class RegionPattern:
    start_line: int
    end_line: int


def parse_anchor(s: str) -> Anchor:
    trimmed = s.strip()
    lower = trimmed.lower()
    # "start"/"end" are positional anchors that resolve to the first/last line of
    # the file at execution time. They carry no hash to verify (like block
    # anchors), so they are always fresh.
    if lower in ("start", "end"):
        return Anchor(keyword=lower)
    if lower.startswith("block ") and lower.endswith(":"):
        line_str = lower.removeprefix("block ")[:-1]
        line = parse_line_number(line_str, s)
        return Anchor(block_line=line)

    normalized = normalize_anchor_input(trimmed)
    if ".." in normalized:
        raise HashlineError(f"invalid anchor {trimmed!r}")
    qualified = _QUALIFIED_ANCHOR.fullmatch(normalized)
    if qualified:
        line = parse_line_number(qualified.group(1), s)
        short = parse_short_hash(qualified.group(2), s)
        return Anchor(line=line, short=short)
    short = parse_short_hash(normalized, s)
    return Anchor(short=short)


def parse_range(s: str) -> RangeAnchor:
    normalized = normalize_anchor_input(s)
    parts = normalized.split("..")
    if len(parts) != 2:
        raise HashlineError(f"invalid range {s.strip()!r}")
    left, right = parts
    start = parse_anchor(left)
    end = parse_anchor(right)
    # Endpoints must be positional: a line:hash anchor or a start/end keyword.
    # A bare unqualified hash has no position and cannot bound a range.
    if not _is_positional(start) or not _is_positional(end):
        raise HashlineError(f"invalid range {s.strip()!r}")
    return RangeAnchor(start=start, end=end)


def _is_positional(anchor: Anchor) -> bool:
    return anchor.line is not None or anchor.keyword is not None


def looks_like_range_anchor(s: str) -> bool:
    normalized = normalize_anchor_input(s)
    if ".." in normalized:
        return True
    if "-" not in normalized:
        return False
    left, right = normalized.split("-", 1)
    try:
        parse_anchor(left)
        parse_anchor(right)
    except HashlineError:
        return False
    return True


def resolve(anchor: Anchor, doc: Document, index: ShortHashIndex) -> ResolvedLine:
    if anchor.keyword is not None:
        return resolve_keyword_anchor(anchor.keyword, doc)
    if anchor.block_line is not None:
        return resolve_block_anchor(anchor.block_line, doc)
    if anchor.line is None:
        if anchor.short is None:
            raise HashlineError("invalid anchor")
        return resolve_unqualified(anchor.short, doc, index)
    if anchor.short is None:
        raise HashlineError("invalid anchor")
    return resolve_qualified(anchor.line, anchor.short, doc, index)


def resolve_without_index(anchor: Anchor, doc: Document) -> ResolvedLine:
    if anchor.keyword is not None:
        return resolve_keyword_anchor(anchor.keyword, doc)
    if anchor.block_line is not None:
        return resolve_block_anchor(anchor.block_line, doc)
    if anchor.line is None:
        index = doc.build_index()
        if anchor.short is None:
            raise HashlineError("invalid anchor")
        return resolve_unqualified(anchor.short, doc, index)
    if anchor.short is None:
        raise HashlineError("invalid anchor")
    try:
        return resolve_qualified(anchor.line, anchor.short, doc, None)
    except HashlineError as error:
        if not str(error).startswith("stale anchor"):
            raise
        index = doc.build_index()
        return resolve_qualified(anchor.line, anchor.short, doc, index)


def resolve_range(
    range: RangeAnchor,
    doc: Document,
    index: ShortHashIndex,
) -> tuple[ResolvedLine, ResolvedLine]:
    start = resolve(range.start, doc, index)
    end = resolve(range.end, doc, index)
    if start.index > end.index:
        raise HashlineError(f"invalid range {display_anchor(range.start)}..{display_anchor(range.end)}")
    return start, end


def resolve_all(
    anchors: list[Anchor],
    doc: Document,
    index: ShortHashIndex,
) -> list[ResolvedLine | HashlineError]:
    resolved: list[ResolvedLine | HashlineError] = []
    for anchor in anchors:
        try:
            resolved.append(resolve(anchor, doc, index))
        except HashlineError as error:
            resolved.append(error)
    return resolved


def resolve_unqualified(short: ShortHash, doc: Document, index: ShortHashIndex) -> ResolvedLine:
    path = str(doc.path)
    rendered_short = format_short_hash(short)
    matches = index[short]
    if not matches:
        raise HashlineError(f"hash {rendered_short} not found in {path}")
    if len(matches) > 1:
        lines = ", ".join(str(idx + 1) for idx in matches)
        raise HashlineError(
            f"hash {rendered_short} is ambiguous in {path}; it appears on lines {lines}"
        )
    resolved_index = matches[0]
    return ResolvedLine(resolved_index, resolved_index + 1, rendered_short)


def resolve_qualified(
    line: int,
    short: ShortHash,
    doc: Document,
    index: ShortHashIndex | None,
) -> ResolvedLine:
    path = str(doc.path)
    rendered_short = format_short_hash(short)
    idx = line - 1
    if idx < 0 or idx >= len(doc.lines):
        raise HashlineError(f"invalid anchor {line}{rendered_short}")

    actual = doc.lines[idx]
    if actual.short_hash == short:
        return ResolvedLine(idx, line, rendered_short)

    candidates: list[int] = []
    if index is not None:
        candidates = index[short]
        relocated = None
        if len(candidates) == 1:
            relocated = candidates[0]
        elif candidates:
            target = idx
            closest = min(candidates, key=lambda candidate: abs(candidate - target))
            dist = abs(closest - target)
            if dist <= FUZZY_RELOCATE_RADIUS:
                relocated = closest

        if relocated is not None:
            return ResolvedLine(relocated, relocated + 1, rendered_short)

    relocated_suffix = stale_anchor_context(doc, idx, rendered_short, candidates)
    raise HashlineError(
        f"stale anchor {line}{rendered_short} in {path}; current line hash is "
        f"{format_short_hash(actual.short_hash)}.{relocated_suffix}"
    )


def stale_anchor_context(
    doc: Document,
    idx: int,
    rendered_short: str,
    candidates: list[int],
) -> str:
    context_radius = 2
    lo = max(0, idx - context_radius)
    hi = min(len(doc.lines) - 1, idx + context_radius)
    context = "\n"
    for i in range(lo, hi + 1):
        line_record = doc.lines[i]
        prefix = ">>> " if i == idx else "    "
        line_no = i + 1
        hash_text = format_short_hash(line_record.short_hash)
        content = line_record.content
        display = content[:80] + "..." if len(content) > 80 else content
        context += f"{prefix}{line_no}{hash_text}|{display}\n"
    if candidates:
        lines = ", ".join(str(idx + 1) for idx in candidates)
        context += f"(hash {rendered_short} also at line(s) {lines})\n"
    return context.rstrip("\n")


def find_line_by_query(doc: Document, query: str) -> int:
    path = str(doc.path)
    matches = [idx + 1 for idx, line in enumerate(doc.lines) if query in line.content]
    if not matches:
        raise HashlineError(f"query {query!r} not found in {path}")
    if len(matches) > 1:
        lines = ", ".join(str(line) for line in matches)
        raise HashlineError(f"query {query!r} is ambiguous in {path}; it appears on lines {lines}")
    return matches[0]


def resolve_query_region(
    doc: Document,
    start_query: str | None,
    end_query: str | None,
) -> RegionPattern | None:
    if start_query is None:
        return None

    start_line = find_line_by_query(doc, start_query)
    end_line = find_line_by_query(doc, end_query) if end_query is not None else start_line

    if start_line > end_line:
        raise HashlineError(f"invalid range query start (line {start_line}) after query end (line {end_line})")

    count = end_line - start_line + 1
    max_query_range = 10_000
    if count > max_query_range:
        raise HashlineError(f"query range too large: {count} lines; max is {max_query_range}")

    return RegionPattern(start_line=start_line, end_line=end_line)


def try_parse_line_anchor(anchor: str) -> tuple[int, ShortHash] | None:
    normalized = anchor.strip()
    if ".." in normalized:
        return None
    try:
        parsed = parse_anchor(normalized)
    except HashlineError:
        return None
    if parsed.line is None or parsed.short is None:
        return None
    return parsed.line - 1, parsed.short


def looks_like_block_anchor(s: str) -> int | None:
    trimmed = s.strip()
    lower = trimmed.lower()
    if not lower.startswith("block ") or not lower.endswith(":"):
        return None
    try:
        line = int(lower.removeprefix("block ")[:-1])
    except ValueError:
        return None
    return line if line != 0 else None


def resolve_block_anchor(line: int, doc: Document) -> ResolvedLine:
    idx = line - 1
    if idx < 0 or idx >= len(doc.lines):
        raise HashlineError(f"invalid anchor block {line}:")
    return ResolvedLine(idx, line, "")


def resolve_keyword_anchor(keyword: str, doc: Document) -> ResolvedLine:
    """Resolves a positional ``start``/``end`` anchor to the first/last line.

    Returns the line's real current hash, so the result is indistinguishable
    from a fresh ``line:hash`` anchor and flows through mutation/verification
    unchanged.
    """
    if not doc.lines:
        raise HashlineError(
            "file is empty; mutations cannot create files or target empty files. "
            "Use bash to create the file, then read for anchors before editing."
        )
    idx = 0 if keyword == "start" else len(doc.lines) - 1
    record = doc.lines[idx]
    return ResolvedLine(idx, idx + 1, format_short_hash(record.short_hash))


def normalize_anchor_input(s: str) -> str:
    return s.strip().lower()


def parse_short_hash(short: str, original: str) -> ShortHash:
    if len(short) == 2 and all(ch in "0123456789abcdefABCDEF" for ch in short):
        return int(short, 16)
    raise HashlineError(f"invalid anchor {original.strip()!r}")


def parse_line_number(raw: str, original: str) -> int:
    try:
        line = int(raw)
    except ValueError as exc:
        raise HashlineError(f"invalid anchor {original.strip()!r}") from exc
    if line == 0:
        raise HashlineError(f"invalid anchor {original.strip()!r}")
    return line


def display_anchor(anchor: Anchor) -> str:
    if anchor.keyword is not None:
        return anchor.keyword
    if anchor.block_line is not None:
        return f"block {anchor.block_line}:"
    if anchor.line is None:
        return format_short_hash(anchor.short or 0)
    return format_line_ref(anchor.line, anchor.short or 0)
