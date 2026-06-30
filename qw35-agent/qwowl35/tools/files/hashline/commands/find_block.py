"""Block-finding helpers, mirroring hashline's ``commands/find_block.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..document import Document
from ..hash import format_short_hash


@dataclass(frozen=True)
class IndexLineView:
    n: int
    hash: str
    content: str


@dataclass(frozen=True)
class FindBlockPayload:
    language: str
    start_line: int
    end_line: int
    lines: list[IndexLineView]


def leading_whitespace(text: str) -> str:
    return text[: len(text) - len(text.lstrip(" \t"))]


def language_for_extension(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in {".rs", ".c", ".cpp", ".h", ".js", ".ts"}:
        return "brace"
    if suffix == ".rb":
        return "ruby"
    return "text"


def find_indent_block(doc: Document, target_idx: int) -> tuple[int, int]:
    base_indent = len(leading_whitespace(doc.lines[target_idx].content))
    start = target_idx
    while start > 0 and (not doc.lines[start - 1].content.strip() or len(leading_whitespace(doc.lines[start - 1].content)) >= base_indent):
        start -= 1
    end = target_idx
    while end + 1 < len(doc.lines) and (not doc.lines[end + 1].content.strip() or len(leading_whitespace(doc.lines[end + 1].content)) > base_indent):
        end += 1
    return start, end


def find_brace_pairs(doc: Document) -> list[tuple[int, int]]:
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for idx, line in enumerate(doc.lines):
        if "{" in line.content:
            stack.append(idx)
        if "}" in line.content and stack:
            pairs.append((stack.pop(), idx))
    return pairs


def find_brace_block(doc: Document, target_idx: int) -> tuple[int, int]:
    for start, end in find_brace_pairs(doc):
        if start <= target_idx <= end:
            return start, end
    return target_idx, target_idx


def find_python_block(doc: Document, target_idx: int) -> tuple[int, int]:
    return find_indent_block(doc, target_idx)


def ruby_opener_count(line: str) -> int:
    return sum(1 for token in ("def ", "class ", "module ", "do") if token in line)


def find_ruby_block(doc: Document, target_idx: int) -> tuple[int, int]:
    return find_indent_block(doc, target_idx)


def find_block_boundaries(doc: Document, target_idx: int) -> tuple[str, int, int]:
    language = language_for_extension(doc.path)
    if language == "python":
        start, end = find_python_block(doc, target_idx)
    elif language == "ruby":
        start, end = find_ruby_block(doc, target_idx)
    elif language == "brace":
        start, end = find_brace_block(doc, target_idx)
    else:
        start, end = target_idx, target_idx
    return language, start, end


def find_block_payload(doc: Document, target_idx: int) -> FindBlockPayload:
    language, start, end = find_block_boundaries(doc, target_idx)
    lines = [
        IndexLineView(idx + 1, format_short_hash(doc.lines[idx].short_hash), doc.lines[idx].content)
        for idx in range(start, end + 1)
    ]
    return FindBlockPayload(language, start + 1, end + 1, lines)


def run(cmd) -> str:
    doc = Document.load(cmd.file)
    payload = find_block_payload(doc, cmd.line - 1)
    return "\n".join(f"{line.n}:{line.hash}|{line.content}" for line in payload.lines) + "\n"
