"""Document loading and rendering, mirroring hashline's ``document.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import os
from pathlib import Path
from typing import Any

from .error import HashlineError
from .hash import ShortHash, full_hash, short_from_full

ShortHashIndex = list[list[int]]


class NewlineStyle(str, Enum):
    Lf = "\n"
    Crlf = "\r\n"

    def separator(self) -> str:
        return self.value


@dataclass(frozen=True)
class LineView:
    n: int
    hash: str
    content: str


@dataclass(frozen=True)
class FileMeta:
    mtime_secs: int
    mtime_nanos: int
    inode: int
    size: int
    change_secs: int
    change_nanos: int


@dataclass(frozen=True)
class LineRecord:
    content: str
    short_hash: ShortHash


@dataclass(frozen=True)
class FileStats:
    line_count: int
    unique_hashes: int
    collision_count: int
    collision_pairs: list[tuple[int, int]]
    collision_pair_count: int
    collision_pairs_truncated: bool
    estimated_read_tokens: int
    hash_length_advice: int
    suggested_context_n: int
    recommended_read_mode: str
    recommended_anchor_mode: str
    recommended_workflow: str
    warnings: list[str]


@dataclass(frozen=True)
class StreamingDocument:
    path: Path
    newline: NewlineStyle
    trailing_newline: bool
    line_count: int

    @classmethod
    def scan(cls, path: str | Path) -> "StreamingDocument":
        file_path = Path(path)
        raw = file_path.read_bytes()
        path_string = str(file_path)
        if b"\0" in raw[:8000]:
            raise HashlineError(f"binary file is not supported: {path_string}")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HashlineError(f"invalid UTF-8 in {path_string}") from exc
        newline, trailing_newline, lines, _content_len = parse_document_content(content, file_path)
        return cls(file_path, newline, trailing_newline, len(lines))

    def len(self) -> int:
        return self.line_count

    def is_empty(self) -> bool:
        return self.line_count == 0


@dataclass
class Document:
    path: Path
    newline: NewlineStyle
    trailing_newline: bool
    lines: list[LineRecord]
    content_len: int
    file_meta: Any | None = None
    short_hash_index: ShortHashIndex | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Document":
        file_path = Path(path)
        raw = file_path.read_bytes()
        path_string = str(file_path)
        if b"\0" in raw[:8000]:
            raise HashlineError(f"binary file is not supported: {path_string}")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HashlineError(f"invalid UTF-8 in {path_string}") from exc
        newline, trailing_newline, lines, content_len = parse_document_content(content, file_path)
        file_meta = read_file_meta(file_path)
        return cls(file_path, newline, trailing_newline, lines, content_len, file_meta=file_meta)

    @classmethod
    def load_with_hash_cache(cls, path: str | Path, root: str | Path | None = None) -> "Document":
        return cls.load(path)

    @classmethod
    def from_str(cls, path: str | Path, content: str) -> "Document":
        file_path = Path(path)
        newline, trailing_newline, lines, content_len = parse_document_content(content, file_path)
        return cls(file_path, newline, trailing_newline, lines, content_len)

    def build_index(self) -> ShortHashIndex:
        counts = count_short_hashes(self.lines)
        return build_index_from_counts(self.lines, counts)

    @staticmethod
    def build_index_cached(doc: "Document") -> ShortHashIndex:
        if doc.short_hash_index is None:
            counts = count_short_hashes(doc.lines)
            doc.short_hash_index = build_index_from_counts(doc.lines, counts)
        return doc.short_hash_index

    def render(self) -> str:
        if not self.lines:
            return ""
        separator = self.newline.separator()
        rendered = separator.join(line.content for line in self.lines)
        if self.trailing_newline:
            rendered += separator
        return rendered

    def write_to(self, writer) -> None:
        writer.write(self.render())

    def compute_stats(self) -> FileStats:
        counts = count_short_hashes(self.lines)
        unique_hashes, collision_count = summarize_bucket_counts(counts)
        short_hash_index = build_index_from_counts(self.lines, counts)
        collision_pairs, collision_pair_count = collect_collision_pairs_sample(
            short_hash_index,
            COLLISION_PAIRS_SAMPLE_CAP,
        )
        estimated_read_tokens = estimate_read_tokens(self)
        hash_length_advice = recommend_hash_length(self)
        suggested_context_n = suggest_context_n(self)
        recommended_read_mode = recommend_read_mode(self, estimated_read_tokens)
        recommended_anchor_mode = recommend_anchor_mode(self, collision_count, hash_length_advice)
        recommended_workflow = recommend_workflow(self, estimated_read_tokens, collision_count)
        warnings = collect_warnings(self, estimated_read_tokens, collision_count, hash_length_advice)
        return FileStats(
            line_count=len(self.lines),
            unique_hashes=unique_hashes,
            collision_count=collision_count,
            collision_pairs=collision_pairs,
            collision_pair_count=collision_pair_count,
            collision_pairs_truncated=collision_pair_count > len(collision_pairs),
            estimated_read_tokens=estimated_read_tokens,
            hash_length_advice=hash_length_advice,
            suggested_context_n=suggested_context_n,
            recommended_read_mode=recommended_read_mode,
            recommended_anchor_mode=recommended_anchor_mode,
            recommended_workflow=recommended_workflow,
            warnings=warnings,
        )

    def len(self) -> int:
        return len(self.lines)

    def is_empty(self) -> bool:
        return not self.lines


@dataclass
class SearchDocument:
    path: Path
    content: str
    newline: NewlineStyle
    trailing_newline: bool
    line_offsets: list[int]

    @classmethod
    def load(cls, path: str | Path) -> "SearchDocument":
        file_path = Path(path)
        raw = file_path.read_bytes()
        path_string = str(file_path)
        if b"\0" in raw[:8000]:
            raise HashlineError(f"binary file is not supported: {path_string}")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HashlineError(f"invalid UTF-8 in {path_string}") from exc
        newline, trailing_newline, line_offsets = parse_line_offsets(content)
        return cls(file_path, content, newline, trailing_newline, line_offsets)

    @classmethod
    def new(cls, content: str) -> "SearchDocument":
        newline, trailing_newline, line_offsets = parse_line_offsets(content)
        return cls(Path("demo.txt"), content, newline, trailing_newline, line_offsets)

    def grep_lines(self, pattern: str, invert: bool = False) -> list[LineView]:
        results: list[LineView] = []
        self.grep_for_each(
            pattern,
            invert,
            lambda line_idx, content, short_hash: results.append(
                LineView(line_idx + 1, format_short_hash(short_hash), content)
            ),
        )
        return results

    def grep_for_each(self, pattern: str, invert: bool, sink) -> None:
        for line_idx, start in enumerate(self.line_offsets):
            end = self.line_offsets[line_idx + 1] if line_idx + 1 < len(self.line_offsets) else len(self.content)
            line_end = end
            if line_end > start and self.content[line_end - 1 : line_end] == "\n":
                line_end -= 1
            line_content = self.content[start:line_end].removesuffix("\r")
            is_match = pattern in line_content
            include = (not is_match) if invert else is_match
            if include:
                sink(line_idx, line_content, short_from_full(full_hash(line_content)))


def parse_document_content(
    content: str,
    path: Path,
) -> tuple[NewlineStyle, bool, list[LineRecord], int]:
    trailing_newline = content.endswith("\n")
    estimated_line_count = content.count("\n")
    return parse_document_content_sequential(content, content.encode("utf-8"), path, trailing_newline, estimated_line_count)


def parse_document_content_sequential(
    content: str,
    bytes_: bytes,
    path: Path,
    trailing_newline: bool,
    estimated_line_count: int,
) -> tuple[NewlineStyle, bool, list[LineRecord], int]:
    if content == "":
        return NewlineStyle.Lf, False, [], 0

    saw_lf = False
    saw_crlf = False
    saw_bare_cr = False
    newline = NewlineStyle.Lf
    lines: list[LineRecord] = []
    start = 0
    search_from = 0
    content_len = 0

    while search_from < len(content):
        lf = content.find("\n", search_from)
        cr = content.find("\r", search_from)
        positions = [pos for pos in (lf, cr) if pos != -1]
        if not positions:
            break
        index = min(positions)
        if content[index] == "\r":
            if index + 1 < len(content) and content[index + 1] == "\n":
                saw_crlf = True
                newline = NewlineStyle.Crlf
                line = content[start:index]
                content_len += byte_len(line)
                lines.append(build_line_record(line))
                search_from = index + 2
                start = search_from
            else:
                saw_bare_cr = True
                search_from = index + 1
        else:
            saw_lf = True
            line = content[start:index]
            content_len += byte_len(line)
            lines.append(build_line_record(line))
            search_from = index + 1
            start = search_from

    if saw_bare_cr or (saw_crlf and saw_lf):
        raise HashlineError(f"mixed newlines are not supported: {path}")

    if not trailing_newline and start < len(content):
        line = content[start:]
        content_len += byte_len(line)
        lines.append(build_line_record(line))

    return newline, trailing_newline, lines, content_len


def parse_document_content_parallel(
    content: str,
    bytes_: bytes,
    path: Path,
    trailing_newline: bool,
    estimated_line_count: int,
) -> tuple[NewlineStyle, bool, list[LineRecord], int]:
    return parse_document_content_sequential(content, bytes_, path, trailing_newline, estimated_line_count)


def parse_line_offsets(content: str) -> tuple[NewlineStyle, bool, list[int]]:
    if content == "":
        return NewlineStyle.Lf, False, [0]
    saw_lf = False
    saw_crlf = False
    saw_bare_cr = False
    newline = NewlineStyle.Lf
    trailing_newline = content.endswith("\n")
    line_offsets = [0]
    search_from = 0
    while search_from < len(content):
        lf = content.find("\n", search_from)
        cr = content.find("\r", search_from)
        positions = [pos for pos in (lf, cr) if pos != -1]
        if not positions:
            break
        index = min(positions)
        if content[index] == "\r":
            if index + 1 < len(content) and content[index + 1] == "\n":
                saw_crlf = True
                newline = NewlineStyle.Crlf
                search_from = index + 2
                line_offsets.append(search_from)
            else:
                saw_bare_cr = True
                search_from = index + 1
        else:
            saw_lf = True
            search_from = index + 1
            line_offsets.append(search_from)

    if saw_bare_cr or (saw_crlf and saw_lf):
        return NewlineStyle.Lf, trailing_newline, line_offsets
    return newline, trailing_newline, line_offsets


def build_line_record(content: str) -> LineRecord:
    full = full_hash(content)
    return LineRecord(content=content, short_hash=short_from_full(full))


def byte_len(content: str) -> int:
    return len(content.encode("utf-8"))


def format_short_hash(short_hash: ShortHash) -> str:
    from .hash import format_short_hash as _format_short_hash

    return _format_short_hash(short_hash)


def count_short_hashes(lines: list[LineRecord]) -> list[int]:
    counts = [0] * 256
    for line in lines:
        counts[line.short_hash] += 1
    return counts


def build_index_from_counts(lines: list[LineRecord], counts: list[int]) -> ShortHashIndex:
    short_hash_index: ShortHashIndex = [[] for _ in range(256)]
    for idx, line in enumerate(lines):
        short_hash_index[line.short_hash].append(idx)
    return short_hash_index


def read_file_meta(path: str | Path) -> FileMeta:
    stat = Path(path).stat()
    change_secs, change_nanos = change_time_from_metadata(stat)
    return FileMeta(
        mtime_secs=int(stat.st_mtime),
        mtime_nanos=int(getattr(stat, "st_mtime_ns", 0) % 1_000_000_000),
        inode=inode_from_metadata(stat),
        size=stat.st_size,
        change_secs=change_secs,
        change_nanos=change_nanos,
    )


def build_lines_from_hashes_with_meta(
    short_hashes: list[int],
    content: str,
) -> tuple[NewlineStyle, bool, list[LineRecord], int]:
    newline, trailing_newline, lines, content_len = parse_document_content(content, Path(""))
    if len(short_hashes) == len(lines):
        lines = [LineRecord(line.content, short_hashes[idx]) for idx, line in enumerate(lines)]
    return newline, trailing_newline, lines, content_len


def build_lines_from_hashes(short_hashes: list[int], content: str) -> list[LineRecord]:
    return build_lines_from_hashes_with_meta(short_hashes, content)[2]


def empty_index() -> ShortHashIndex:
    return [[] for _ in range(256)]


COLLISION_PAIRS_SAMPLE_CAP = 1024


def summarize_bucket_counts(counts: list[int]) -> tuple[int, int]:
    unique_hashes = 0
    collision_count = 0
    for count in counts:
        if count == 0:
            continue
        unique_hashes += 1
        if count >= 2:
            collision_count += count
    return unique_hashes, collision_count


def collect_collision_pairs_sample(
    index: ShortHashIndex,
    sample_cap: int,
) -> tuple[list[tuple[int, int]], int]:
    total = 0
    sample: list[tuple[int, int]] = []
    for positions in (positions for positions in index if len(positions) >= 2):
        n = len(positions)
        total += n * (n - 1) // 2
        if len(sample) < sample_cap:
            for left in range(len(positions)):
                for right in range(left + 1, len(positions)):
                    sample.append((positions[left] + 1, positions[right] + 1))
                    if len(sample) >= sample_cap:
                        break
                if len(sample) >= sample_cap:
                    break
    return sample, total


def estimate_read_tokens(doc: Document) -> int:
    anchor_overhead = len(doc.lines) * 8
    return (doc.content_len + anchor_overhead) // 4


def recommend_hash_length(doc: Document) -> int:
    line_count = doc.len()
    for hash_len in (2, 3, 4):
        buckets = 16.0**hash_len
        if collision_probability(line_count, buckets) < 0.01:
            return hash_len
    return 4


def collision_probability(line_count: int, buckets: float) -> float:
    if line_count <= 1:
        return 0.0
    n = float(line_count)
    return 1.0 - math.exp(-(n * (n - 1.0)) / (2.0 * buckets))


def suggest_context_n(doc: Document) -> int:
    markers = [
        idx + 1
        for idx, line in enumerate(doc.lines)
        if is_structure_marker(line.content)
    ]
    if len(markers) < 2:
        return 5
    gaps = sorted(markers[idx + 1] - markers[idx] for idx in range(len(markers) - 1))
    median_gap = gaps[len(gaps) // 2]
    return max(3, min(20, median_gap // 2))


def recommend_read_mode(doc: Document, estimated_read_tokens: int) -> str:
    if doc.is_empty() or (estimated_read_tokens <= 2000 and doc.len() <= 400):
        return "read"
    if estimated_read_tokens <= 8000:
        return "read --anchor <line:hash> --context N"
    return "index or read --anchor <line:hash> --context N"


def recommend_anchor_mode(doc: Document, collision_count: int, hash_length_advice: int) -> str:
    if doc.is_empty() or collision_count > 0 or doc.len() >= 200 or hash_length_advice > 2:
        return "qualified"
    return "bare-or-qualified"


def recommend_workflow(doc: Document, estimated_read_tokens: int, collision_count: int) -> str:
    if doc.is_empty():
        return "read-empty-file"
    if collision_count > 0:
        return "stats -> annotate/grep -> read --anchor --context -> edit/patch -> verify"
    if estimated_read_tokens > 8000:
        return "index -> annotate/grep -> read --anchor --context -> edit/patch -> verify"
    return "read -> annotate/grep -> verify -> edit/patch -> verify"


def collect_warnings(
    doc: Document,
    estimated_read_tokens: int,
    collision_count: int,
    hash_length_advice: int,
) -> list[str]:
    warnings: list[str] = []
    if collision_count > 0:
        warnings.append("short-hash collisions detected; prefer qualified anchors like 12:ab")
    if hash_length_advice > 2:
        warnings.append("2-char hashes may be cramped for this file; use stats and qualified anchors to avoid ambiguity")
    if estimated_read_tokens > 8000:
        warnings.append("full read output will be expensive; orient with index/stats, then narrow with --anchor and --context")
    if doc.len() > 2000:
        warnings.append("large file: prefer patch/find-block workflows over many tiny edits")
    return warnings


def is_structure_marker(content: str) -> bool:
    return any(marker in content for marker in ("function ", "def ", "class ", "fn ", "impl "))


def inode_from_metadata(metadata: os.stat_result) -> int:
    return int(getattr(metadata, "st_ino", 0))


def change_time_from_metadata(metadata: os.stat_result) -> tuple[int, int]:
    ctime_ns = int(getattr(metadata, "st_ctime_ns", 0))
    if ctime_ns:
        return ctime_ns // 1_000_000_000, ctime_ns % 1_000_000_000
    return int(getattr(metadata, "st_ctime", 0)), 0
