"""Per-content-type shrink passes for tool results.

Strategies adapted from Headroom (github.com/headroomlabs-ai/headroom,
Apache-2.0): lossless-first, keep first/last/errors, cluster repeated lines.
Mostly pure text passes over strings the tools already produced; the code
lane additionally routes through optional, guarded content-type detection
(Magika when installed, extension fallback — tools.compress.detect) into
tree-sitter comment pruning (tools.compress.comments), both fail-open and
costing nothing beyond the already-required grammars and Magika's small ONNX
model.

Every collapse leaves an explicit in-place count (`[line repeated × N]`,
`[… N lines elided]`), so nothing disappears silently; comment pruning is the
exception — it reports its total once, via the count `compress_code` returns,
which the framework in ``tools.compress`` folds into the single global marker
alongside the recovery instruction, instead of a marker per comment block.
"""

from __future__ import annotations

import json
import re

from .comments import prune_comment_lines
from .detect import detect_language
from .rerank import rerank_prose

# Lines matching this are load-bearing and are never clustered or elided.
PROTECTED = re.compile(
    r"error|warn|fail|fatal|exception|traceback|panic|denied|critical|assert",
    re.IGNORECASE,
)

# compress_log knobs.
REPEAT_MIN = 3  # identical-line run collapse threshold
SIMILAR_MIN = 5  # masked near-duplicate run collapse threshold
LOG_BUDGET = 12_000  # chars; beyond this, head+tail elision kicks in
LOG_HEAD_LINES = 30
LOG_TAIL_LINES = 40

# compress_code knob: only flagrant repetition (generated/data files) collapses.
CODE_REPEAT_MIN = 8

# compress_grep knobs.
MAX_MATCHES_PER_FILE = 8
GREP_LINE_MAX = 200

# compress_web knobs.
NAV_RUN_MIN = 10  # consecutive short link-ish lines that count as boilerplate
NAV_LINE_MAX = 40
WEB_BUDGET = 30_000
WEB_HEAD_CHARS = 24_000
WEB_TAIL_CHARS = 4_000
# Bodies still bigger than this after the lossless passes get the query-aware
# rerank lane (when a query exists); smaller ones aren't worth it.
RERANK_MIN_BODY = 9_000

_MASK_HEX = re.compile(r"0x[0-9a-fA-F]+")
_MASK_NUM = re.compile(r"\d+")


def _is_protected(line: str) -> bool:
    stripped = line.strip()
    # The bash result's section anchors must survive every pass.
    if stripped.startswith(("stderr:", "Exit code:")):
        return True
    return bool(PROTECTED.search(line))


def _mask(line: str) -> str:
    """Signature for near-duplicate clustering: numbers/addresses wildcarded."""
    return _MASK_NUM.sub("#", _MASK_HEX.sub("0x#", line.strip()))


def _looks_like_json(text: str) -> bool:
    head = text.lstrip()[:1]
    if head not in ("{", "["):
        return False
    try:
        json.loads(text)
    except (ValueError, RecursionError):
        return False
    return True


def _collapse_identical(
    lines: list[str], min_run: int, exact: bool = False, skip_blank: bool = False
) -> list[str]:
    """Runs of >= min_run identical lines -> first line + an explicit count.

    ``skip_blank`` exempts whitespace-only runs: the code strategy blanks
    pruned comment lines specifically to keep line numbering stable, and
    folding those runs into a marker would shift every line below again.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        key = line if exact else line.strip()
        j = i + 1
        while j < len(lines) and (lines[j] if exact else lines[j].strip()) == key:
            j += 1
        run = j - i
        if (
            run >= min_run
            and not (not exact and _is_protected(line))
            and not (skip_blank and not line.strip())
        ):
            out.append(line)
            out.append(f"[line repeated × {run}]")
        else:
            out.extend(lines[i:j])
        i = j
    return out


def _collapse_similar(lines: list[str], min_run: int) -> list[str]:
    """Runs of >= min_run same-signature lines -> first 2 + last 1 + count."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_protected(line) or not line.strip():
            out.append(line)
            i += 1
            continue
        signature = _mask(line)
        j = i + 1
        while (
            j < len(lines)
            and lines[j].strip()
            and not _is_protected(lines[j])
            and _mask(lines[j]) == signature
        ):
            j += 1
        run = j - i
        if run >= min_run:
            out.extend(lines[i : i + 2])
            out.append(f"[× {run - 3} similar lines]")
            out.append(lines[j - 1])
        else:
            out.extend(lines[i:j])
        i = j
    return out


def _collapse_blanks(lines: list[str]) -> list[str]:
    """More than two consecutive blank lines -> one."""
    out: list[str] = []
    blanks = 0
    for line in lines:
        if not line.strip():
            blanks += 1
            continue
        if blanks:
            out.extend([""] if blanks > 2 else [""] * blanks)
            blanks = 0
        out.append(line)
    if blanks:
        out.extend([""] if blanks > 2 else [""] * blanks)
    return out


def _elide_middle(lines: list[str], head: int, tail: int) -> list[str]:
    """Keep head+tail plus every protected line between, with elision counts."""
    if len(lines) <= head + tail:
        return lines
    out = lines[:head]
    elided = 0
    for line in lines[head : len(lines) - tail]:
        if _is_protected(line):
            if elided:
                out.append(f"[… {elided} lines elided]")
                elided = 0
            out.append(line)
        else:
            elided += 1
    if elided:
        out.append(f"[… {elided} lines elided]")
    out.extend(lines[len(lines) - tail :])
    return out


def compress_log(text: str) -> str:
    """Shell output: collapse repeats/near-dups, then head+tail if still huge."""
    if _looks_like_json(text):
        # Structured output stays verbatim in v1: a sampled array rendered as
        # text is too easy for the model to copy back as if it were valid JSON.
        return text
    lines = text.splitlines()
    lines = _collapse_identical(lines, REPEAT_MIN)
    lines = _collapse_similar(lines, SIMILAR_MIN)
    lines = _collapse_blanks(lines)
    if sum(len(line) + 1 for line in lines) > LOG_BUDGET:
        lines = _elide_middle(lines, LOG_HEAD_LINES, LOG_TAIL_LINES)
    return "\n".join(lines)


_READ_HEADER = re.compile(r"\AShowing lines \d+-\d+ of \d+ total lines\.\n\n---\n\n")


def compress_code(text: str, file_path: str = "") -> tuple[str, int]:
    """File reads: whole-line comment runs and flagrant identical-line runs.

    Returns ``(text, comment_lines_elided)`` — the count lets the caller fold
    a single note into its own end-of-result marker instead of this function
    inserting one per comment block. Models cite line numbers, so code lines
    are never altered AND numbering is preserved: comment lines are blanked
    in place (never deleted — a deleted block shifted every real-line
    reference below it: LSP diagnostics, offset/limit paging, the model's
    own counting), and the collapse skips blank runs for the same reason.
    Exact-repeat code runs still fold to an explicit in-place count.
    Detection (Magika when installed, extension fallback) picks the
    tree-sitter grammar; comment pruning runs FIRST because the parser must
    see genuine source — collapse markers would shift byte spans. Paging
    via offset/limit remains the intended way to shrink reads further.
    """
    header = ""
    match = _READ_HEADER.match(text)
    if match:
        header = match.group(0)
        text = text[match.end() :]
    language = detect_language(text, file_path)
    comment_count = 0
    if language:
        text, comment_count = prune_comment_lines(text, language)
    lines = _collapse_identical(
        text.splitlines(), CODE_REPEAT_MIN, exact=True, skip_blank=True
    )
    return header + "\n".join(lines), comment_count


def compress_grep(text: str) -> str:
    """grep_search output: cap matches per file, keeping headers and totals."""
    if not text.startswith("Found "):
        return text
    out: list[str] = []
    kept_in_file = 0
    skipped_in_file = 0
    for line in text.splitlines():
        if line.startswith("File: ") or line == "---":
            if skipped_in_file:
                out.append(f"  … (+{skipped_in_file} more matches in this file)")
            kept_in_file = 0
            skipped_in_file = 0
            out.append(line)
            continue
        if re.match(r"L\d+: ", line):
            kept_in_file += 1
            if kept_in_file > MAX_MATCHES_PER_FILE:
                skipped_in_file += 1
                continue
            if len(line) > GREP_LINE_MAX:
                line = line[: GREP_LINE_MAX - 1] + "…"
        out.append(line)
    if skipped_in_file:
        out.append(f"  … (+{skipped_in_file} more matches in this file)")
    return "\n".join(out)


def _is_nav_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and len(stripped) < NAV_LINE_MAX and stripped[-1] not in ".!?"


def _collapse_nav_runs(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not _is_nav_line(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and _is_nav_line(lines[j]):
            j += 1
        run = j - i
        if run >= NAV_RUN_MIN:
            out.extend(lines[i : i + 3])
            out.append(f"[… {run - 3} short nav/link lines]")
        else:
            out.extend(lines[i:j])
        i = j
    return out


def compress_web(text: str, query: str = "") -> str:
    """web_fetch text: strip boilerplate, then keep what the query needs.

    Lossless passes first (nav collapse, paragraph dedup). A body still large
    after them is shrunk by the query-aware rerank lane (tools.compress.rerank)
    when the call carried a query; without one — or when rerank declines —
    the head+tail elision is the fallback.
    """
    # Preserve the "Content from {url} ...:" header line(s) up to the blank.
    header = ""
    split = text.find("\n\n")
    if split != -1 and text.startswith("Content from "):
        header = text[: split + 2]
        text = text[split + 2 :]

    lines = _collapse_nav_runs(text.splitlines())

    # Exact paragraph dedup: repeated footers/banners appear once.
    paragraphs = "\n".join(lines).split("\n\n")
    seen: set[str] = set()
    kept: list[str] = []
    dropped = 0
    for paragraph in paragraphs:
        key = " ".join(paragraph.split())
        if key and key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(paragraph)
    if dropped:
        kept.append(f"[{dropped} repeated paragraphs deduplicated]")

    body = None
    if query and sum(len(p) + 2 for p in kept) > RERANK_MIN_BODY:
        reranked = rerank_prose(kept, query)
        if reranked is not None:
            body = "\n\n".join(reranked)
    if body is None:
        body = "\n\n".join(kept)
        if len(body) > WEB_BUDGET:
            elided = len(body) - WEB_HEAD_CHARS - WEB_TAIL_CHARS
            body = (
                body[:WEB_HEAD_CHARS]
                + f"\n[… {elided} chars elided]\n"
                + body[-WEB_TAIL_CHARS:]
            )
    return header + body
