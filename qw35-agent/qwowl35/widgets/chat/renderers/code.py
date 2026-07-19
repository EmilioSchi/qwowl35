"""Code-view renderers: anchored (hashline) file views, inspect_file windows,
streaming edit content, colored diffs, and syntax-check status blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Group, RenderableType
from rich.syntax import Syntax
from rich.text import Text

import theme
from agents.editor import parse_line_ranges
from config import TOOL_PREVIEW_LINES
from tools.diagnostics import is_section_start, split_trailing_section
from widgets.chat.markdown import _code_theme
from widgets.chat.primitives import (
    _BlockLine,
    _CURSOR,
    _FullWidthLines,
    _line_with_bg,
    highlight_refs,
)


_HASHLINE_ANCHOR_LINE = re.compile(r"^\s*(\d+:?[0-9a-f]{2})\|(.*)$", re.IGNORECASE)
_DIFF_HUNK = re.compile(r"^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@")


# run_inspect_file's paged-window header.
_INSPECT_WINDOW = re.compile(r"^Showing lines (\d+)-(\d+) of (\d+) total lines\.$")
# The compression layer's trailing marker (MARKER_PREFIX in tools/compress).
_COMPRESS_MARKER = "[compressed: "


def _split_inspect_result(body: str) -> tuple[int, int | None, list[str], str | None]:
    """Split an `inspect_file` success into (start_line, total_lines, content
    lines, compression marker). Whole-file reads have no window header; paged
    reads start `Showing lines a-b of N total lines.` + blank/---/blank."""
    lines = body.splitlines()
    marker = None
    if lines and lines[-1].startswith(_COMPRESS_MARKER):
        marker = lines.pop()
        while lines and not lines[-1]:
            lines.pop()
    window = _INSPECT_WINDOW.match(lines[0]) if lines else None
    if (
        window is not None
        and len(lines) >= 4
        and lines[1] == ""
        and lines[2] == "---"
        and lines[3] == ""
    ):
        return int(window.group(1)), int(window.group(3)), lines[4:], marker
    return 1, None, lines, marker


def _numbered_code_rows(path: str, code_lines: list[str], start: int) -> RenderableType:
    """Line-numbered, syntax-highlighted file content on the code background —
    the anchorless sibling of `_render_anchored_code` for inspect_file reads."""
    code = "\n".join(code_lines)
    try:
        lexer = Syntax.guess_lexer(path or "text", code)
        highlighted_lines = Syntax(
            code,
            lexer,
            theme=_code_theme(),
            background_color=theme.CODE_BG,
            word_wrap=False,
        ).highlight(code).split("\n", allow_blank=True)
    except Exception:
        highlighted_lines = [Text(line) for line in code_lines]
    width = len(str(start + len(code_lines) - 1))
    rows: list[_BlockLine] = []
    for index, line in enumerate(code_lines):
        row = Text(no_wrap=True)
        row.append(f"{start + index:>{width}}  ", style=f"{theme.FG_GHOST} on {theme.CODE_BG}")
        if index < len(highlighted_lines):
            code_line = highlighted_lines[index].copy()
            code_line.stylize(f"on {theme.CODE_BG}", 0, len(code_line.plain))
            row.append_text(code_line)
        else:
            row.append(line, style=f"on {theme.CODE_BG}")
        rows.append(_BlockLine(row, f"on {theme.CODE_BG}"))
    return _FullWidthLines(rows)


@dataclass
class _SpawnSnippet:
    """The pre-edit file slice a delegator `edit` hands the Editor sub-agent:
    captured from disk before the editor runs, frozen for the block's life.
    Rows carry the hashline ids (`12af|content`) — the dialect the Editor
    actually reads — computed from the same pre-edit content."""

    path: str
    total_lines: int
    spans: list[tuple[int, int]]  # merged 1-based inclusive, exactly as named
    span_rows: list[list[str]]  # id-annotated rows per span


def _capture_spawn_snippet(filename: str, line_ranges: str) -> _SpawnSnippet | None:
    """Read the slice `line_ranges` names from `filename`, id-annotated.

    Spans are the ranges the call actually named (no margin, no small-file
    widening — the card is a focus view, the Editor's own slice may be
    wider); ids come from the hashline engine so the rows match the Editor's
    read_file view byte-for-byte."""
    from tools.files.hashline.document import Document
    from tools.files.hashline.hash import format_line_ref

    try:
        text = Path(filename).read_text(encoding="utf-8", errors="replace")
        doc = Document.from_str(filename, text)
    except Exception:
        return None
    total = len(doc.lines)
    spans = parse_line_ranges(line_ranges, total, margin=0, whole_file_under=0)
    if not spans:
        return None
    rows = [
        f"{format_line_ref(index + 1, line.short_hash)}|{line.content}"
        for index, line in enumerate(doc.lines)
    ]
    return _SpawnSnippet(
        path=filename,
        total_lines=total,
        spans=spans,
        span_rows=[rows[start - 1:end] for start, end in spans],
    )


def _render_spawn_snippet(snippet: _SpawnSnippet, expanded: bool) -> list[RenderableType]:
    """The id-annotated slice inside the Spawn card: one anchored code block
    with `... (K lines not shown) ...` label rows between non-contiguous
    spans (the editor's own sliced-view wording), capped at TOOL_PREVIEW_LINES
    code lines (across all spans) when collapsed."""
    total_code_lines = sum(len(rows) for rows in snippet.span_rows)
    budget = total_code_lines if expanded else TOOL_PREVIEW_LINES
    rows: list[str] = []
    shown = 0
    previous_end = 0
    for (start, _end), span in zip(snippet.spans, snippet.span_rows):
        if shown >= budget:
            break
        if start > previous_end + 1 and previous_end:
            rows.append(f"... ({start - previous_end - 1} lines not shown) ...")
        taken = span[: budget - shown]
        rows.extend(taken)
        shown += len(taken)
        previous_end = start + len(taken) - 1
    parts: list[RenderableType] = [_render_anchored_code(snippet.path, rows)]
    if shown < total_code_lines:
        parts.append(
            Text(f"... {total_code_lines - shown} more lines (Ctrl+o to expand)", style="dim")
        )
    return parts


def _render_inspect_result(args: dict[str, Any], body: str, expanded: bool) -> RenderableType:
    """The inspect_file card: `path (lines a-b of N)` header + numbered,
    syntax-highlighted content, capped collapsed like every other card."""
    path = args.get("file_path") if isinstance(args.get("file_path"), str) else ""
    # A trailing `LSP diagnostics (…)` section is never file content: carve it
    # off before the window split so it can't render as numbered code rows
    # (and the compress marker, which the section trails, is the last line
    # again for `_split_inspect_result`'s marker pop).
    body, diagnostics = split_trailing_section(body)
    start, total, content, marker = _split_inspect_result(body)
    header = Text()
    header.append(path or "file", style=f"bold {theme.FG_BRIGHT}")
    if total is not None:
        header.append(
            f" (lines {start}-{start + len(content) - 1} of {total})", style=theme.FG_GHOST
        )
    else:
        header.append(f" ({len(content)} lines)", style=theme.FG_GHOST)
    shown = content
    if not expanded and len(content) > TOOL_PREVIEW_LINES:
        shown = content[:TOOL_PREVIEW_LINES]
    parts: list[RenderableType] = [header, _numbered_code_rows(path, shown, start)]
    if len(shown) < len(content):
        parts.append(
            Text(f"... {len(content) - len(shown)} more lines (Ctrl+o to expand)", style="dim")
        )
    if marker:
        parts.append(Text(marker, style="dim"))
    if diagnostics:
        _append_trailing(parts, diagnostics.splitlines(), expanded, path=path)
    return Group(*parts)


def _anchor_header(path: str) -> str:
    """The ids header the read_file tool emits on a file's first open."""
    label = path or "file"
    return f"{label} (ids: each line is '<line><hash>|<content>'):"


def _partition_anchored(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split a hashline body into anchored code rows and trailing notes."""
    anchored: list[str] = []
    trailing: list[str] = []
    for line in lines:
        if _HASHLINE_ANCHOR_LINE.match(line):
            anchored.append(line)
        elif anchored:
            if line.startswith("... ("):
                trailing.append(line)
            else:
                anchored.append(line)
        elif line.startswith("... ("):
            trailing.append(line)
        else:
            trailing.append(line)
    return anchored, trailing


def _parse_file_view(body: str, fallback_path: str = "") -> tuple[str, str, list[str], list[str]]:
    """Return (path, header, anchored lines, trailing lines)."""
    lines = body.splitlines()
    header_index: int | None = None
    path = fallback_path

    for i, line in enumerate(lines):
        if "(ids" in line or "(anchors" in line or "(hashline anchors" in line:
            header_index = i
            head = line[len("Current "):] if line.startswith("Current ") else line
            path = head.split(" (", 1)[0].strip() or path
            break
        if line.startswith("Current ") and ":" in line:
            header_index = i
            rest = line[len("Current "):]
            path = re.split(r"\s*\(|:", rest, maxsplit=1)[0].strip() or path
            break
        if line.startswith("Lines (paste") and ":" in line:
            header_index = i
            break

    if header_index is None:
        # No header: repeat reads of a file omit it (the tool only emits the
        # anchors explainer the first time). Detect the anchored shape directly
        # so the rendering stays identical across calls.
        anchored, trailing = _partition_anchored(lines)
        if anchored:
            return path, "", anchored, trailing
        return path, "", [], lines

    anchored, trailing = _partition_anchored(lines[header_index + 1:])
    return path, lines[header_index], anchored, trailing


def _render_anchored_code(path: str, anchored_lines: list[str]) -> RenderableType:
    parsed: list[tuple[str, str, bool]] = []
    for line in anchored_lines:
        hashline_match = _HASHLINE_ANCHOR_LINE.match(line)
        if hashline_match is not None:
            parsed.append((hashline_match.group(1).strip(), hashline_match.group(2), False))
        else:
            parsed.append(("", line, True))

    if not parsed:
        return highlight_refs("\n".join(anchored_lines))

    code = "\n".join(content for _, content, _ in parsed)
    try:
        lexer = Syntax.guess_lexer(path or "text", code)
        highlighted_lines = Syntax(
            code,
            lexer,
            theme=_code_theme(),
            background_color=theme.CODE_BG,
            word_wrap=False,
        ).highlight(code).split("\n", allow_blank=True)
    except Exception:
        highlighted_lines = [Text(content) for _, content, _ in parsed]

    width = max((len(anchor) for anchor, _, is_label in parsed if not is_label), default=1)
    rows: list[_BlockLine] = []
    for index, (anchor, content, is_label) in enumerate(parsed):
        row = Text(no_wrap=True)
        if is_label:
            row.append(" " * (width + 2), style=f"on {theme.CODE_BG}")
        else:
            row.append(f"{anchor:>{width}}  ", style=f"bold {theme.ACCENT} on {theme.CODE_BG}")
        if index < len(highlighted_lines):
            code_line = highlighted_lines[index].copy()
            code_line.stylize(f"on {theme.CODE_BG}", 0, len(code_line.plain))
            row.append_text(code_line)
        else:
            row.append(content, style=f"on {theme.CODE_BG}")
        rows.append(_BlockLine(row, f"on {theme.CODE_BG}"))
    return _FullWidthLines(rows)


def _streaming_code_text(
    path: str, code: str, *, cursor: bool = False, reveal: int | None = None
) -> RenderableType:
    """Live file-content block for a streaming edit/insert call.

    Same reveal contract as ``_command_rows``: the full ``code`` is highlighted
    once (stable colors), then ``reveal`` slices the already-styled Text —
    counting 1 char per '\\n' — so the type-out never re-lexes a shifting
    prefix. The block uses the code background to match the result view the
    call will settle into.
    """
    lines = code.split("\n")
    try:
        lexer = Syntax.guess_lexer(path or "text", code)
        highlighted_lines = Syntax(
            code,
            lexer,
            theme=_code_theme(),
            background_color=theme.CODE_BG,
            word_wrap=False,
        ).highlight(code).split("\n", allow_blank=True)
    except Exception:
        highlighted_lines = [Text(line) for line in lines]
    if reveal is None:
        reveal = len(code)
    texts: list[Text] = []
    remaining = reveal
    for index, line in enumerate(lines):
        if index > 0:
            remaining -= 1  # the '\n' separating this line from the previous
            if remaining <= 0:
                break
        take = min(len(line), max(0, remaining))
        remaining -= take
        row = highlighted_lines[index][:take] if index < len(highlighted_lines) else Text(line[:take])
        row.no_wrap = True
        texts.append(row)
    if cursor and texts:
        texts[-1].append(_CURSOR, style=theme.FG_BRIGHT)
    rows: list[_BlockLine] = []
    for row in texts:
        row.stylize(f"on {theme.CODE_BG}", 0, len(row.plain))
        rows.append(_BlockLine(row, f"on {theme.CODE_BG}"))
    return _FullWidthLines(rows)


def _render_code_block(
    path: str,
    code: str,
    *,
    max_lines: int | None = None,
) -> RenderableType:
    code_lines = code.splitlines()
    if not code_lines:
        code_lines = [""]
    hidden = 0
    if max_lines is not None and len(code_lines) > max_lines:
        hidden = len(code_lines) - max_lines
        code_lines = code_lines[:max_lines]

    visible_code = "\n".join(code_lines)
    try:
        lexer = Syntax.guess_lexer(path or "text", visible_code)
        highlighted_lines = Syntax(
            visible_code,
            lexer,
            theme=_code_theme(),
            background_color=theme.CODE_BG,
            word_wrap=False,
        ).highlight(visible_code).split("\n", allow_blank=True)
    except Exception:
        highlighted_lines = [Text(line) for line in code_lines]

    rows: list[_BlockLine] = []
    for index, line in enumerate(code_lines):
        row = highlighted_lines[index].copy() if index < len(highlighted_lines) else Text(line)
        row.no_wrap = True
        row.stylize(f"on {theme.CODE_BG}", 0, len(row.plain))
        rows.append(_BlockLine(row, f"on {theme.CODE_BG}"))
    if hidden:
        rows.append(_line_with_bg(Text(f"... {hidden} more lines", style="dim"), theme.CODE_BG))
    return _FullWidthLines(rows)


# A diagnostics section's echoed file rows, one per dialect: the hashline
# `  replace id: <line><hash>|<content>` anchor echo ("edit id:" accepted for
# pre-rename transcripts) and the plain-report `  line N: <content>` row
# (agent.build_post_write_report).
_ECHO_HASHLINE = re.compile(r"^  (edit|replace) id: (\d+:?[0-9a-f]{2})\|(.*)$", re.IGNORECASE)
_ECHO_PLAIN = re.compile(r"^  line (\d+): (.*)$")


def _echo_parts(line: str) -> tuple[str, str] | None:
    """Split an echoed file row into (gutter label, file content), or None."""
    match = _ECHO_HASHLINE.match(line)
    if match is not None:
        return f"{match.group(1)} id: {match.group(2)}", match.group(3)
    match = _ECHO_PLAIN.match(line)
    if match is not None:
        return f"line {match.group(1)}", match.group(2)
    return None


def _dim_section_bullet(line: str) -> bool:
    """Bullets that summarise rather than report: ``- … and N more`` overflow
    markers and the dedup ``- N unchanged … (not repeated)`` notes."""
    return line.startswith("- … and ") or (
        line.startswith("- ") and line.endswith("already reported above (not repeated)")
    )


def _echo_code_texts(path: str, contents: list[str]) -> list[Text]:
    """Syntax-highlight a section's echoed file rows in one pass (the lexer
    sees them together, so colors stay stable), lexer picked from ``path``."""
    if not contents:
        return []
    code = "\n".join(contents)
    try:
        lexer = Syntax.guess_lexer(path or "text", code)
        return Syntax(
            code,
            lexer,
            theme=_code_theme(),
            background_color=theme.CODE_BG,
            word_wrap=False,
        ).highlight(code).split("\n", allow_blank=True)
    except Exception:
        return [Text(content) for content in contents]


# Tone accents inside otherwise-neutral section text: the count fragments a
# header carries ("9 issue(s)", "2 error(s)", "1 warning(s)"), a bullet's
# leading position ("- line 14, col 1:"), and its trailing "(source)".
_COUNT_FRAGMENT = re.compile(r"\d+ (?:issue|error|warning)\(s\)")
_BULLET_POSITION = re.compile(r"^- line \d+(?:, col \d+)?:")
_BULLET_SOURCE = re.compile(r"^(.*?)(\s\([^()]+\))$")


def _section_header_text(line: str) -> Text:
    """A section header with neutral body text and tone-accented counts only.

    The long instruction clauses ("Fix ONLY these lines …", "fix each line
    below with edit …") stay FG_DIM; just the ``N issue(s)``/``N error(s)``
    fragments wear the soft error tone and ``N warning(s)`` the warning tone.
    An ``: OK`` header keeps the full success green as before."""
    if ": OK" in line:
        return Text(line, style=theme.SUCCESS)
    text = Text(line, style=theme.FG_DIM)
    for match in _COUNT_FRAGMENT.finditer(line):
        tone = theme.WARNING if "warning" in match.group(0) else theme.ERROR_SOFT
        text.stylize(tone, match.start(), match.end())
    return text


def _section_bullet_text(line: str, tone: str) -> Text:
    """A diagnostic bullet with color as small accents, not a full-line paint:
    the leading ``- line N, col M:`` position in ``tone``, a ``[code-tag]`` and
    the trailing ``(source)`` faint, the human message bright. Any parse miss
    renders the whole bullet dim-neutral — never a red wall."""
    position = _BULLET_POSITION.match(line)
    if position is None:
        return Text(line, style=theme.FG_DIM)
    text = Text()
    text.append(line[: position.end()], style=tone)
    rest = line[position.end():]
    stripped = rest.lstrip(" ")
    if len(rest) > len(stripped):
        text.append(rest[: len(rest) - len(stripped)], style=theme.FG_BRIGHT)
        rest = stripped
    if rest.startswith("[") and "]" in rest:
        tag_end = rest.index("]") + 1
        text.append(rest[:tag_end], style=theme.FG_FAINT)
        rest = rest[tag_end:]
    source = _BULLET_SOURCE.match(rest)
    if source is not None:
        text.append(source.group(1), style=theme.FG_BRIGHT)
        text.append(source.group(2), style=theme.FG_FAINT)
    else:
        text.append(rest, style=theme.FG_BRIGHT)
    return text


def _section_rows(lines: list[str], path: str) -> list[RenderableType]:
    """Per-line renderables for a diagnostics section.

    Neutral body tones with small accents (see ``_section_header_text`` /
    ``_section_bullet_text``): bullets carry the soft error tone only on their
    position fragment — the warning tone inside a ``Warnings (not blocking)``
    run — ``… and N more`` and dedup bullets go dim, and echoed file rows
    become anchor-style code rows (accent gutter label + content highlighted
    by ``path``'s lexer on the code background). Any unrecognized line falls
    back to FG_DIM — this must never crash or paint a red wall."""
    echoes = [e for e in (_echo_parts(line) for line in lines) if e]
    width = max((len(label) for label, _ in echoes), default=0)
    code_texts = _echo_code_texts(path, [content for _, content in echoes])
    rows: list[RenderableType] = []
    in_warnings = False
    echo_index = 0
    for line in lines:
        echo = _echo_parts(line)
        if echo is not None:
            label, content = echo
            row = Text(no_wrap=True)
            row.append(
                f"  {label:>{width}}  ",
                style=f"bold {theme.ACCENT} on {theme.CODE_BG}",
            )
            if echo_index < len(code_texts):
                code_line = code_texts[echo_index].copy()
                code_line.stylize(f"on {theme.CODE_BG}", 0, len(code_line.plain))
                row.append_text(code_line)
            else:
                row.append(content, style=f"on {theme.CODE_BG}")
            rows.append(_FullWidthLines([_BlockLine(row, f"on {theme.CODE_BG}")]))
            echo_index += 1
            continue
        if is_section_start(line):
            in_warnings = False
            rows.append(_section_header_text(line))
            continue
        if line.startswith("Warnings (not blocking)"):
            in_warnings = True
            text = Text(line, style=theme.FG_DIM)
            count = re.search(r"\d+", line)
            if count is not None:
                text.stylize(theme.WARNING, count.start(), count.end())
            rows.append(text)
            continue
        if _dim_section_bullet(line):
            rows.append(Text(line, style="dim"))
            continue
        if line.startswith("- "):
            rows.append(
                _section_bullet_text(
                    line, theme.WARNING if in_warnings else theme.ERROR_SOFT
                )
            )
            continue
        rows.append(Text(line, style=theme.FG_DIM))
    return rows


def _render_syntax_status(text: str, path: str = "") -> RenderableType:
    """A diagnostics section as a structured status block: neutral text with
    tone accents on counts and bullet positions (see :func:`_section_rows`),
    the ``: OK`` header in full success green, echoed ``edit id:``/``line N:``
    rows as anchor-style code rows highlighted by ``path``'s lexer."""
    return Group(*_section_rows(text.split("\n"), path))


def _split_trailing_syntax(trailing: list[str]) -> tuple[list[str], list[str]]:
    """Split file-view trailing lines into (other notes, diagnostics block) —
    any section per the tools/diagnostics grammar (`Syntax check (…)` or
    `LSP diagnostics (…)`) gets the always-visible coloured status treatment."""
    for i, line in enumerate(trailing):
        if is_section_start(line):
            return trailing[:i], trailing[i:]
    return trailing, []


def _append_trailing(
    parts: list[RenderableType], trailing: list[str], expanded: bool, path: str = ""
) -> None:
    """Append file-view trailing notes: a syntax-check block always shows (as a
    structured coloured status; ``path`` picks the lexer for echoed file rows),
    other notes stay dim and only when expanded."""
    other, syntax = _split_trailing_syntax(trailing)
    if expanded:
        text = "\n".join(other).strip("\n")
        if text:
            parts.append(Text(text, style="dim"))
    if syntax:
        parts.append(_render_syntax_status("\n".join(syntax).strip("\n"), path))


def _body_has_diff(body: str) -> bool:
    """Whether a tool result carries a ``Diff:`` section worth the colored
    diff view — true for an attention-flagged (but still successful) edit,
    not for a hard failure with no diff to show."""
    return any(line.strip() == "Diff:" for line in body.splitlines())


def _split_diff_section(body: str) -> tuple[list[str], str, str]:
    """Return lines before Diff:, diff text, and text after the diff.

    The scan also terminates on a diagnostics-section header (tools/diagnostics
    grammar): callers carve the trailing section off before diff parsing, but
    should one ever not, a `Syntax check (…)`/`LSP diagnostics (…)` block must
    still never be swallowed into the diff text and drawn as removal rows.
    """
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "Diff:":
            continue
        after_index = len(lines)
        for j in range(index + 1, len(lines)):
            if (
                lines[j].startswith("Current ")
                or lines[j].startswith("Lines (paste")
                or lines[j].startswith("Created ")
                or lines[j].startswith("Overwrote ")
                or is_section_start(lines[j])
            ) and j > index + 1:
                after_index = j
                break
        return lines[:index], "\n".join(lines[index + 1:after_index]), "\n".join(lines[after_index:])
    return [], "", body


def _render_diff(diff: str) -> RenderableType:
    rows: list[_BlockLine] = []
    old_line = 0
    new_line = 0
    width = 4

    for raw in diff.splitlines():
        hunk = _DIFF_HUNK.match(raw)
        if hunk:
            old_line = int(hunk.group(1))
            new_line = int(hunk.group(2))
            width = max(width, len(str(old_line)), len(str(new_line)))

    old_line = 0
    new_line = 0
    for raw in diff.splitlines() or ["(no diff)"]:
        hunk = _DIFF_HUNK.match(raw)
        if hunk:
            old_line = int(hunk.group(1))
            new_line = int(hunk.group(2))
            rows.append(_line_with_bg(Text(raw, style=f"bold {theme.ACCENT}"), theme.DIFF_CONTEXT_BG))
            continue
        if raw.startswith("---") or raw.startswith("+++"):
            rows.append(_line_with_bg(Text(raw, style=f"bold {theme.FG_DIM}"), theme.DIFF_CONTEXT_BG))
            continue
        if raw.startswith("-"):
            line_no = f"{old_line:>{width}}" if old_line else " " * width
            row = Text(no_wrap=True)
            row.append(f"{line_no}  ", style=f"{theme.FG_GHOST} on {theme.DIFF_REMOVE_BG}")
            # Marker in its own column + a separator space, so content lines up
            # with context rows (which reserve "  " for the same two columns).
            row.append("- " + raw[1:], style=f"{theme.ERROR} on {theme.DIFF_REMOVE_BG}")
            rows.append(_BlockLine(row, f"on {theme.DIFF_REMOVE_BG}"))
            old_line += 1
            continue
        if raw.startswith("+"):
            line_no = f"{new_line:>{width}}" if new_line else " " * width
            row = Text(no_wrap=True)
            row.append(f"{line_no}  ", style=f"{theme.FG_GHOST} on {theme.DIFF_ADD_BG}")
            row.append("+ " + raw[1:], style=f"{theme.SUCCESS} on {theme.DIFF_ADD_BG}")
            rows.append(_BlockLine(row, f"on {theme.DIFF_ADD_BG}"))
            new_line += 1
            continue
        if raw.startswith("\\"):
            rows.append(_line_with_bg(Text(raw, style="dim"), theme.DIFF_CONTEXT_BG))
            continue
        line_no = f"{new_line:>{width}}" if new_line else " " * width
        content = raw[1:] if raw.startswith(" ") else raw
        row = Text(no_wrap=True)
        row.append(f"{line_no}  ", style=f"{theme.FG_GHOST} on {theme.DIFF_CONTEXT_BG}")
        row.append("  " + content, style=f"dim {theme.FG_DIM} on {theme.DIFF_CONTEXT_BG}")
        rows.append(_BlockLine(row, f"on {theme.DIFF_CONTEXT_BG}"))
        if old_line:
            old_line += 1
        if new_line:
            new_line += 1

    return _FullWidthLines(rows)
