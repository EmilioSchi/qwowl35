"""Shell-call rendering: prompt/command rows with the type-out reveal, output
rows, the bash syntax pass, and the advisory blocks appended for the model."""

from __future__ import annotations

import re

from rich.console import RenderableType
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Span, Text

import theme
from config import TOOL_PREVIEW_LINES
from widgets.chat.markdown import _code_theme
from widgets.chat.primitives import (
    _BlockLine,
    _CURSOR,
    _FullWidthLines,
    _line_with_bg,
    highlight_refs,
)
from widgets.chat.renderers.code import (
    _parse_file_view,
    _render_anchored_code,
    _render_syntax_status,
    _section_header_text,
    _section_rows,
)


# First lines of the advisory blocks the agent appends to a tool result for the
# model (post-write anchor reads, syntax-check notes). Used to peel them off the
# command's own output and render them as a labelled "what the model saw" preview.
_ADVISORY_MARKER = re.compile(r"^(You just wrote `|Syntax check \()")


def _command_rows(
    command: str,
    *,
    cursor: bool,
    reveal: int | None = None,
    first_prompt: Text | None = None,
    bg: str | None = None,
) -> list[_BlockLine]:
    """Shell command rows (``$``/``> `` prompts) on the terminal background.

    The full ``command`` is always highlighted (so the bash lexer sees complete,
    well-formed syntax and produces stable colors); ``reveal`` then trims how
    much of that highlight is shown. Slicing the already-highlighted ``Text``
    preserves color spans, so the type-out animation no longer re-lexes a
    different prefix each frame — which is what made long commands flicker.
    ``reveal`` counts characters over the full command *including* the newlines
    between lines, matching ``target[: reveal]`` semantics used by the caller.
    """
    if not command:
        command = "(waiting for command...)"
    # split("\n"), NOT splitlines(): reveal charges exactly 1 char per line
    # boundary, but splitlines also breaks on \r\n, \v,   etc., which
    # desynced the per-line reveal offsets from target[:reveal].
    lines = command.split("\n")
    highlighted_lines = _highlight_bash_lines(command)
    if reveal is None:
        reveal = len(command)
    texts: list[Text] = []
    remaining = reveal
    for index, line in enumerate(lines):
        if index > 0:
            remaining -= 1  # the '\n' separating this line from the previous
            if remaining <= 0:
                break
        take = min(len(line), max(0, remaining))
        remaining -= take
        # The prompt is prepended AFTER reveal slicing (it is chrome, not
        # command text), so `reveal` keeps counting only command characters.
        if index == 0 and first_prompt is not None:
            text = first_prompt.copy()
        else:
            text = Text("$ " if index == 0 else "> ", style=theme.ACCENT)
        if index < len(highlighted_lines):
            seg = highlighted_lines[index][:take]
            seg.no_wrap = True
            text.append_text(seg)
        else:
            text.append(line[:take], style=theme.FG_BRIGHT)
        texts.append(text)
    if cursor and texts:
        texts[-1].append(_CURSOR, style=theme.FG_BRIGHT)
    return [_line_with_bg(text, bg or theme.BG_BASE) for text in texts]


def _highlight_bash_lines(command: str) -> list[Text]:
    """Apply a hidden ```bash-style syntax pass to command text."""
    try:
        highlighted = Syntax(
            command,
            "bash",
            theme=_code_theme(),
            background_color=theme.BG_BASE,
            word_wrap=False,
        ).highlight(command)
        lines = highlighted.split("\n", allow_blank=True)
    except Exception:
        return [Text(line, style=theme.FG_BRIGHT) for line in command.split("\n")]

    styled: list[Text] = []
    for line in lines[: len(command.split("\n"))]:
        item = line.copy()
        item.no_wrap = True
        _strip_bold_and_background(item)
        if not item.plain:
            item.append("", style=theme.FG_BRIGHT)
        item.style = theme.FG_BRIGHT
        styled.append(item)
    return styled


def _strip_bold_and_background(text: Text) -> None:
    """Keep lexer foregrounds, but no bold and no per-token background."""
    spans: list[Span] = []
    for span in text.spans:
        style = span.style
        if not isinstance(style, Style):
            try:
                style = Style.parse(str(style))
            except Exception:
                style = Style()
        spans.append(
            Span(
                span.start,
                span.end,
                Style(
                    color=style.color,
                    bold=False,
                    italic=False,
                    underline=False,
                ),
            )
        )
    text.spans = spans


def _output_rows(output: str, *, bg: str | None = None) -> list[_BlockLine]:
    """Command-output rows: dimmer grey, no prompt, ref accents preserved."""
    rows: list[_BlockLine] = []
    for line in output.splitlines() or [output]:
        text = highlight_refs(line)
        text.style = theme.FG_DIM  # base color; ref accents layer on top
        rows.append(_line_with_bg(text, bg or theme.BG_BASE))
    return rows


def _terminal_box(
    command: str,
    *,
    cursor: bool = False,
    reveal: int | None = None,
    output: str | None = None,
    note: str | None = None,
) -> RenderableType:
    """Command and (optional) output as one continuous dark rectangle."""
    rows = _command_rows(command, cursor=cursor, reveal=reveal)
    if output is not None:
        rows.extend(_output_rows(output))
    if note:
        rows.append(_line_with_bg(Text(note, style="dim"), theme.BG_BASE))
    return _FullWidthLines(rows, wrap=True)


def _shell_text(
    command: str, *, cursor: bool = False, reveal: int | None = None
) -> RenderableType:
    return _terminal_box(command, cursor=cursor, reveal=reveal)


def _split_bash_advisories(body: str) -> tuple[str, str]:
    """Split a bash result into (command output, advisory text appended for the
    model).

    The advisory region starts at the first injected block — a post-write anchor
    read or a syntax-check note — recognised by its marker after a blank line, so
    the command's own output is never misclassified.
    """
    lines = body.split("\n")
    for i in range(1, len(lines)):
        if lines[i - 1] == "" and _ADVISORY_MARKER.match(lines[i]):
            j = i
            while j > 0 and lines[j - 1] == "":
                j -= 1
            return "\n".join(lines[:j]), "\n".join(lines[i:])
    return body, ""


def _advisory_segments(advisory: str) -> list[tuple[str, str]]:
    """Classify the advisory region into (kind, text) paragraphs: ``autoread``
    for post-write anchor reads, ``report`` for the plain (no-anchor) post-write
    validation report — its first line carries the ``Syntax check (…)`` headline
    inline, which the anchor intro never does — ``written`` for the clean
    ``Wrote `…` … Syntax check (…): OK`` confirmation, ``syntax`` for a bare
    syntax-check note, ``note`` otherwise."""
    segments: list[tuple[str, str]] = []
    for para in advisory.split("\n\n"):
        para = para.strip("\n")
        if not para.strip():
            continue
        first = para.split("\n", 1)[0]
        if first.startswith("You just wrote `"):
            kind = "report" if "Syntax check (" in first else "autoread"
        elif first.startswith("Syntax check ("):
            kind = "syntax"
        elif first.startswith("Wrote `") and "Syntax check (" in first:
            kind = "written"
        else:
            kind = "note"
        segments.append((kind, para))
    return segments


def _render_autoread_segment(text: str, expanded: bool) -> list[RenderableType]:
    """Render a post-write anchor block with the read file view (it arrived as
    plain ``line:hash|content`` text inside the bash result)."""
    lines = text.split("\n")
    intro = lines[0]
    match = re.match(r"You just wrote `(.+?)`", intro)
    path = match.group(1) if match else ""
    parts: list[RenderableType] = [Text(intro, style=theme.FG_MUTED)]
    p_path, header, anchored, trailing = _parse_file_view("\n".join(lines[1:]), path)
    if not anchored:
        rest = "\n".join(lines[1:]).strip()
        if rest:
            parts.append(highlight_refs(rest))
        return parts
    if header:
        parts.append(Text(header, style=f"bold {theme.FG_DIM}"))
    limited, hidden = anchored, 0
    if not expanded and len(anchored) > TOOL_PREVIEW_LINES:
        limited, hidden = anchored[:TOOL_PREVIEW_LINES], len(anchored) - TOOL_PREVIEW_LINES
    parts.append(_render_anchored_code(p_path or path, limited))
    if hidden:
        parts.append(Text(f"... {hidden} more lines (Ctrl+o to expand)", style="dim"))
    for note in trailing:
        if note.strip():
            parts.append(Text(note, style="dim"))
    return parts


# The intros naming the file a post-write block is about: they carry the path
# that picks the lexer for the block's echoed file rows.
_REPORT_INTRO = re.compile(r"^You just wrote `(.+?)`")
_WRITTEN_INTRO = re.compile(r"^Wrote `(.+?)`")


def _render_report_segment(text: str, expanded: bool) -> list[RenderableType]:
    """Render the plain (no-anchor) post-write validation report pretty.

    Header line: the ``You just wrote …`` intro stays muted like other intros;
    the inline ``Syntax check (…) — N issue(s)…`` remainder gets the shared
    header styling (neutral text, tone-accented counts only). The body goes
    through the shared structured section renderer
    (``_section_rows``): soft-red issue bullets, warning-toned warnings, dim
    ``… and N more``/dedup bullets, and the echoed ``  line N: <content>``
    rows as gutter-labelled, syntax-highlighted code rows. Collapsed, the body
    is capped at TOOL_PREVIEW_LINES with the usual expand hint. Also covers
    the bullet-less all-unchanged one-liner (header styling only)."""
    lines = text.split("\n")
    first = lines[0]
    intro = _REPORT_INTRO.match(first)
    path = intro.group(1) if intro else ""
    cut = first.find("Syntax check (")
    # The remainder gets the shared header styling — neutral FG_DIM text with
    # only the count fragments tone-accented — never a full-line red paint
    # (the instruction clauses are long; painting them ERROR_SOFT read as a
    # wall of red, confirmed ugly on a real 9-issue report).
    header = Text()
    if cut > 0:
        header.append(first[:cut], style=theme.FG_MUTED)
        header.append_text(_section_header_text(first[cut:]))
    else:
        header.append_text(_section_header_text(first))
    parts: list[RenderableType] = [header]

    body = lines[1:]
    hidden = 0
    if not expanded and len(body) > TOOL_PREVIEW_LINES:
        hidden = len(body) - TOOL_PREVIEW_LINES
        body = body[:TOOL_PREVIEW_LINES]
    parts.extend(_section_rows(body, path))
    if hidden:
        parts.append(Text(f"... {hidden} more lines (Ctrl+o to expand)", style="dim"))
    return parts


def _render_written_segment(text: str) -> list[RenderableType]:
    """Clean post-write confirmation: the muted ``Wrote `…` (N lines).`` intro,
    then the ``Syntax check (…): OK`` block (with any riding warnings) through
    the same status styling every other diagnostics block gets."""
    cut = text.find("Syntax check (")
    if cut < 0:
        return [Text(text, style="dim")]
    intro_match = _WRITTEN_INTRO.match(text)
    path = intro_match.group(1) if intro_match else ""
    parts: list[RenderableType] = []
    intro = text[:cut].rstrip()
    if intro:
        parts.append(Text(intro, style=theme.FG_MUTED))
    parts.append(_render_syntax_status(text[cut:], path))
    return parts


def _render_advisory(advisory: str, expanded: bool) -> list[RenderableType]:
    """Render the 'what the model also received' preview shown under a bash box."""
    parts: list[RenderableType] = [Text("Model also received", style=f"bold {theme.FG_DIM}")]
    last_path = ""
    for kind, text in _advisory_segments(advisory):
        if kind in ("autoread", "report"):
            # Remember which file this block is about: a bare `Syntax check (…)`
            # paragraph that follows (the auto-read's embedded section, split
            # off by the paragraph split) echoes rows from the same file.
            intro = _REPORT_INTRO.match(text)
            if intro:
                last_path = intro.group(1)
        if kind == "autoread":
            parts.extend(_render_autoread_segment(text, expanded))
        elif kind == "report":
            parts.extend(_render_report_segment(text, expanded))
        elif kind == "written":
            parts.extend(_render_written_segment(text))
        elif kind == "syntax":
            parts.append(_render_syntax_status(text, last_path))
        else:
            parts.append(Text(text, style="dim"))
    return parts
