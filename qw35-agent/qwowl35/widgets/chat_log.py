"""The scrollable chat transcript, styled after little-coder.

Each message is its own widget in a vertical scroller. Assistant and user text is
rendered as **Markdown** (headings, lists, inline code, syntax-highlighted code
blocks). Tool calls/results carry plain **text labels** (no emoji/unicode glyphs,
which many terminals won't render). Streaming assistant/reasoning text is repainted
on a fast timer so generation visibly evolves without re-parsing on every token.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.markdown import Markdown
from rich.segment import Segment
from rich.syntax import Syntax
from rich.style import Style
from rich.text import Span, Text
from textual.containers import VerticalScroll
from textual.widgets import Static

import theme
from client import parse_xml_tool_args, recover_json_string_field_object, recover_xml_parameter
from config import TOOL_PREVIEW_LINES

# https://github.com/owner/repo  (optional trailing path)
_GITHUB_URL = re.compile(r"https?://github\.com/[\w.\-]+/[\w.\-]+(?:/[\w./\-]*)?")
_OWNER_REPO = re.compile(r"(?<![\w./@])([A-Za-z0-9][\w.\-]+/[\w.\-]{2,})(?![\w./])")
_KNOWN_LIBS = re.compile(
    r"\b(textual|rich|httpx|tree[_-]sitter|asyncio|pytest|numpy|pandas|requests)\b",
    re.IGNORECASE,
)
def _ref_style() -> str:
    return f"bold {theme.ACCENT} underline"


def _lib_style() -> str:
    return theme.ACCENT


def _code_theme() -> str:
    """Pygments/Rich syntax theme matching the active light/dark mode.

    ``monokai`` reads well on dark backgrounds, but its bright foregrounds wash
    out on a light theme, so use a light-oriented style there instead.
    """
    return "monokai" if theme.is_dark() else "default"


def _markdown(text: str) -> Markdown:
    """Markdown renderable using the mode-appropriate code-block theme."""
    return Markdown(text, code_theme=_code_theme())
_HASHLINE_ANCHOR_LINE = re.compile(r"^\s*(\d+:?[0-9a-f]{2})\|(.*)$", re.IGNORECASE)
_DIFF_HUNK = re.compile(r"^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@")
# First lines of the advisory blocks the agent appends to a tool result for the
# model (post-write anchor reads, syntax-check notes). Used to peel them off the
# command's own output and render them as a labelled "what the model saw" preview.
_ADVISORY_MARKER = re.compile(r"^(You just wrote `|Syntax check \()")
_FILE_CHANGE_TOOLS = {
    "edit",
    "insert",
    "delete",
}
_FILE_READ_TOOLS = {"beginTransaction"}
_FILE_VIEW_TOOLS = _FILE_READ_TOOLS | _FILE_CHANGE_TOOLS
_CURSOR = "▋"  # trailing block cursor while a command types out
# Shell / code / diff colors are read from ``theme.*`` at render time (below) so
# they follow a live theme change; there are no baked color aliases here.

# Repaint streaming text ~20x/s so it visibly evolves while still batching the
# (relatively costly) markdown re-parse.
_REFRESH_INTERVAL = 0.05

# Client-side typewriter reveal for tool-call args. The server emits the whole
# XML tool call in one delta, so growth can't ride on fragment arrival anymore;
# we instead reveal a few more characters every tick. Long commands get an
# adaptive bump so they don't crawl, but the bump is capped so a very long
# multiline write (e.g. a big `cat <<EOF` heredoc) still types out line by line
# instead of dumping a dozen lines in the first frame.
_REVEAL_MIN_STEP = 2
_REVEAL_MAX_STEP = 24
# Cursor blink: toggle every N ticks (N * _REFRESH_INTERVAL seconds).
_BLINK_TICKS = 10


@dataclass
class _BlockLine:
    text: Text
    pad_style: str


class _FullWidthLines:
    """Render text rows with background extending to the widget width.

    By default each logical line is clipped to the widget width (terminal-style,
    used for code/diff views). With ``wrap=True`` long lines fold onto extra rows
    instead of being clipped — used for bash commands/output and tool-call
    previews so nothing is hidden on a narrow terminal.
    """

    def __init__(self, lines: list[_BlockLine], *, wrap: bool = False) -> None:
        self._lines = lines
        self._wrap = wrap

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        if not self._wrap:
            width = max(1, options.max_width)
            for line in self._lines:
                text = line.text.copy()
                text.no_wrap = True
                missing = width - text.cell_len
                if missing > 0:
                    text.append(" " * missing, style=line.pad_style)
                yield text
            return

        # Render each logical line at its NATURAL height: clear options.height so
        # render_lines doesn't pad every single line out to the widget's full
        # height. (When height is set — e.g. inside an auto-height container like
        # the approval modal — each line would otherwise be padded to the whole
        # height, so only the first line shows and the rest collapse to blanks.)
        line_options = options.update(height=None)
        for line in self._lines:
            text = line.text.copy()
            text.no_wrap = False
            # render_lines folds the line and pads every visual row to the full
            # width with the row's background, so wrapped continuations stay dark.
            rows = console.render_lines(
                text, line_options, pad=True, style=Style.parse(line.pad_style)
            )
            # Terminate EVERY visual row with a newline, including the very last.
            # Textual measures a widget's height by counting '\n' in the rendered
            # segments (RichVisual.get_height), then crops the strips to that
            # height. A box whose final row had no trailing newline was measured
            # one row short, so its last line — the bash output, a tool-call arg
            # preview, or the final line of the approval command — got cropped
            # away. It also let the next renderable in a Group ride onto that
            # unterminated row (the "swallow" the advisory block worked around).
            for segments in rows:
                yield from segments
                yield Segment.line()


def _line_with_bg(text: Text, bg: str) -> _BlockLine:
    styled = text.copy()
    styled.stylize(f"on {bg}", 0, len(styled.plain))
    return _BlockLine(styled, f"on {bg}")


def highlight_refs(text: str) -> Text:
    rich = Text(text)
    for match in _GITHUB_URL.finditer(text):
        rich.stylize(f"{_ref_style()} link {match.group(0)}", match.start(), match.end())
    for match in _OWNER_REPO.finditer(text):
        if "://" in text[max(0, match.start() - 8): match.start()]:
            continue
        rich.stylize(_ref_style(), match.start(1), match.end(1))
    for match in _KNOWN_LIBS.finditer(text):
        rich.stylize(_lib_style(), match.start(), match.end())
    return rich


def _parse_args(buffer: str) -> dict[str, Any]:
    """Parse streamed tool-call JSON, tolerating partial wrappers."""
    if not buffer.strip():
        return {}
    try:
        parsed = json.loads(buffer)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = buffer.find("{")
        end = buffer.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(buffer[start:end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                pass
    parsed_xml = parse_xml_tool_args(buffer)
    if parsed_xml is not None:
        return parsed_xml
    return {}


def _compact_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    parts: list[str] = []
    for key in ("file", "pattern", "symbol", "id", "position", "mode", "from", "to"):
        value = args.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value!r}")
    if not parts and "command" in args:
        parts.append(f"command={args['command']!r}")
    return " ".join(parts)


def _command_from_args(buffer: str) -> str:
    args = _parse_args(buffer)
    command = args.get("command")
    if isinstance(command, str):
        return command
    recovered = recover_json_string_field_object(buffer, "command")
    if recovered is not None:
        return recovered
    recovered_xml = recover_xml_parameter(buffer, "command", partial=True)
    if recovered_xml is not None:
        return recovered_xml
    match = re.search(r'"command"\s*:\s*"((?:\\.|[^"\\])*)', buffer, re.DOTALL)
    if not match:
        return buffer.strip()
    try:
        return json.loads(f'"{match.group(1)}"')
    except Exception:
        return match.group(1)


_TOOL_BADGES = {
    "bash": ">_",
    "beginTransaction": "<>",
    "insert": "++",
    "delete": "--",
    "edit": "~~",
    "file": ">>",
}


def _tool_badge(name: str) -> str:
    """A compact 2-char status badge standing in for the tool name."""
    return _TOOL_BADGES.get(name, "::")


def _tool_title(name: str, args: dict[str, Any] | None, color: str) -> Text:
    """Tool header: a black-backed badge in `color` (no bold) + grey path."""
    title = Text()
    title.append(_tool_badge(name), style=f"{color} on {theme.BG_BASE}")
    path = (args or {}).get("file") or (args or {}).get("path")
    if isinstance(path, str) and path:
        title.append("  ")
        title.append(path, style=theme.FG_MUTED)
    return title


def _command_rows(
    command: str, *, cursor: bool, reveal: int | None = None
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
    lines = command.splitlines() or [command]
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
        prompt = "$ " if index == 0 else "> "
        text = Text(prompt, style=theme.ACCENT)
        if index < len(highlighted_lines):
            seg = highlighted_lines[index][:take]
            seg.no_wrap = True
            text.append_text(seg)
        else:
            text.append(line[:take], style=theme.FG_BRIGHT)
        texts.append(text)
    if cursor and texts:
        texts[-1].append(_CURSOR, style=theme.FG_BRIGHT)
    return [_line_with_bg(text, theme.BG_BASE) for text in texts]


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
        return [Text(line, style=theme.FG_BRIGHT) for line in command.splitlines() or [command]]

    styled: list[Text] = []
    for line in lines[: len(command.splitlines() or [command])]:
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


def _output_rows(output: str) -> list[_BlockLine]:
    """Command-output rows: dimmer grey, no prompt, ref accents preserved."""
    rows: list[_BlockLine] = []
    for line in output.splitlines() or [output]:
        text = highlight_refs(line)
        text.style = theme.FG_DIM  # base color; ref accents layer on top
        rows.append(_line_with_bg(text, theme.BG_BASE))
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


def _preview_lines(body: str, expanded: bool) -> tuple[str, int]:
    lines = body.splitlines()
    if not expanded and len(lines) > TOOL_PREVIEW_LINES:
        return "\n".join(lines[:TOOL_PREVIEW_LINES]), len(lines) - TOOL_PREVIEW_LINES
    if len(body) > 20000:
        return body[:20000] + "\n... (truncated)", 0
    return body, 0


def _path_from_args(args: dict[str, Any]) -> str:
    path = args.get("file") or args.get("path")
    return path if isinstance(path, str) else ""


def _anchor_header(path: str) -> str:
    """The ids header the beginTransaction tool emits on a file's first open."""
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
        highlighted_lines = [Text(content) for _, content in parsed]

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
    """Classify the advisory region into (kind, text) paragraphs: ``autoread`` for
    post-write anchors, ``syntax`` for a syntax-check note, ``note`` otherwise."""
    segments: list[tuple[str, str]] = []
    for para in advisory.split("\n\n"):
        para = para.strip("\n")
        if not para.strip():
            continue
        first = para.split("\n", 1)[0]
        if first.startswith("You just wrote `"):
            kind = "autoread"
        elif first.startswith("Syntax check ("):
            kind = "syntax"
        else:
            kind = "note"
        segments.append((kind, para))
    return segments


def _render_syntax_status(text: str) -> RenderableType:
    """A syntax-check note as a coloured status block: green OK / soft-red issues."""
    color = theme.SUCCESS if ": OK" in text.split("\n", 1)[0] else theme.ERROR_SOFT
    return Text(text, style=color)


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
        parts.append(Text(f"... {hidden} more lines (Ctrl+O to expand)", style="dim"))
    for note in trailing:
        if note.strip():
            parts.append(Text(note, style="dim"))
    return parts


def _render_advisory(advisory: str, expanded: bool) -> list[RenderableType]:
    """Render the 'what the model also received' preview shown under a bash box."""
    parts: list[RenderableType] = [Text("Model also received", style=f"bold {theme.FG_DIM}")]
    for kind, text in _advisory_segments(advisory):
        if kind == "autoread":
            parts.extend(_render_autoread_segment(text, expanded))
        elif kind == "syntax":
            parts.append(_render_syntax_status(text))
        else:
            parts.append(Text(text, style="dim"))
    return parts


def _split_trailing_syntax(trailing: list[str]) -> tuple[list[str], list[str]]:
    """Split file-view trailing lines into (other notes, syntax-check block)."""
    for i, line in enumerate(trailing):
        if line.startswith("Syntax check ("):
            return trailing[:i], trailing[i:]
    return trailing, []


def _append_trailing(parts: list[RenderableType], trailing: list[str], expanded: bool) -> None:
    """Append file-view trailing notes: a syntax-check block always shows (as a
    coloured status), other notes stay dim and only when expanded."""
    other, syntax = _split_trailing_syntax(trailing)
    if expanded:
        text = "\n".join(other).strip("\n")
        if text:
            parts.append(Text(text, style="dim"))
    if syntax:
        parts.append(_render_syntax_status("\n".join(syntax).strip("\n")))


def _split_diff_section(body: str) -> tuple[list[str], str, str]:
    """Return lines before Diff:, diff text, and text after the diff."""
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


class ToolBlock(Static):
    """One tool call+result: grows as args stream, collapsible result."""

    def __init__(self, name: str) -> None:
        super().__init__(classes="msg tool-pending")
        self.tool_name = name
        self.args_buf = ""
        self.full_result: str | None = None
        self.is_error = False
        self.expanded = False
        self.args_dirty = False
        self.reveal = 0  # chars of the command/detail currently typed out
        self.result_ready = False  # result arrived but reveal still typing out


class ChatView(VerticalScroll):
    """A vertical scroller holding one widget per message."""

    DEFAULT_CSS = """
    ChatView {
        height: 1fr;
        padding: 0 0 0 1;
        scrollbar-size-vertical: 1;
        scrollbar-background: $bg-base;
        scrollbar-background-hover: $bg-base;
        scrollbar-background-active: $bg-base;
        scrollbar-color: $scroll-bar;
        scrollbar-color-hover: $scroll-bar-hover;
        scrollbar-color-active: $scroll-bar-active;
        scrollbar-corner-color: $bg-base;
    }
    ChatView .msg { margin: 1 0 0 0; width: 1fr; }
    ChatView .user {
        background: $bg-surface;
        color: $fg-bright;
        padding: 0 1;
    }
    ChatView .assistant { color: $fg-bright; }
    ChatView .thinking { color: $fg-faint; text-style: italic; padding: 0 0 0 1; }
    ChatView .tool-pending { color: $accent; padding: 0 1; }
    ChatView .tool-success { color: $success-soft; padding: 0 1; }
    ChatView .tool-error   { color: $error-soft; padding: 0 1; }
    ChatView .system { color: $fg-faint; text-style: italic; }
    ChatView .warning { color: $warning; text-style: bold; }
    ChatView .error { color: $error; text-style: bold; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._assistant: Static | None = None
        self._assistant_buf = ""
        self._assistant_dirty = False
        self._reasoning: Static | None = None
        self._reasoning_buf = ""
        self._reasoning_dirty = False
        self._tool_blocks: dict[int, ToolBlock] = {}  # in-flight calls by index
        self._collapsibles: list[ToolBlock] = []      # finished results (Ctrl+O)
        self.tools_expanded = False
        self._frame = 0          # tick counter, drives the reveal cursor blink
        self._blink_on = True    # current cursor-visible phase

    def on_mount(self) -> None:
        self.can_focus = True
        # One persistent timer drives smooth repainting of streaming text.
        self.set_interval(_REFRESH_INTERVAL, self._tick)

    # -- streaming repaint -------------------------------------------------- #
    def _tick(self) -> None:
        self._frame += 1
        blink_on = (self._frame // _BLINK_TICKS) % 2 == 0
        blink_changed = blink_on != self._blink_on
        self._blink_on = blink_on
        if self._assistant_dirty and self._assistant is not None:
            self._assistant.update(_markdown(self._assistant_buf))
            self._assistant_dirty = False
            self._bump()
        if self._reasoning_dirty and self._reasoning is not None:
            self._reasoning.update(Text(self._reasoning_buf))
            self._reasoning_dirty = False
            self._bump()
        for index, block in list(self._tool_blocks.items()):
            target = self._call_target(block)
            revealing = block.reveal < len(target)
            if revealing:
                # Adaptive: long commands type out faster so they don't crawl,
                # but capped so a huge multiline write still reveals ~a line per
                # frame rather than dumping many lines at once.
                step = min(
                    _REVEAL_MAX_STEP,
                    max(_REVEAL_MIN_STEP, (len(target) - block.reveal) // 8),
                )
                block.reveal = min(len(target), block.reveal + step)
                # Reveal just finished and a result was waiting → show it now.
                if block.reveal >= len(target) and block.result_ready:
                    self._promote_result(index)
                    continue
            # Repaint when content advanced or new args arrived. While typing,
            # the per-tick advance already repaints, so the blink rides along.
            if block.args_dirty or revealing:
                block.update(self._render_tool_call(block))
                block.args_dirty = False
                self._bump()

    # -- low-level append --------------------------------------------------- #
    def _at_bottom(self) -> bool:
        return self.scroll_offset.y >= self.max_scroll_y - 1

    def _append(self, widget: Static) -> Static:
        stick = self._at_bottom()
        self.mount(widget)
        if stick:
            self.scroll_end(animate=False)
        return widget

    def _bump(self) -> None:
        if self._at_bottom():
            self.scroll_end(animate=False)

    # -- user --------------------------------------------------------------- #
    def add_user(self, text: str) -> None:
        self._append(Static(_markdown(text), classes="msg user"))

    def add_system(self, text: str) -> None:
        self._append(Static(Text(text), classes="msg system"))

    def clear(self) -> None:
        """Wipe the transcript: remove every message widget and streaming state.

        Used by the ``/clear`` command. The persistent repaint timer set up in
        ``on_mount`` keeps running; only content and in-flight buffers reset.
        """
        self.remove_children()
        self._assistant = None
        self._assistant_buf = ""
        self._assistant_dirty = False
        self._reasoning = None
        self._reasoning_buf = ""
        self._reasoning_dirty = False
        self._tool_blocks.clear()
        self._collapsibles.clear()
        self.tools_expanded = False

    # -- assistant (streaming markdown) ------------------------------------- #
    def add_assistant_chunk(self, text: str) -> None:
        self._flush_held_tools()
        if self._assistant is None:
            self._assistant_buf = text
            self._assistant = self._append(
                Static(_markdown(self._assistant_buf), classes="msg assistant")
            )
        else:
            self._assistant_buf += text
        self._assistant_dirty = True

    def flush_assistant(self) -> None:
        if self._assistant is not None:
            self._assistant.update(_markdown(self._assistant_buf))
            self._bump()
        self._assistant = None
        self._assistant_buf = ""
        self._assistant_dirty = False

    # -- reasoning (dim italic) --------------------------------------------- #
    def add_reasoning_chunk(self, text: str) -> None:
        self._flush_held_tools()
        if self._reasoning is None:
            self._reasoning_buf = text
            self._reasoning = self._append(
                Static(Text(self._reasoning_buf), classes="msg thinking")
            )
        else:
            self._reasoning_buf += text
        self._reasoning_dirty = True

    def flush_reasoning(self) -> None:
        if self._reasoning is not None:
            self._reasoning.update(Text(self._reasoning_buf))
            self._bump()
        self._reasoning = None
        self._reasoning_buf = ""
        self._reasoning_dirty = False

    # -- tools (streaming call box + collapsible result) -------------------- #
    def _flush_held_tools(self) -> None:
        """Promote any finished-but-still-typing tool box immediately.

        A held block (result arrived, reveal not yet complete) is waiting on the
        timer to finish its type-out. When new turn output arrives — another tool
        call, reasoning, or assistant text — we must let it go now: the call
        index resets every turn, so a lingering block at index 0 would be
        orphaned (its output never shown) when the next turn reuses that index,
        and its output must not paint after newer content. The reveal already had
        the tool-execution + prefill gap to play.
        """
        for index in [i for i, b in self._tool_blocks.items() if b.result_ready]:
            self._promote_result(index)

    def begin_tool_call(self, index: int, name: str) -> ToolBlock:
        self._flush_held_tools()
        block = ToolBlock(name)
        self._tool_blocks[index] = block
        self._append(block)
        block.update(self._render_tool_call(block))
        return block

    def update_tool_call(self, index: int, fragment: str) -> None:
        block = self._tool_blocks.get(index)
        if block is None:  # defensive: args before begin
            block = self.begin_tool_call(index, "?")
        block.args_buf += fragment
        block.args_dirty = True

    def add_tool_result(self, index: int, name: str, text: str, is_error: bool = False) -> None:
        block = self._tool_blocks.get(index)
        if block is None:  # defensive: result with no streamed call
            block = ToolBlock(name)
            self._tool_blocks[index] = block
            self._append(block)
        block.full_result = text
        block.is_error = is_error
        # Tools often finish faster than the command types out. Let the reveal
        # play to the end first; `_tick` promotes the block once it catches up.
        if block.reveal < len(self._call_target(block)):
            block.result_ready = True
            return
        self._promote_result(index)

    def _promote_result(self, index: int) -> None:
        """Swap a finished, fully-revealed call from pending box to result box."""
        block = self._tool_blocks.pop(index, None)
        if block is None:
            return
        block.result_ready = False
        block.reveal = len(self._call_target(block))  # result shows full command
        block.remove_class("tool-pending")
        block.add_class("tool-error" if block.is_error else "tool-success")
        self._collapsibles.append(block)
        block.update(self._render_tool_result(block))
        self._bump()

    def toggle_tools(self) -> None:
        self.tools_expanded = not self.tools_expanded
        for block in self._collapsibles:
            block.update(self._render_tool_result(block))
        self._bump()

    def _call_detail(self, block: ToolBlock) -> str:
        """The non-bash args preview that types out under the label."""
        detail = _compact_args(_parse_args(block.args_buf))
        if not detail:
            detail = block.args_buf.strip()
            if len(detail) > 240:
                detail = detail[:240] + " ..."
        return detail

    def _call_target(self, block: ToolBlock) -> str:
        """Full text the reveal animation is typing toward."""
        if block.tool_name == "bash":
            return _command_from_args(block.args_buf)
        return self._call_detail(block)

    def _render_tool_call(self, block: ToolBlock) -> RenderableType:
        args = _parse_args(block.args_buf)
        label = _tool_title(block.tool_name, args, theme.ACCENT)
        target = self._call_target(block)
        revealing = block.reveal < len(target)
        shown = target[: block.reveal] if revealing else target
        cursor = revealing and self._blink_on

        if block.tool_name == "bash":
            # Pass the FULL command + a reveal budget (not a pre-sliced prefix):
            # the lexer highlights complete syntax once, and the reveal slices the
            # stable, already-coloured Text — so colours no longer flicker as the
            # type-out crosses token boundaries (quotes, heredocs, etc.).
            reveal = block.reveal if revealing else None
            return Group(label, _shell_text(target, cursor=cursor, reveal=reveal))

        if target:
            text = Text(shown, style=theme.FG_DIM)
            if cursor:
                text.append(_CURSOR, style=theme.FG_DIM)
            return Group(label, _FullWidthLines([_line_with_bg(text, theme.BG_BASE)], wrap=True))
        return label

    def _render_tool_result(self, block: ToolBlock) -> RenderableType:
        args = _parse_args(block.args_buf)
        body = block.full_result or ""
        expanded = block.expanded or self.tools_expanded
        color = theme.ERROR_SOFT if block.is_error else theme.SUCCESS
        title = _tool_title(block.tool_name, args, color)

        if block.tool_name == "bash":
            output, advisory = _split_bash_advisories(body)
            preview, hidden = _preview_lines(output, expanded)
            note = f"... {hidden} more lines (Ctrl+O to expand)" if hidden else None
            box = _terminal_box(
                _command_from_args(block.args_buf),
                output=preview or None,
                note=note,
            )
            if advisory.strip():
                # The empty Text both separates the sections and terminates the
                # terminal box's final full-width row, which otherwise swallows the
                # renderable that immediately follows it.
                return Group(title, box, Text(""), *_render_advisory(advisory, expanded))
            return Group(title, box)

        if block.tool_name in _FILE_VIEW_TOOLS and not block.is_error:
            rendered = self._render_file_tool_result(block.tool_name, args, body, expanded)
            if rendered is not None:
                return Group(title, rendered)

        preview, hidden = _preview_lines(body, expanded)
        out = Text("Result\n", style=color)
        out.append(highlight_refs(preview))
        if hidden:
            out.append(Text(f"\n... {hidden} more lines (Ctrl+O to expand)", style="dim"))
        return Group(title, out)

    def _render_file_tool_result(
        self,
        name: str,
        args: dict[str, Any],
        body: str,
        expanded: bool,
    ) -> RenderableType | None:
        fallback_path = _path_from_args(args)

        if name in _FILE_CHANGE_TOOLS:
            intro_lines, diff, after = _split_diff_section(body)
            parts: list[RenderableType] = []
            if intro_lines:
                parts.append(Text("\n".join(intro_lines), style=theme.FG_BRIGHT))
            if diff:
                parts.append(Text("Diff", style=f"bold {theme.FG_DIM}"))
                parts.append(_render_diff(diff))
            if after:
                path, header, anchored, trailing = _parse_file_view(after, fallback_path)
                if header and anchored:
                    parts.append(Text(header, style=f"bold {theme.FG_DIM}"))
                    limited = anchored
                    hidden = 0
                    if not expanded and len(anchored) > TOOL_PREVIEW_LINES:
                        limited = anchored[:TOOL_PREVIEW_LINES]
                        hidden = len(anchored) - TOOL_PREVIEW_LINES
                    parts.append(_render_anchored_code(path, limited))
                    if hidden:
                        parts.append(Text(f"... {hidden} more lines (Ctrl+O to expand)", style="dim"))
                    _append_trailing(parts, trailing, expanded)
                elif after.strip():
                    preview, hidden = _preview_lines(after, expanded)
                    text = highlight_refs(preview)
                    if hidden:
                        text.append(Text(f"\n... {hidden} more lines (Ctrl+O to expand)", style="dim"))
                    parts.append(text)
            return Group(*parts) if parts else None

        path, header, anchored, trailing = _parse_file_view(body, fallback_path)
        if not anchored:
            return None
        # Repeat reads of the same file omit the anchors header (token saving on
        # the model side). Synthesize a consistent one so the view looks the same
        # on every read instead of degrading to plain text after the first.
        if not header:
            header = _anchor_header(path or fallback_path)
        limited = anchored
        hidden = 0
        if not expanded and len(anchored) > TOOL_PREVIEW_LINES:
            limited = anchored[:TOOL_PREVIEW_LINES]
            hidden = len(anchored) - TOOL_PREVIEW_LINES
        parts = [
            Text(header, style=f"bold {theme.FG_DIM}"),
            _render_anchored_code(path, limited),
        ]
        if hidden:
            parts.append(Text(f"... {hidden} more lines (Ctrl+O to expand)", style="dim"))
        _append_trailing(parts, trailing, expanded)
        return Group(*parts)
