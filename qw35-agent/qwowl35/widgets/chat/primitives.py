"""Low-level renderables shared across the chat widgets: full-width line
blocks, justified chrome rows, clickable spans, reference highlighting, and
the shimmer label animation."""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass

from rich.console import Console, ConsoleOptions, RenderResult, RenderableType
from rich.measure import Measurement
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

import theme


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


_CURSOR = "▋"  # trailing block cursor while a command types out
# Shell / code / diff colors are read from ``theme.*`` at render time (below) so
# they follow a live theme change; there are no baked color aliases here.


# The collapsed "Thinking ..." label animation: a soft hue drift spread across
# the label's characters plus one bouncing dimmed character, running only while
# reasoning is streaming. Colors are the active theme's accent read at render
# time — its saturation/value verbatim, only the hue drifting — so the label
# follows theme switches and keeps each theme's color character.
_THINK_HUE_SPAN = 0.3    # hue-wheel spread across the label string
_THINK_HUE_DRIFT = 0.02  # hue offset advance per label frame


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


class _JustifiedRow:
    """One full-width chrome row: ``left`` and ``right`` justified apart on a
    ``bg`` fill — the mini terminal window's title/footer bars.

    Yields exactly one visual row. The Text's default ``end`` newline satisfies
    the same height contract ``_FullWidthLines`` documents (Textual counts
    newlines to measure widget height); on a too-narrow terminal the left side
    truncates so the right-side controls stay reachable.
    """

    def __init__(self, left: Text, right: Text, bg: str) -> None:
        self._left = left
        self._right = right
        self._bg = bg

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        width = max(1, options.max_width)
        left = self._left.copy()
        right = self._right.copy()
        if left.cell_len + 1 + right.cell_len > width:
            left.truncate(max(0, width - right.cell_len - 1))
        gap = max(1, width - left.cell_len - right.cell_len)
        row = Text(no_wrap=True)
        row.append_text(left)
        row.append(" " * gap)
        row.append_text(right)
        # Background only — the spans' colors and @click metas layer on top.
        row.stylize(f"on {self._bg}", 0, len(row.plain))
        yield row


def _click_span(label: str, action: str, color: str) -> Text:
    """A clickable chrome control: Textual routes a click on any segment whose
    style meta carries ``@click`` to ``action_<name>`` on the widget that
    rendered it (``ToolBlock`` here), and paints its hover style for free."""
    return Text(label, style=Style.parse(color) + Style(meta={"@click": action}))


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


def _hex_hsv(hex_color: str) -> tuple[float, float, float]:
    """HSV (0..1 each) of a ``#rrggbb`` color. An achromatic color degrades
    to hue 0 (red-centered drift) — acceptable for a decorative label."""
    stripped = hex_color.lstrip("#")
    r, g, b = (int(stripped[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hsv(r, g, b)


def _shimmer_text(label: str, frame: int) -> Text:
    """An animated label: every character wears the theme accent's own color
    with only its hue drifting — span ``_THINK_HUE_SPAN`` across the string,
    offset advancing ``_THINK_HUE_DRIFT`` per frame — and one character
    ping-pongs across the label carrying a dim attribute. Saturation and
    value are inherited verbatim from ``theme.ACCENT``, so the label keeps
    each theme's color character instead of an assumed pastel."""
    text = Text()
    n = len(label)
    base, sat, val = _hex_hsv(theme.ACCENT)
    offset = (frame * _THINK_HUE_DRIFT) % 1.0
    period = max(2 * (n - 1), 1)
    t = frame % period
    bounce = t if t < n else period - t  # ping-pong 0..n-1..0
    for i, ch in enumerate(label):
        hue = (base + offset + (i / max(n - 1, 1) - 0.5) * _THINK_HUE_SPAN) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        style = f"dim italic {color}" if i == bounce else f"italic {color}"
        text.append(ch, style=style)
    return text


_EDGE_BAR = "▌ "  # solid left bar + gap, 2 cells like the "> " it replaced


class _BlockquoteFrame:
    """A quoted-message frame with a solid left bar::

        ▌ User  16:47:00
        ▌ This is the message content

    The bar uses the edge color (a palette token name like ``"FG_MUTED"``);
    the title is bold and bright; the timestamp is ghosted. Colors are read
    from ``theme.*`` at render time so a live theme switch restyles on the
    next repaint. Every visual row ends in a newline (the height contract
    ``_FullWidthLines`` documents). On a too-narrow terminal the timestamp is
    dropped first, then the title truncates.
    """

    def __init__(
        self,
        inner: RenderableType,
        *,
        title: str,
        timestamp: str | None = None,
        edge: str | None = None,  # palette token name; FG_MUTED if None
    ) -> None:
        self._inner = inner
        self._title = title
        self._timestamp = timestamp
        self._edge = edge

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        return Measurement(12, options.max_width)

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        width = max(12, options.max_width)
        inner_width = width - len(_EDGE_BAR)
        edge_color = getattr(theme, self._edge) if self._edge else theme.FG_MUTED

        # -- header row: ▌ Title  timestamp ──────────────────────────────────── #
        title = self._title
        ts = f"  {self._timestamp}" if self._timestamp else ""
        if len(_EDGE_BAR) + len(title) + len(ts) > width:
            ts = ""
        if len(_EDGE_BAR) + len(title) > width:
            title = title[: max(1, inner_width)]
        header = Text(no_wrap=True)
        header.append(_EDGE_BAR, style=edge_color)
        header.append(title, style=f"bold {theme.FG_BRIGHT}")
        if ts:
            header.append(ts, style=theme.FG_GHOST)
        yield header

        # -- body rows: ▌ content ---------------------------------------------- #
        edge_style = Style.parse(edge_color)
        body_options = options.update(width=inner_width, height=None)
        rows = console.render_lines(self._inner, body_options, pad=True)
        if not rows:
            rows = [[Segment(" " * inner_width)]]
        for row in rows:
            yield Segment(_EDGE_BAR, edge_style)
            yield from row
            yield Segment.line()


# Public name for the blockquote frame: app.py and chat_view.py use it for
# user messages and the queued-message panel.
BlockquoteFrame = _BlockquoteFrame
