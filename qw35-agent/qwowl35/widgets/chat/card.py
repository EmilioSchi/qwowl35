"""The bordered message card frame (User/Spawn/Resume chrome) and the
sub-agent color mapping shared by chips and card edges."""

from __future__ import annotations

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.measure import Measurement
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

import theme


# One source of truth for sub-agent colors, as PALETTE TOKEN NAMES: the chip
# AND the card edge draw from it, so every visual reference to a role shares
# one theme color.
AGENT_COLOR_TOKENS: dict[str, str] = {
    "Explorer": "ACCENT",
    "Editor": "WARNING",
}


def _agent_edge(chip: str | None) -> str:
    """The card-edge palette token for an agent chip name."""
    return AGENT_COLOR_TOKENS.get(chip or "", "ACCENT")


class _CardFrame:
    """A message card: rounded border with a title embedded in the top-left
    edge, an optional agent chip riding the bottom-left edge, and a static
    timestamp on the bottom-right::

        ╭─ Title ──────────────────── top right ─╮
        │ inner content                          │
        ╰─ Chip ────────────────────── 14:32:05 ─╯

    Rich/Textual borders allow only one text per edge, and the bottom edge
    needs two (chip + timestamp), so the frame is hand-drawn in the same
    idiom as the mini terminal chrome. The chip sits one cell after the ╰
    corner (a plain reverse-video block, no ╞/╡ notch glyphs — those render
    broken in some terminal fonts). The border takes the card's scope color
    (``edge``, a palette token name like "ACCENT"); colors are read from
    ``theme.*`` at render time so a live theme switch restyles cards on the
    next repaint. Every visual row ends in a newline (the height contract
    ``_FullWidthLines`` documents). On a too-narrow terminal the top-right
    text, then the timestamp, then the chip are dropped, and the title
    truncates last — the border itself always survives.
    """

    def __init__(
        self,
        inner: RenderableType,
        *,
        title: str,
        top_right: str | None = None,
        chip: str | None = None,
        chip_color: str | None = None,  # resolved from the chip name if None
        timestamp: str | None = None,
        edge: str | None = None,  # palette token name; FG_GHOST if None
    ) -> None:
        self._inner = inner
        self._title = title
        self._top_right = top_right
        self._chip = chip
        self._chip_color = chip_color
        self._timestamp = timestamp
        self._edge = edge

    def _resolve_chip_color(self) -> str:
        if self._chip_color:
            return self._chip_color
        return getattr(theme, _agent_edge(self._chip))

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        # Cards fill their container (widgets are `width: 1fr`); never let a
        # measuring parent fall back to the 80-col default.
        return Measurement(12, options.max_width)

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        width = max(12, options.max_width)
        inner_width = width - 4  # "│ " ... " │"
        border = getattr(theme, self._edge) if self._edge else theme.FG_GHOST

        # -- top edge: ╭─ Title ──( top right )──╮ -------------------------- #
        title = self._title
        right = f" {self._top_right} " if self._top_right else ""
        # cells: "╭─ "(3) + title + " "(1) + filler + right + "─╮"(2)
        filler = width - 6 - len(title) - len(right)
        if filler < 0 and right:
            right = ""
            filler = width - 6 - len(title)
        if filler < 0:
            title = title[: max(1, width - 7)]
            filler = max(0, width - 6 - len(title))
        top = Text(no_wrap=True)
        top.append("╭─ ", style=border)
        top.append(title, style=f"bold {theme.FG_BRIGHT}")
        top.append(" ", style=border)
        top.append("─" * filler, style=border)
        if right:
            top.append(right, style=theme.FG_GHOST)
        top.append("─╮", style=border)
        yield top

        # -- body: inner renderable framed by │ ... │ ----------------------- #
        edge = Style.parse(border)
        body_options = options.update(width=inner_width, height=None)
        rows = console.render_lines(self._inner, body_options, pad=True)
        if not rows:
            # Never let the two border rows touch: an empty card keeps one
            # blank body row.
            rows = [[Segment(" " * inner_width)]]
        for row in rows:
            yield Segment("│ ", edge)
            yield from row
            yield Segment(" │", edge)
            yield Segment.line()

        # -- bottom edge: ╰─ Chip ──( timestamp )──╯ ------------------------ #
        chip = self._chip
        ts = f" {self._timestamp} " if self._timestamp else ""
        left_cells = (len(chip) + 4) if chip else 1  # ╰─ + " Chip ", or ╰
        if left_cells + len(ts) + 2 > width:
            ts = ""
        if chip and left_cells + 2 > width:
            chip = None
            left_cells = 1
        bottom = Text(no_wrap=True)
        if chip:
            bottom.append("╰─", style=border)
            bottom.append(f" {chip} ", style=f"reverse {self._resolve_chip_color()}")
        else:
            bottom.append("╰", style=border)
        filler = max(0, width - left_cells - len(ts) - 2)
        bottom.append("─" * filler, style=border)
        if ts:
            bottom.append(ts, style=theme.FG_GHOST)
        bottom.append("─╯", style=border)
        yield bottom


# Public name for the card frame: app.py wraps the queued-message panel in the
# same chrome the chat's User/Spawn/Resume cards use.
CardFrame = _CardFrame
