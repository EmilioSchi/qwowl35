"""The mini terminal window chrome: title/footer bars, PS1 prompt, body
backdrop, per-window ids, and the registered server host shown in prompts."""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.text import Text

import theme
from config import TERMINAL_BODY_LINES
from widgets.chat.primitives import (
    _BlockLine,
    _FullWidthLines,
    _JustifiedRow,
    _click_span,
    _line_with_bg,
)


_term_seq = 0


def _next_term_hash() -> str:
    """A short hex id naming one mini terminal window (``terminal #79b1``).

    A Knuth-mixed counter, not a content hash: the command is still streaming
    when the window appears, and the id must never change under the user."""
    global _term_seq
    _term_seq += 1
    return f"{(_term_seq * 2654435761) & 0xFFFF:04x}"


def _term_bg() -> str:
    """The terminal body's background: the active theme's ``BG_BASE`` pushed
    darker, so the window reads as its own surface against the chat behind it.
    Derived (not a palette token) so every theme gets it for free."""
    h = theme.BG_BASE.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    factor = 0.55 if theme.is_dark() else 0.90
    return "#{:02x}{:02x}{:02x}".format(*(int(c * factor) for c in (r, g, b)))


def _window_title_row(title: str, color: str, *, collapsed: bool = False) -> _JustifiedRow:
    """The mini terminal's title bar: window name left, `- □ x` controls right,
    traffic-light colored from the active theme."""
    left = Text()
    left.append(" ")
    left.append(title, style=color)
    controls = Text()
    controls.append_text(_click_span(" + " if collapsed else " - ", "win_collapse", theme.WARNING))
    controls.append_text(_click_span(" □ ", "win_expand", theme.SUCCESS))
    controls.append_text(_click_span(" x ", "win_close", theme.ERROR))
    return _JustifiedRow(left, controls, theme.BG_SURFACE)


def _window_footer_row(started_at: str, tokens: int | None) -> _JustifiedRow:
    """The mini terminal's bottom bar: copy button left, run time (+ a rough
    chars/4 token estimate once the result is in — the server reports usage
    per turn only, never per call) right."""
    left = Text()
    # Static underline: with auto_links off there is no hover restyle, so the
    # button carries its own affordance.
    left.append_text(_click_span(" copy ", "win_copy", f"underline {theme.ACCENT}"))
    right = Text()
    right.append(started_at, style=theme.FG_GHOST)
    if tokens is not None:
        right.append(f" · ~{tokens} tok", style=theme.FG_GHOST)
    right.append(" ")
    return _JustifiedRow(left, right, theme.BG_SURFACE)


_TERM_HOST: str | None = None


def set_terminal_host(host: str | None) -> None:
    """Name the PS1 prompt's host: the qw35 server's address (set by the app
    at startup from ``--base-url``). Unset falls back to the local hostname."""
    global _TERM_HOST
    _TERM_HOST = host or None


def _prompt_text(host: str, path: str) -> Text:
    """PS1-style ``qwowl@{host}:{path}$ `` prompt for the terminal's first
    line — user, ``@`` and host each in their own color."""
    prompt = Text()
    prompt.append("qwowl", style=theme.SUCCESS)
    prompt.append("@", style=theme.FG_MUTED)
    prompt.append(host, style=theme.WARNING)
    prompt.append(":", style=theme.FG_DIM)
    prompt.append(path, style=theme.ACCENT)
    prompt.append("$ ", style=theme.FG_BRIGHT)
    return prompt


def _head_capped_rows(
    command_rows: list[_BlockLine],
    output_rows: list[_BlockLine],
    budget: int = TERMINAL_BODY_LINES,
) -> tuple[list[_BlockLine], int]:
    """HEAD-view body cap: the command always shows in full; output fills
    whatever budget remains, earliest lines first."""
    remaining = max(1, budget - len(command_rows))
    if len(output_rows) <= remaining:
        return command_rows + output_rows, 0
    return command_rows + output_rows[:remaining], len(output_rows) - remaining


def _terminal_window(
    *,
    title: str,
    color: str,
    body_rows: list[_BlockLine],
    collapsed: bool,
    started_at: str,
    tokens: int | None,
    note: str | None = None,
    bg: str | None = None,
) -> RenderableType:
    """A bash call dressed as a mini terminal window: title bar with `- □ x`
    controls, dark body, footer with copy/time/tokens. Collapsed = title only."""
    title_row = _window_title_row(title, color, collapsed=collapsed)
    if collapsed:
        return Group(title_row)
    rows = list(body_rows)
    if note:
        rows.append(_line_with_bg(Text(note, style="dim"), bg or theme.BG_BASE))
    return Group(
        title_row,
        _FullWidthLines(rows, wrap=True),
        _window_footer_row(started_at, tokens),
    )
