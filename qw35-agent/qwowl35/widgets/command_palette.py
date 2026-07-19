"""A non-interrupting slash-command palette shown above the prompt.

Unlike the theme/session pickers (focus-stealing ``ModalScreen``s), this is a
plain ``Static`` mounted as a sibling inside the bottom ``#prompt-dock`` and
toggled with ``.display`` — the same recipe as ``#queue-panel``. Because the
dock is ``dock: bottom; height: auto``, showing it grows the column upward like
a completion dropdown while the ``PromptInput`` keeps focus, so the user keeps
typing to filter the list.

The app owns the open/close decision (it watches the prompt text); this widget
only holds the filtered matches + highlight and renders them. ``render_palette``
is a pure helper reading ``theme.*`` at render time, so an open palette recolors
live when the theme changes.
"""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

import theme
from commands import CommandSpec, filter_commands


# Blank columns between the name (+ arg hint) column and the description column.
_DESC_GAP = 2


def _name_part(spec: CommandSpec) -> str:
    """The name plus its inline arg hint, exactly as rendered — used to size the
    name column so every description lines up."""
    if spec.takes_args and spec.arg_hint:
        return f"{spec.name} {spec.arg_hint}"
    return spec.name


def render_palette(matches: list[CommandSpec], selected: int, query: str) -> Text:
    """One row per match: a ``›`` marker + name (+ arg hint) + description. The
    name column is padded to the widest entry shown so every description starts
    at the same offset. The highlighted row is bright/bold; the rest are dim.
    Empty match list renders a single ghost "no matching command" line."""
    if not matches:
        return Text("no matching command", style=theme.FG_GHOST)
    name_width = max(len(_name_part(spec)) for spec in matches)
    out = Text()
    for i, spec in enumerate(matches):
        if i:
            out.append("\n")
        active = i == selected
        out.append("› " if active else "  ", style=theme.ACCENT)
        out.append(
            spec.name, style=f"bold {theme.FG_BRIGHT}" if active else theme.FG_DIM
        )
        if spec.takes_args and spec.arg_hint:
            out.append(f" {spec.arg_hint}", style=theme.FG_GHOST)
        # Pad the name column to align descriptions, then a fixed gap.
        out.append(" " * (name_width - len(_name_part(spec)) + _DESC_GAP))
        out.append(
            spec.description, style=theme.FG_DIM if active else theme.FG_GHOST
        )
    return out


class CommandPalette(Static):
    DEFAULT_CSS = """
    CommandPalette {
        display: none;              /* toggled like #queue-panel */
        width: 1fr;
        max-height: 10;
        background: $bg-surface;    /* reads as attached to the input bar */
        color: $fg-dim;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__("", id="command-palette")
        self.matches: list[CommandSpec] = []
        self.selected = 0
        self.query = ""

    def update_query(self, query: str) -> None:
        """Re-filter for ``query`` (the text after the leading ``/``), clamp the
        highlight into range, and repaint."""
        self.query = query
        self.matches = filter_commands(query)
        self.selected = min(self.selected, len(self.matches) - 1) if self.matches else 0
        self.repaint()

    def move(self, delta: int) -> None:
        """Move the highlight, wrapping. No-op when nothing matches."""
        if self.matches:
            self.selected = (self.selected + delta) % len(self.matches)
            self.repaint()

    def current_spec(self) -> CommandSpec | None:
        return self.matches[self.selected] if self.matches else None

    def repaint(self) -> None:
        self.update(render_palette(self.matches, self.selected, self.query))
