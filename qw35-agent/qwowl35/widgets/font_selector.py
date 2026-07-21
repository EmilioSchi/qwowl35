"""A keyboard-driven picker for the web-UI font.

Mirrors :class:`~widgets.theme_selector.ThemeSelector`, minus the live preview:
the font is a property of the *browser* page (``--ui webgui``/``gui``), so the
TUI process cannot restyle it — the committed choice is persisted and takes
effect when the browser tab reloads.

    Up/Down    highlight a family
    Enter      confirm (returns the family slug)
    Escape     cancel (returns ``None``)
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

import theme


class FontSelector(ModalScreen["str | None"]):
    # Transparent screen (no dim, no border), matching ThemeSelector.
    DEFAULT_CSS = """
    FontSelector { align: center middle; background: transparent; }
    FontSelector #panel {
        width: 50%;
        max-width: 60;
        height: auto;
        max-height: 80%;
        background: $bg-base;
        border: none;
        padding: 1 2;
    }
    FontSelector #title { width: 1fr; color: $fg-dim; padding: 0 1; }
    FontSelector #font-list {
        width: 1fr;
        height: auto;
        max-height: 60%;
        margin: 1 0;
        scrollbar-size-vertical: 1;
        scrollbar-background: $bg-base;
        scrollbar-background-hover: $bg-base;
        scrollbar-background-active: $bg-base;
        scrollbar-color: $scroll-bar;
        scrollbar-color-hover: $scroll-bar-hover;
        scrollbar-color-active: $scroll-bar-active;
        scrollbar-corner-color: $bg-base;
    }
    FontSelector .option { width: 1fr; color: $fg-bright; padding: 0 1; }
    FontSelector .option-active { background: $bg-surface; color: $fg-bright; text-style: bold; }
    FontSelector #hint { width: 1fr; color: $fg-ghost; padding: 0 1; }
    """

    can_focus = True

    def __init__(self, options: list[tuple[str, str]], current: str) -> None:
        """``options`` is ``(slug, label)`` pairs; ``current`` the active slug."""
        super().__init__()
        self._options = options
        slugs = [slug for slug, _ in options]
        self._selected = slugs.index(current) if current in slugs else 0

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Static("", id="title")
            with VerticalScroll(id="font-list"):
                for i in range(len(self._options)):
                    yield Static("", id=f"font-{i}", classes="option")
            yield Static("", id="hint")

    def on_mount(self) -> None:
        self._repaint()
        self.focus()

    # ------------------------------------------------------------------ #
    def _repaint(self) -> None:
        """Repaint rows + title/hint. Reads ``theme.*`` so it uses the live palette."""
        self.query_one("#title", Static).update(
            Text("Font — web/gui only", style=theme.FG_DIM)
        )
        for i, (_, label) in enumerate(self._options):
            opt = self.query_one(f"#font-{i}", Static)
            marker = "› " if i == self._selected else "  "
            opt.update(f"{marker}{label}")
            opt.set_class(i == self._selected, "option-active")
        self.query_one(f"#font-{self._selected}", Static).scroll_visible()
        self.query_one("#hint", Static).update(
            Text(
                "↑↓ font   enter save   esc cancel — applies after a browser reload",
                style=theme.FG_GHOST,
            )
        )

    def on_key(self, event) -> None:
        key = event.key
        if key == "up":
            self._selected = (self._selected - 1) % len(self._options)
            self._repaint()
            event.stop()
        elif key == "down":
            self._selected = (self._selected + 1) % len(self._options)
            self._repaint()
            event.stop()
        elif key == "enter":
            self.dismiss(self._options[self._selected][0])
            event.stop()
        elif key == "escape":
            self.dismiss(None)
            event.stop()
