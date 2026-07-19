"""A keyboard-driven theme picker with live preview.

Mirrors :class:`~widgets.approval_modal.ApprovalModal`: a transparent, centered
overlay, so the header (mascot/top-row) and footer (footer text + status bar +
prompt) painted on the screen *below* stay visible and recolor live as the
selection moves.

    Up/Down    highlight a theme  → previews it immediately
    Left/Right toggle dark/light  → re-previews
    Enter      confirm (returns ``(name, mode)``)
    Escape     revert to the theme active when the picker opened (returns ``None``)

The app does the actual restyling via ``App.apply_theme_preview``; this screen
only drives the selection and reports the choice.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

import theme


class ThemeSelector(ModalScreen["tuple[str, str] | None"]):
    # Transparent screen (no dim, no border); only the panel paints, so the live
    # header/footer recolor is visible behind it.
    DEFAULT_CSS = """
    ThemeSelector { align: center middle; background: transparent; }
    ThemeSelector #panel {
        width: 50%;
        max-width: 60;
        height: auto;
        max-height: 80%;
        background: $bg-base;
        border: none;
        padding: 1 2;
    }
    ThemeSelector #title { width: 1fr; color: $fg-dim; padding: 0 1; }
    ThemeSelector #theme-list {
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
    ThemeSelector .option { width: 1fr; color: $fg-bright; padding: 0 1; }
    ThemeSelector .option-active { background: $bg-surface; color: $fg-bright; text-style: bold; }
    ThemeSelector #hint { width: 1fr; color: $fg-ghost; padding: 0 1; }
    """

    can_focus = True

    def __init__(self, names: list[str], current: str, mode: str) -> None:
        super().__init__()
        self._names = names
        self._selected = names.index(current) if current in names else 0
        self._original = current
        self._original_mode = mode
        self._mode = mode

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Static("", id="title")
            with VerticalScroll(id="theme-list"):
                for i in range(len(self._names)):
                    yield Static("", id=f"theme-{i}", classes="option")
            yield Static("", id="hint")

    def on_mount(self) -> None:
        self._repaint()
        self.focus()

    # ------------------------------------------------------------------ #
    def _current_name(self) -> str:
        return self._names[self._selected]

    def _repaint(self) -> None:
        """Repaint rows + title/hint. Reads ``theme.*`` so it uses the live palette."""
        self.query_one("#title", Static).update(
            Text(f"Theme — {self._mode}", style=theme.FG_DIM)
        )
        for i, name in enumerate(self._names):
            opt = self.query_one(f"#theme-{i}", Static)
            marker = "› " if i == self._selected else "  "
            opt.update(f"{marker}{name}")
            opt.set_class(i == self._selected, "option-active")
        self.query_one(f"#theme-{self._selected}", Static).scroll_visible()
        self.query_one("#hint", Static).update(
            Text("↑↓ theme   ←→ dark/light   enter apply   esc cancel", style=theme.FG_GHOST)
        )

    def _apply(self) -> None:
        """Preview the current (name, mode), then repaint using the new palette."""
        self.app.apply_theme_preview(self._current_name(), self._mode)
        self._repaint()

    def on_key(self, event) -> None:
        key = event.key
        if key == "up":
            self._selected = (self._selected - 1) % len(self._names)
            self._apply()
            event.stop()
        elif key == "down":
            self._selected = (self._selected + 1) % len(self._names)
            self._apply()
            event.stop()
        elif key in ("left", "right"):
            self._mode = "light" if self._mode == "dark" else "dark"
            self._apply()
            event.stop()
        elif key == "enter":
            self.dismiss((self._current_name(), self._mode))
            event.stop()
        elif key == "escape":
            self.app.apply_theme_preview(self._original, self._original_mode)
            self.dismiss(None)
            event.stop()
