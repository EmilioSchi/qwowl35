"""A minimal, keyboard-driven "are you sure you want to quit?" confirmation.

Mirrors :class:`~widgets.theme_selector.ThemeSelector` and
:class:`~widgets.approval.ApprovalModal`: a transparent, centered overlay so the
app painted below stays visible. Two options — No (default) and Yes — toggled
with Left/Right or Up/Down; Enter confirms the highlighted option.

The highlight starts on **No** and Escape also means No, so a reflexive second
Ctrl+C (or a stray Escape) cancels the quit rather than confirming it — which is
the whole point: Ctrl+C is easy to hit by accident.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

import theme

_OPTIONS = ("No", "Yes")


class QuitConfirm(ModalScreen[bool]):
    # Transparent screen (no dim, no border); only the panel paints, matching the
    # theme picker and approval modal.
    DEFAULT_CSS = """
    QuitConfirm { align: center middle; background: transparent; }
    QuitConfirm #panel {
        width: 50%;
        max-width: 48;
        height: auto;
        background: $bg-base;
        border: none;
        padding: 1 2;
    }
    QuitConfirm #question { width: 1fr; color: $fg-bright; padding: 0 1; margin-bottom: 1; }
    QuitConfirm .option { width: 1fr; color: $fg-bright; padding: 0 1; }
    QuitConfirm .option-active { background: $bg-surface; color: $fg-bright; text-style: bold; }
    QuitConfirm #hint { width: 1fr; color: $fg-ghost; padding: 0 1; margin-top: 1; }
    """

    can_focus = True

    def __init__(self) -> None:
        super().__init__()
        self._selected = 0  # default highlight on "No"

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Static("", id="question")
            for i in range(len(_OPTIONS)):
                yield Static("", id=f"opt{i}", classes="option")
            yield Static("", id="hint")

    def on_mount(self) -> None:
        self._repaint()
        self.focus()

    # ------------------------------------------------------------------ #
    # NB: not ``_render`` — that name is a Textual ``Widget`` internal (returns
    # the screen's Visual); shadowing it makes the modal render as a blank crash.
    def _repaint(self) -> None:
        """Repaint the prompt + options. Reads ``theme.*`` for the live palette."""
        self.query_one("#question", Static).update(
            Text("Are you sure you want to quit?", style=theme.FG_BRIGHT)
        )
        for i, label in enumerate(_OPTIONS):
            opt = self.query_one(f"#opt{i}", Static)
            marker = "› " if i == self._selected else "  "
            opt.update(f"{marker}{label}")
            opt.set_class(i == self._selected, "option-active")
        self.query_one("#hint", Static).update(
            Text("←→ choose   enter confirm   esc cancel", style=theme.FG_GHOST)
        )

    def on_key(self, event) -> None:
        key = event.key
        if key in ("left", "right", "up", "down"):
            # Only two options, so any arrow just toggles the highlight.
            self._selected = (self._selected + 1) % len(_OPTIONS)
            self._repaint()
            event.stop()
        elif key in ("1", "2"):
            self._selected = int(key) - 1
            self._repaint()
            self.dismiss(self._selected == 1)
            event.stop()
        elif key == "enter":
            self.dismiss(self._selected == 1)
            event.stop()
        elif key == "escape":
            self.dismiss(False)
            event.stop()
