"""Shown when a running explorer's round budget runs out, instead of a
silent cutoff. Mirrors :class:`~widgets.approval_modal.ApprovalModal`: a
borderless, keyboard-driven prompt with 2-3 selectable lines:

    1. Stop — use notes so far
    2. Grow to {next tier} ({N} rounds)     (omitted at the top tier)
    3. Force finish now (resume with what you have)

Number keys or Up/Down move the highlight; Enter confirms. Escape defaults
to "Stop" — the safe choice, mirroring QuitConfirm's escape-picks-No
convention — since a stray Escape mid-exploration shouldn't accidentally
burn more budget or force a possibly-premature summary.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

import theme
from agent import BudgetDecision
from agents.explorer import RESUME_NAME, ExplorerBudgetContext

_STOP_LABEL = "Stop — use notes so far"
_FORCE_LABEL = "Force finish now (resume with what you have)"


class ExplorerBudgetModal(ModalScreen[BudgetDecision]):
    DEFAULT_CSS = """
    ExplorerBudgetModal { align: center middle; background: transparent; }
    ExplorerBudgetModal #panel {
        width: 90%;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $bg-base;
        border: none;
        padding: 1 2;
    }
    ExplorerBudgetModal #notes-scroll {
        width: 1fr;
        height: auto;
        max-height: 50%;
        margin-bottom: 1;
        scrollbar-size-vertical: 1;
        scrollbar-background: $bg-base;
        scrollbar-background-hover: $bg-base;
        scrollbar-background-active: $bg-base;
        scrollbar-color: $scroll-bar;
        scrollbar-color-hover: $scroll-bar-hover;
        scrollbar-color-active: $scroll-bar-active;
        scrollbar-corner-color: $bg-base;
    }
    ExplorerBudgetModal #notes { width: 1fr; }
    ExplorerBudgetModal .option { width: 1fr; color: $fg-bright; padding: 0 1; }
    ExplorerBudgetModal .option-active { background: $bg-surface; color: $fg-bright; text-style: bold; }
    ExplorerBudgetModal #hint { width: 1fr; color: $fg-ghost; padding: 0 1; margin-top: 1; }
    """

    can_focus = True

    def __init__(self, context: ExplorerBudgetContext) -> None:
        super().__init__()
        self._context = context
        self._labels = [_STOP_LABEL]
        if context.next_tier is not None:
            tier, rounds = context.next_tier
            self._labels.append(f"Grow to {tier} ({rounds} rounds)")
        self._labels.append(_FORCE_LABEL)
        self._selected = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            with VerticalScroll(id="notes-scroll"):
                text = Text()
                text.append(
                    f"The explorer's round budget ran out ({self._context.max_rounds} "
                    f"rounds at {self._context.effort} effort) before it finished.\n",
                    style=theme.FG_BRIGHT,
                )
                text.append(f"Task: {self._context.task}", style=theme.FG_DIM)
                if self._context.notes_preview:
                    text.append("\n\nNotes so far:\n", style=theme.FG_DIM)
                    text.append(self._context.notes_preview, style=theme.FG_BRIGHT)
                yield Static(text, id="notes")
            for i in range(len(self._labels)):
                yield Static("", id=f"opt{i}", classes="option")
            yield Static("", id="hint")

    def on_mount(self) -> None:
        self._render_options()
        self.focus()

    def _render_options(self) -> None:
        for i, label in enumerate(self._labels):
            opt = self.query_one(f"#opt{i}", Static)
            marker = "› " if i == self._selected else "  "
            opt.update(f"{marker}{i + 1}. {label}")
            opt.set_class(i == self._selected, "option-active")
        self.query_one("#hint", Static).update(
            Text("↑↓ choose   enter confirm   esc stop", style=theme.FG_GHOST)
        )

    def on_key(self, event) -> None:
        key = event.key
        count = len(self._labels)
        if key.isdigit() and 1 <= int(key) <= count:
            self._selected = int(key) - 1
            self._render_options()
            self._confirm()
            event.stop()
        elif key == "up":
            self._selected = (self._selected - 1) % count
            self._render_options()
            event.stop()
        elif key == "down":
            self._selected = (self._selected + 1) % count
            self._render_options()
            event.stop()
        elif key == "enter":
            self._confirm()
            event.stop()
        elif key == "escape":
            self.dismiss(BudgetDecision(kind="stop"))
            event.stop()

    def _confirm(self) -> None:
        label = self._labels[self._selected]
        if label == _STOP_LABEL:
            self.dismiss(BudgetDecision(kind="stop"))
        elif label == _FORCE_LABEL:
            self.dismiss(BudgetDecision(kind="force", forced_tool=RESUME_NAME))
        else:
            # The "Grow to {tier}" option — only present when next_tier is set.
            _, rounds = self._context.next_tier
            self.dismiss(BudgetDecision(kind="grow", max_rounds=rounds))
