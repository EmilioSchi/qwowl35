"""A minimal, borderless, keyboard-driven bash-approval prompt.

No box, no buttons. Three selectable lines:

    1. Accept
    2. Deny
    3. Write to do differently   (Tab to write)

Number keys or Up/Down move the highlight; Enter confirms. Choosing option 3 (or
pressing Tab) drops into a one-line text field where the user types an alternative
instruction that is relayed to the model instead of running the command.
"""

from __future__ import annotations

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static

import theme
from approval import ApprovalDecision
from widgets.chat_log import _shell_text

_OPTIONS = ("Accept", "Deny", "Write to do differently")


class ApprovalModal(ModalScreen[ApprovalDecision]):
    # Transparent screen (no dim, no border); only the panel paints.
    DEFAULT_CSS = """
    ApprovalModal { align: center middle; background: transparent; }
    ApprovalModal #panel {
        width: 90%;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $bg-base;
        border: none;
        padding: 1 2;
    }
    /* Command/warnings can be long; scroll them with the app's scrollbar
       palette while the options below stay fixed and keyboard-reachable. */
    ApprovalModal #cmd-scroll {
        width: 1fr;
        height: auto;
        max-height: 60%;
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
    ApprovalModal #cmd { width: 1fr; }
    ApprovalModal #warnings { width: 1fr; margin-top: 1; }
    ApprovalModal .option { width: 1fr; color: $fg-bright; padding: 0 1; }
    ApprovalModal .option-active { background: $bg-surface; color: $fg-bright; text-style: bold; }
    ApprovalModal #alt-input {
        display: none;
        width: 1fr;
        margin-top: 1;
        border: none;
        background: $bg-surface;
        color: $fg-bright;
    }
    """

    can_focus = True

    def __init__(self, command: str, warnings: list[str], allowlist_info: str) -> None:
        super().__init__()
        self._command = command
        self._warnings = warnings
        self._allowlist_info = allowlist_info
        self._selected = 0
        self._mode = "select"  # or "alt"

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            with VerticalScroll(id="cmd-scroll"):
                # Same bash-highlighted terminal box the chat transcript uses.
                yield Static(_shell_text(self._command), id="cmd")
                if self._warnings:
                    warnings = Text(
                        "\n".join(f"• {w}" for w in self._warnings),
                        # Matches ChatView's warning accent so callouts read the same.
                        style=theme.WARNING,
                    )
                    yield Static(warnings, id="warnings")
            for i, label in enumerate(_OPTIONS):
                yield Static("", id=f"opt{i}", classes="option")
            yield Input(placeholder="Describe what to do instead…", id="alt-input")

    def on_mount(self) -> None:
        self._render_options()
        self.focus()

    # ------------------------------------------------------------------ #
    def _render_options(self) -> None:
        for i, label in enumerate(_OPTIONS):
            opt = self.query_one(f"#opt{i}", Static)
            marker = "› " if i == self._selected else "  "
            opt.update(f"{marker}{i + 1}. {label}")
            opt.set_class(i == self._selected, "option-active")

    def on_key(self, event) -> None:
        if self._mode == "alt":
            # The Input owns keys in alt mode; only Escape returns to select.
            if event.key == "escape":
                self._exit_alt_mode()
                event.stop()
            return

        key = event.key
        if key in ("1", "2", "3"):
            self._selected = int(key) - 1
            self._render_options()
            self._confirm()
            event.stop()
        elif key == "up":
            self._selected = (self._selected - 1) % len(_OPTIONS)
            self._render_options()
            event.stop()
        elif key == "down":
            self._selected = (self._selected + 1) % len(_OPTIONS)
            self._render_options()
            event.stop()
        elif key == "enter":
            self._confirm()
            event.stop()
        elif key == "tab":
            self._enter_alt_mode()
            event.stop()
        elif key == "escape":
            self.dismiss(ApprovalDecision("deny"))
            event.stop()

    def _confirm(self) -> None:
        if self._selected == 0:
            self.dismiss(ApprovalDecision("accept"))
        elif self._selected == 1:
            self.dismiss(ApprovalDecision("deny"))
        else:
            self._enter_alt_mode()

    def _enter_alt_mode(self) -> None:
        self._mode = "alt"
        self._selected = 2
        self._render_options()
        alt = self.query_one("#alt-input", Input)
        alt.styles.display = "block"
        alt.focus()

    def _exit_alt_mode(self) -> None:
        self._mode = "select"
        alt = self.query_one("#alt-input", Input)
        alt.value = ""
        alt.styles.display = "none"
        self.focus()

    @on(Input.Submitted, "#alt-input")
    def _alt_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.stop()
        if not text:
            return
        self.dismiss(ApprovalDecision("alternative", text))
