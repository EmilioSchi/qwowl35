"""A readline-style multiline prompt, Textual-native.

Python's GNU ``readline`` can't drive a fullscreen Textual app, so this is the
equivalent built on ``TextArea``: it auto-grows with content, Enter submits while
Ctrl+J (or Shift+Enter / Alt+Enter on terminals that can send them) inserts a
newline, Up/Down recall a persisted history (stored in the OS cache dir — see
``history.py``), and large pastes collapse to a ``[paste #N …]`` token that expands
back to the real text on submit (little-coder behaviour).
"""

from __future__ import annotations

import re

from textual import events
from textual.message import Message
from textual.widgets import TextArea

from config import PASTE_CHAR_THRESHOLD, PASTE_LINE_THRESHOLD
from history import MessageHistory

_PASTE_RE = re.compile(r"\[paste #(\d+)[^\]]*\]")


class PromptInput(TextArea):
    DEFAULT_CSS = """
    PromptInput {
        width: 1fr;
        height: auto;
        min-height: 3;
        max-height: 12;
        background: $bg-surface;
        color: $fg-bright;
        border: none;
        outline: none;
        padding: 1 1 1 0;
        scrollbar-size-vertical: 1;
        scrollbar-background: $bg-surface;
        scrollbar-background-hover: $bg-surface;
        scrollbar-background-active: $bg-surface;
        scrollbar-color: $scroll-bar;
        scrollbar-color-hover: $scroll-bar-hover;
        scrollbar-color-active: $scroll-bar-active;
        scrollbar-corner-color: $bg-surface;
    }
    PromptInput:focus {
        border: none;
        outline: none;
    }
    PromptInput > .text-area--cursor-line {
        background: $bg-surface;
    }
    PromptInput > .text-area--gutter {
        background: $bg-surface;
    }
    PromptInput > .text-area--cursor {
        background: $accent;
        color: $bg-base;
    }
    """

    class Submitted(Message):
        """Posted when the user presses Enter. ``text`` has pastes expanded."""

        def __init__(self, prompt: "PromptInput", text: str) -> None:
            super().__init__()
            self.prompt = prompt
            self.text = text

    def __init__(self, *, history: MessageHistory | None = None, **kwargs) -> None:
        super().__init__(soft_wrap=True, show_line_numbers=False, **kwargs)
        # Keep the cursor visible without rapid flashing. A static caret is the
        # safest default for users sensitive to blinking UI.
        self.cursor_blink = False
        # Injectable so tests can point history at a temp dir; entries + the
        # Up/Down navigation cursor live inside the container.
        self._history = history or MessageHistory()
        self._pastes: dict[int, str] = {}
        self._paste_counter = 0

    # ------------------------------------------------------------------ #
    # Submission + history persistence
    # ------------------------------------------------------------------ #
    def append_history(self, text: str) -> None:
        self._history.append(text)

    def clear(self):
        self._history.reset_navigation()
        return super().clear()

    def _expand(self, raw: str) -> str:
        return _PASTE_RE.sub(
            lambda m: self._pastes.get(int(m.group(1)), m.group(0)), raw
        )

    def _submit(self) -> None:
        self.post_message(self.Submitted(self, self._expand(self.text)))

    # ------------------------------------------------------------------ #
    # Key handling
    # ------------------------------------------------------------------ #
    async def _on_key(self, event: events.Key) -> None:
        key = event.key
        if key == "enter":
            event.prevent_default()
            event.stop()
            self._submit()
            return
        # Ctrl+J is the literal line-feed every terminal can send, so it always
        # works; shift+enter / alt+enter only arrive on terminals that speak the
        # Kitty keyboard protocol (Kitty, Ghostty, recent WezTerm/iTerm2) — on
        # the rest they're indistinguishable from a plain Enter.
        if key in ("shift+enter", "alt+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        if key == "up" and self.cursor_at_first_line:
            event.prevent_default()
            event.stop()
            self._history_prev()
            return
        if key == "down" and self.cursor_at_last_line:
            event.prevent_default()
            event.stop()
            self._history_next()
            return
        await super()._on_key(event)

    def _load_entry(self, value: str) -> None:
        self.text = value
        self.move_cursor(self.document.end)

    def _history_prev(self) -> None:
        entry = self._history.prev(self.text)
        if entry is not None:
            self._load_entry(entry)

    def _history_next(self) -> None:
        entry = self._history.next()
        if entry is not None:
            self._load_entry(entry)

    # ------------------------------------------------------------------ #
    # Paste collapsing
    # ------------------------------------------------------------------ #
    async def _on_paste(self, event: events.Paste) -> None:
        text = event.text
        if not text:
            return
        event.prevent_default()
        event.stop()
        lines = text.count("\n") + 1
        chars = len(text)
        if lines > PASTE_LINE_THRESHOLD or chars > PASTE_CHAR_THRESHOLD:
            self._paste_counter += 1
            n = self._paste_counter
            self._pastes[n] = text
            placeholder = (
                f"[paste #{n} +{lines} lines]"
                if lines > PASTE_LINE_THRESHOLD
                else f"[paste #{n} {chars} chars]"
            )
            self.insert(placeholder)
            return
        self.insert(text)
