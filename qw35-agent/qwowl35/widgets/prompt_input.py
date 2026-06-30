"""A readline-style multiline prompt, Textual-native.

Python's GNU ``readline`` can't drive a fullscreen Textual app, so this is the
equivalent built on ``TextArea``: it auto-grows with content, Enter submits while
Ctrl+J (or Shift+Enter / Alt+Enter on terminals that can send them) inserts a
newline, Up/Down recall a persisted history (``~/.qwowl35/history``), and large
pastes collapse to a ``[paste #N …]`` token that expands back to the real text on
submit (little-coder behaviour).
"""

from __future__ import annotations

import json
import re

from textual import events
from textual.message import Message
from textual.widgets import TextArea

import theme
from config import (
    HISTORY_DIR,
    HISTORY_FILE,
    HISTORY_MAX,
    PASTE_CHAR_THRESHOLD,
    PASTE_LINE_THRESHOLD,
)

_PASTE_RE = re.compile(r"\[paste #(\d+)[^\]]*\]")


def load_history() -> list[str]:
    """Read history (JSON-lines, so multiline entries round-trip)."""
    if not HISTORY_FILE.exists():
        return []
    out: list[str] = []
    try:
        for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return out[-HISTORY_MAX:]


def save_history(history: list[str]) -> None:
    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(entry, ensure_ascii=False) for entry in history[-HISTORY_MAX:]]
        HISTORY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass  # history is best-effort; never break the app over it


class PromptInput(TextArea):
    DEFAULT_CSS = f"""
    PromptInput {{
        width: 1fr;
        height: auto;
        min-height: 3;
        max-height: 12;
        background: {theme.BG_SURFACE};
        color: {theme.FG_BRIGHT};
        border: none;
        outline: none;
        padding: 1 1 1 0;
        scrollbar-size-vertical: 1;
        scrollbar-background: {theme.BG_SURFACE};
        scrollbar-background-hover: {theme.BG_SURFACE};
        scrollbar-background-active: {theme.BG_SURFACE};
        scrollbar-color: {theme.SCROLL_BAR};
        scrollbar-color-hover: {theme.SCROLL_BAR_HOVER};
        scrollbar-color-active: {theme.SCROLL_BAR_ACTIVE};
        scrollbar-corner-color: {theme.BG_SURFACE};
    }}
    PromptInput:focus {{
        border: none;
        outline: none;
    }}
    PromptInput > .text-area--cursor-line {{
        background: {theme.BG_SURFACE};
    }}
    PromptInput > .text-area--gutter {{
        background: {theme.BG_SURFACE};
    }}
    PromptInput > .text-area--cursor {{
        background: {theme.ACCENT};
        color: {theme.BG_BASE};
    }}
    """

    class Submitted(Message):
        """Posted when the user presses Enter. ``text`` has pastes expanded."""

        def __init__(self, prompt: "PromptInput", text: str) -> None:
            super().__init__()
            self.prompt = prompt
            self.text = text

    def __init__(self, **kwargs) -> None:
        super().__init__(soft_wrap=True, show_line_numbers=False, **kwargs)
        # Keep the cursor visible without rapid flashing. A static caret is the
        # safest default for users sensitive to blinking UI.
        self.cursor_blink = False
        self._history: list[str] = load_history()
        self._hidx: int | None = None  # None = live draft; else index into history
        self._draft = ""
        self._pastes: dict[int, str] = {}
        self._paste_counter = 0

    # ------------------------------------------------------------------ #
    # Submission + history persistence
    # ------------------------------------------------------------------ #
    def append_history(self, text: str) -> None:
        text = text.rstrip("\n")
        if not text or (self._history and self._history[-1] == text):
            return
        self._history.append(text)
        del self._history[:-HISTORY_MAX]
        save_history(self._history)

    def clear(self):
        self._hidx = None
        self._draft = ""
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
        if not self._history:
            return
        if self._hidx is None:
            self._draft = self.text
            self._hidx = len(self._history) - 1
        elif self._hidx > 0:
            self._hidx -= 1
        self._load_entry(self._history[self._hidx])

    def _history_next(self) -> None:
        if self._hidx is None:
            return
        if self._hidx < len(self._history) - 1:
            self._hidx += 1
            self._load_entry(self._history[self._hidx])
        else:
            self._hidx = None
            self._load_entry(self._draft)

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
