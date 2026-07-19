"""The collapsed/expandable reasoning card with its animated label and tips."""

from __future__ import annotations

import random
import time

from rich.text import Text
from textual import events
from textual.widgets import Static

import theme
from widgets.chat.primitives import _shimmer_text
from widgets.status_bar import compact_count, rough_token_count


# Short command/key reminders shown under the streaming "Thinking ..." label,
# one picked at random per reasoning segment. Every entry must describe a real
# feature (slash commands in app._dispatch_command, keys in app.BINDINGS and
# prompt_input) — keep each under ~70 chars so the row rarely wraps.
_TIPS: tuple[str, ...] = (
    "type /clear to start a new conversation",
    "type /theme to pick a different color theme",
    "press Ctrl+O to expand or collapse tool output",
    "press Ctrl+J for a newline without sending",
    "press Enter while a turn is running to queue your next message",
    "click a Thinking or tool badge to expand it",
    "type /quit — or press Ctrl+C — to exit",
)


def _pick_tip(rng: random.Random | None = None) -> str:
    chooser = rng or random
    return chooser.choice(_TIPS)


def _format_elapsed(seconds: int) -> str:
    """"33s" under a minute, "1m 12s" beyond — compact, single-line friendly."""
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def _thinking_card(
    body: str,
    expanded: bool,
    done: bool,
    frame: int,
    stats: str | None = None,
    tip: str | None = None,
) -> Text:
    """The thinking card: an animated (or frozen) label, plus the reasoning
    body when expanded.

    While streaming (``done=False``) the label shimmers via ``_shimmer_text``.
    Once done the label freezes to the static faint-italic style. The body is
    never animated: it stays unstyled and inherits the ``.thinking`` CSS.

    ``stats`` (pre-formatted, e.g. "33s · ↓ 900 tokens") and ``tip`` decorate
    only the streaming card — a ghost-gray suffix on the label and a
    "⮑  Tip: ..." row beneath it; both vanish when the card freezes. Neither
    carries the ``dim`` attribute: the bouncing label character must stay the
    only dim span.
    """
    marker = "⌄" if expanded else "»"
    label = f"{marker} Thinking ..."
    text = Text()
    if done:
        text.append(label, style=f"italic {theme.FG_FAINT}")
    else:
        text.append_text(_shimmer_text(label, frame))
    if not done and stats:
        text.append(f" ({stats})", style=theme.FG_GHOST)
    if not done and tip:
        text.append(f"\n⮑  Tip: {tip}", style=theme.FG_GHOST)
    if expanded and body:
        text.append("\n")
        text.append(body)
    return text


class ThinkingBlock(Static):
    """One reasoning segment: a collapsed one-line label, click to expand.

    The block owns its text and expand state so every historical segment stays
    independently togglable after it froze; ``ChatView._tick`` drives the
    label animation of the (single) streaming block via ``anim_frame``.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        super().__init__(classes="msg thinking")
        self.body = ""
        self.expanded = False
        self.done = False       # flush_reasoning froze this card
        self.dirty = False      # body grew since the last paint
        self.anim_frame = 0     # label animation frame
        self.started_mono = time.monotonic()  # anchor for the streaming stats
        self.tip = _pick_tip(rng)             # picked once per segment

    def render_card(self) -> Text:
        stats = tip = None
        if not self.done:
            elapsed = _format_elapsed(int(time.monotonic() - self.started_mono))
            tokens = compact_count(rough_token_count(self.body))
            stats = f"{elapsed} · ↓ {tokens} tokens"
            tip = self.tip
        return _thinking_card(self.body, self.expanded, self.done, self.anim_frame,
                              stats=stats, tip=tip)

    def repaint(self) -> None:
        if self.is_mounted:
            self.update(self.render_card())
        self.dirty = False

    def toggle(self) -> None:
        self.expanded = not self.expanded
        self.repaint()

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.toggle()
        # Lazy import: chat_view imports this module.
        from widgets.chat.chat_view import ChatView

        parent = self.parent
        if isinstance(parent, ChatView):
            parent._bump()
