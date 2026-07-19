"""Keyboard-driven pickers for /sessions: the session list, then the turn
cut-off.

Both mirror :class:`~widgets.theme_selector.ThemeSelector`: a transparent,
centered overlay with a scrolling option list. :class:`SessionSelector` rows
are restorable sessions — when each was last active, its last mode, how many
turns it holds, and a preview of its first goal; enter hands the chosen
session hash back. :class:`TurnSelector` then lists that session's turns in
conversation order (highlight starts on the last one) and enter restores
everything up to the highlighted turn; escape steps back to the session
list.

The app performs the actual restore; these screens only drive the selection
and report the choice.
"""

from __future__ import annotations

import asyncio
import time

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

import theme
from modes import Mode

GOAL_PREVIEW_CHARS = 80
CHAT_MSG_PREVIEW_CHARS = 72


def _preview(text: str, limit: int = GOAL_PREVIEW_CHARS) -> str:
    flat = " ".join(str(text).split())
    if len(flat) > limit:
        return flat[:limit] + "…"
    return flat


def _message_line(message: dict) -> str:
    """One verbatim chat_messages entry as a compact ``role ▸ body`` line."""
    role = str(message.get("role", "?"))
    content = message.get("content")
    body = _preview(content, CHAT_MSG_PREVIEW_CHARS) if isinstance(content, str) else ""
    calls = message.get("tool_calls") or []
    names = [
        str(call.get("function", {}).get("name", "?"))
        for call in calls
        if isinstance(call, dict)
    ]
    line = f"{role} ▸ {body}".rstrip()
    if names:
        line = f"{line}  ⚙ {', '.join(names)}".rstrip()
    return line


def render_restore_payload(turns: list) -> Text:
    """What restoring these turns re-feeds the model, mirroring
    :meth:`Orchestrator.restore`: the ``(goal → outcome)`` turn log, plus the
    verbatim ``chat_messages`` a CHAT turn re-extends. This is the literal
    prompt state a restore rebuilds — not a rendered conversation."""
    text = Text()
    if not turns:
        text.append("no restorable turns", style=theme.FG_GHOST)
        return text
    for index, turn in enumerate(turns):
        if index:
            text.append("\n")
        text.append(f"turn {index + 1} ", style=theme.FG_BRIGHT)
        text.append(f"[{turn.mode or 'normal'}]\n", style=theme.ACCENT)
        text.append("  goal:    ", style=theme.FG_DIM)
        text.append(f"{_preview(turn.goal, 100)}\n", style=theme.FG_BRIGHT)
        outcome = (turn.outcome or "").strip()
        text.append("  outcome: ", style=theme.FG_DIM)
        text.append(
            f"{_preview(outcome, 100) or '—'}\n",
            style=theme.FG_DIM if outcome else theme.FG_GHOST,
        )
        messages = turn.chat_messages or []
        if messages:
            text.append(f"  chat_messages: {len(messages)}\n", style=theme.FG_DIM)
            for message in messages:
                text.append(f"    {_message_line(message)}\n", style=theme.FG_GHOST)
        else:
            text.append("  chat_messages: ", style=theme.FG_DIM)
            text.append("(none — non-CHAT turn)\n", style=theme.FG_GHOST)
    text.append("\n")
    text.append(
        "on restore ▸ turn-log (goal→outcome) re-seeded; CHAT messages re-fed "
        "verbatim; the server re-primes its KV cache on the next request.",
        style=theme.FG_GHOST,
    )
    return text


def _format_when(last_active: str) -> str:
    """Turn a ``YYYYMMDD-HHMMSS`` stamp into a friendly ``18 Jul 17:19``."""
    try:
        parsed = time.strptime(last_active, "%Y%m%d-%H%M%S")
    except (ValueError, TypeError):
        return last_active or "—"
    return time.strftime("%d %b %H:%M", parsed)


def _mode_badge(mode_value: str) -> Text:
    """The active mode as a small Vim-style inverted box. One calm,
    theme-harmonious colour for every mode (the uppercased label already says
    which mode it is) so the list doesn't read as a wall of mode colours."""
    try:
        mode = Mode(mode_value or "normal")
    except ValueError:
        mode = Mode.NORMAL
    badge = Text()
    badge.append(
        f" {mode.value.upper()} ", style=f"{theme.BG_BASE} on {theme.FG_DIM} bold"
    )
    return badge


def _summary_text(summary, selected: bool) -> Text:
    """A two-line session row: the goal on top (the thing you actually
    recognise a session by), then a dim meta line (mode badge · turns · when).
    The highlight bar comes from the ``.option-active`` background; colour
    here carries the hierarchy."""
    goal = _preview(summary.first_goal, 120) or "(no goal)"
    turns = summary.turn_count
    noun = "turn" if turns == 1 else "turns"

    text = Text()
    text.append("› " if selected else "  ", style=theme.ACCENT)
    text.append(
        goal, style=f"bold {theme.FG_BRIGHT}" if selected else theme.FG_DIM
    )
    text.append("\n  ")
    text.append_text(_mode_badge(summary.last_mode))
    text.append(f" · {turns} {noun} · {_format_when(summary.last_active)}",
                style=theme.FG_GHOST)
    return text


def _turn_row(index: int, turn) -> str:
    mode = turn.mode or "normal"
    return f"{index + 1:04d} · {mode} · {_preview(turn.goal)}"


class SessionSelector(ModalScreen["str | None"]):
    DEFAULT_CSS = """
    SessionSelector { align: center middle; background: transparent; }
    SessionSelector #panel {
        width: 100%;
        max-width: 100%;
        height: 100%;
        max-height: 100%;
        background: $bg-base;
        border: none;
        padding: 0 1;
    }
    SessionSelector #title { width: 1fr; color: $fg-dim; padding: 0 1; }
    SessionSelector #body { width: 1fr; height: 1fr; margin: 1 0; }
    SessionSelector #session-list {
        width: 42%;
        height: 1fr;
        scrollbar-size-vertical: 1;
        scrollbar-background: $bg-base;
        scrollbar-background-hover: $bg-base;
        scrollbar-background-active: $bg-base;
        scrollbar-color: $scroll-bar;
        scrollbar-color-hover: $scroll-bar-hover;
        scrollbar-color-active: $scroll-bar-active;
        scrollbar-corner-color: $bg-base;
    }
    SessionSelector #preview {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        border-left: solid $bg-surface;
        scrollbar-size-vertical: 1;
        scrollbar-background: $bg-base;
        scrollbar-background-hover: $bg-base;
        scrollbar-background-active: $bg-base;
        scrollbar-color: $scroll-bar;
        scrollbar-color-hover: $scroll-bar-hover;
        scrollbar-color-active: $scroll-bar-active;
        scrollbar-corner-color: $bg-base;
    }
    SessionSelector #preview-body { width: 1fr; color: $fg-dim; }
    SessionSelector .option {
        width: 1fr;
        height: auto;
        padding: 0 1;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    SessionSelector .option-active { background: $bg-surface; }
    SessionSelector #hint { width: 1fr; color: $fg-ghost; padding: 0 1; }
    """

    can_focus = True

    def __init__(self, summaries: list, load_turns=None) -> None:
        super().__init__()
        self._summaries = summaries
        self._selected = 0
        # hash -> list[RestoredTurn]; loaded lazily off-thread as the highlight
        # moves so the preview stays snappy and disk I/O never blocks the UI.
        self._load_turns = load_turns
        self._preview_cache: dict = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Static("", id="title")
            with Horizontal(id="body"):
                with VerticalScroll(id="session-list"):
                    for i in range(len(self._summaries)):
                        yield Static("", id=f"session-{i}", classes="option")
                with VerticalScroll(id="preview"):
                    yield Static("", id="preview-body")
            yield Static("", id="hint")

    def on_mount(self) -> None:
        self._repaint()
        self._update_preview()
        self.focus()

    # ------------------------------------------------------------------ #
    def _current_hash(self) -> str:
        return self._summaries[self._selected].session_hash

    def _repaint(self) -> None:
        self.query_one("#title", Static).update(
            Text(f"Sessions — {len(self._summaries)} saved", style=theme.FG_DIM)
        )
        for i, summary in enumerate(self._summaries):
            opt = self.query_one(f"#session-{i}", Static)
            opt.update(_summary_text(summary, i == self._selected))
            opt.set_class(i == self._selected, "option-active")
        self.query_one(f"#session-{self._selected}", Static).scroll_visible()
        self.query_one("#hint", Static).update(
            Text("↑↓ select   enter restore   esc cancel", style=theme.FG_GHOST)
        )

    def _update_preview(self) -> None:
        """Show the highlighted session's restore payload, loading it (once,
        off-thread) if it is not already cached."""
        session_hash = self._current_hash()
        if session_hash in self._preview_cache:
            self._paint_preview(self._preview_cache[session_hash])
            return
        if self._load_turns is None:
            self._paint_preview(None)
            return
        self.query_one("#preview-body", Static).update(
            Text("loading…", style=theme.FG_GHOST)
        )
        self.run_worker(
            self._load_preview(session_hash), group="preview", exclusive=True
        )

    async def _load_preview(self, session_hash: str) -> None:
        try:
            turns = await asyncio.to_thread(self._load_turns, session_hash)
        except Exception:  # noqa: BLE001 - preview is best-effort
            turns = None
        self._preview_cache[session_hash] = turns
        # Only paint if this session is still the highlighted one (rapid
        # arrowing may have moved on) and the screen is still up.
        try:
            if self._current_hash() == session_hash:
                self._paint_preview(turns)
        except Exception:  # noqa: BLE001 - screen dismissed mid-load
            pass

    def _paint_preview(self, turns) -> None:
        body = self.query_one("#preview-body", Static)
        if turns is None:
            body.update(Text("(preview unavailable)", style=theme.FG_GHOST))
        else:
            body.update(render_restore_payload(turns))
        self.query_one("#preview", VerticalScroll).scroll_home(animate=False)

    def on_key(self, event) -> None:
        key = event.key
        if key == "up":
            self._selected = (self._selected - 1) % len(self._summaries)
            self._repaint()
            self._update_preview()
            event.stop()
        elif key == "down":
            self._selected = (self._selected + 1) % len(self._summaries)
            self._repaint()
            self._update_preview()
            event.stop()
        elif key == "enter":
            self.dismiss(self._current_hash())
            event.stop()
        elif key == "escape":
            self.dismiss(None)
            event.stop()


class TurnSelector(ModalScreen["int | None"]):
    DEFAULT_CSS = """
    TurnSelector { align: center middle; background: transparent; }
    TurnSelector #panel {
        width: 90%;
        max-width: 90%;
        height: 90%;
        max-height: 100%;
        background: $bg-base;
        border: none;
        padding: 1 2;
    }
    TurnSelector #title { width: 1fr; color: $fg-dim; padding: 0 1; }
    TurnSelector #turn-list {
        width: 1fr;
        height: 1fr;
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
    TurnSelector .option { width: 1fr; color: $fg-bright; padding: 0 1; }
    TurnSelector .option-active { background: $bg-surface; color: $fg-bright; text-style: bold; }
    TurnSelector #hint { width: 1fr; color: $fg-ghost; padding: 0 1; }
    """

    can_focus = True

    def __init__(self, turns: list) -> None:
        super().__init__()
        self._turns = turns
        self._selected = max(0, len(turns) - 1)

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Static("", id="title")
            with VerticalScroll(id="turn-list"):
                for i in range(len(self._turns)):
                    yield Static("", id=f"turn-{i}", classes="option")
            yield Static("", id="hint")

    def on_mount(self) -> None:
        self._repaint()
        self.focus()

    # ------------------------------------------------------------------ #
    def _repaint(self) -> None:
        self.query_one("#title", Static).update(
            Text(
                f"Restore up to turn {self._selected + 1} of {len(self._turns)}",
                style=theme.FG_DIM,
            )
        )
        for i, turn in enumerate(self._turns):
            opt = self.query_one(f"#turn-{i}", Static)
            marker = "› " if i == self._selected else "  "
            opt.update(f"{marker}{_turn_row(i, turn)}")
            opt.set_class(i == self._selected, "option-active")
        self.query_one(f"#turn-{self._selected}", Static).scroll_visible()
        self.query_one("#hint", Static).update(
            Text("↑↓ turn   enter restore up to here   esc back", style=theme.FG_GHOST)
        )

    def on_key(self, event) -> None:
        key = event.key
        if key == "up":
            self._selected = (self._selected - 1) % len(self._turns)
            self._repaint()
            event.stop()
        elif key == "down":
            self._selected = (self._selected + 1) % len(self._turns)
            self._repaint()
            event.stop()
        elif key == "enter":
            self.dismiss(self._selected)
            event.stop()
        elif key == "escape":
            self.dismiss(None)
            event.stop()
