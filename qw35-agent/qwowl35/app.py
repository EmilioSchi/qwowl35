"""The qwowl35 Textual application.

Layout: the owl mascot is docked top-left and never moves; the chat log fills
the middle and scrolls; the prompt input is docked at the bottom. The mascot
animates on a timer whose period tracks the current state's animation interval.
The agent loop runs as a Textual worker so streaming and tool execution never
block the UI.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

import mascot
import mascot_states
import theme
from theme import preference as theme_preference
from theme import registry as theme_registry
from agent import Agent
from approval import ApprovalDecision
from client import Qw35Client
from config import Config, load_config
from tools_registry import ToolRegistry
from widgets.approval import ApprovalModal
from widgets.chat_log import ChatView
from widgets.mascot_widget import MascotWidget
from widgets.prompt_input import PromptInput
from widgets.quit_confirm import QuitConfirm
from widgets.status_panel import StatusBar, display_path
from widgets.theme_selector import ThemeSelector

QUEUE_PREVIEW_CHARS = 120
QUEUE_DISPLAY_LIMIT = 4
HEALTH_RETRY_SECONDS = 10


def _preview_queued_message(text: str, limit: int = QUEUE_PREVIEW_CHARS) -> str:
    preview = " ".join(text.split())
    if len(preview) <= limit:
        return preview
    return preview[: max(0, limit - 3)].rstrip() + "..."


def format_queued_user_batch(messages: list[str]) -> str:
    return "\n\n".join(messages)


class QwowlApp(App):
    TITLE = "qwowl35"
    # One uniform background across every region; only message boxes set their own.
    # Colors come from Textual theme variables ($bg-base …) so the whole app
    # restyles live when ``self.theme`` changes; see the ``theme`` package.
    CSS = """
    Screen { background: $bg-base; }
    MascotWidget { background: $bg-base; }
    #top-row {
        height: 4;
        width: 1fr;
        background: $bg-base;
    }
    ChatView { background: $bg-base; }
    /* The input dock contains a compact prompt bar plus the footer. */
    #prompt-dock {
        dock: bottom;
        height: auto;
        max-height: 40%;
        width: 1fr;
        background: $bg-base;
        border: none;
        padding: 0;
    }
    #prompt-row {
        width: 1fr;
        height: auto;
        background: $bg-surface;
        padding: 0;
    }
    #queue-panel {
        display: none;
        width: 1fr;
        max-height: 8;
        background: $bg-base;
        color: $fg-dim;
        padding: 0 1 1 3;
    }
    #prompt-mark {
        width: 3;
        height: 3;
        color: $accent;
        background: $bg-surface;
        content-align: center middle;
        text-style: bold;
    }
    #footer { color: $fg-ghost; background: $bg-base; padding: 0 1; height: auto; }
    """
    BINDINGS = [
        # Ctrl+C opens a confirmation modal instead of quitting outright — it's an
        # easy key to hit by accident. priority=True so it fires even while the
        # prompt Input is focused. The confirm's own dispatch calls exit().
        Binding("ctrl+c", "request_quit", "Quit", priority=True),
        Binding("ctrl+o", "toggle_tools", "Expand tools", show=False, priority=True),
    ]

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config = config or load_config()
        self.client = Qw35Client(self.config.base_url, timeout=self.config.request_timeout)
        self.registry = ToolRegistry(
            approval=self._confirm_bash,
            restricted_bash=self.config.restricted_bash,
        )
        self.mascot = MascotWidget(info=display_path(os.getcwd()))
        self.status = StatusBar(
            base_url=self.config.base_url,
            think=self.config.think,
            effort=self.config.reasoning_effort,
            max_tokens=self.config.max_tokens,
        )
        self.chat = ChatView()
        self.queue_panel = Static("", id="queue-panel")
        self._queued_messages: list[str] = []
        self.agent = Agent(self.client, self.registry, self.config, self)
        # Theme catalog (built-in default + bundled opencode themes). The last
        # committed choice is persisted across launches (env override, then saved
        # file, then the built-in default); see ``theme.preference``.
        self._theme_catalog = theme_registry.load_catalog()
        self._theme_name, self._theme_mode = theme_preference.load(
            self._theme_catalog, default_name=theme_registry.BUILTIN_NAME
        )
        self._mascot_timer = None
        self._busy = False
        self._quit_pending = False
        self._copied_revert = None
        self._copied_prev = None
        self._notice_revert = None

    def get_theme_variable_defaults(self) -> dict[str, str]:
        """Fallback values for the app's custom CSS variables ($bg-base …).

        Guarantees they always resolve — before a theme is applied, or if a theme
        omits a token — so the CSS never fails to parse.
        """
        return theme.to_css_variables(theme.DEFAULT)

    # ------------------------------------------------------------------ #
    # Composition / lifecycle
    # ------------------------------------------------------------------ #
    def compose(self) -> ComposeResult:
        with Horizontal(id="top-row"):
            yield self.mascot
        yield self.chat
        with Vertical(id="prompt-dock"):
            yield self.queue_panel
            with Horizontal(id="prompt-row"):
                yield Static(">", id="prompt-mark")
                yield PromptInput(id="prompt")
            yield Static(self._footer_text(), id="footer")
            yield self.status

    @staticmethod
    def _footer_text() -> Text:
        """Footer hints: command keys in bright white, their meaning in gray."""
        key = theme.FG_BRIGHT
        meaning = theme.FG_GHOST
        sep = Text(" · ", style=meaning)
        out = Text()
        for idx, (k, m) in enumerate(
            (("Enter", "send/queue"), ("Ctrl+o", "expand tools"), ("Ctrl+j", "newline"))
        ):
            if idx:
                out.append_text(sep)
            out.append(k, style=key)
            out.append(" ")
            out.append(m, style=meaning)
        return out

    async def on_mount(self) -> None:
        self._theme_catalog.register_all(self)
        theme.set_active(self._theme_catalog.palette(self._theme_name, self._theme_mode))
        self.theme = self._theme_catalog.textual_name(self._theme_name, self._theme_mode)
        self._reschedule_mascot()
        self.set_state(mascot.State.WAITING)
        self.query_one(PromptInput).focus()
        self._monitor_health()

    @work(exclusive=True, group="health")
    async def _monitor_health(self) -> None:
        """Poll the server for the app's whole lifetime, every HEALTH_RETRY_SECONDS.

        Polling never stops once connected — that is what catches a server that
        drops mid-session (the previous version returned on first success and
        went blind to later disconnects). The footer status dot always reflects
        the latest probe; the mascot only reacts on a connectivity *edge*, and
        only while idle: during a turn the generation animation owns the mascot
        and a mid-flight drop is surfaced by the stream's own error handler. An
        edge seen while busy is left pending (``ready`` is not advanced), so the
        next idle poll applies it.
        """
        ready: bool | None = None  # last *reflected* state; None until first probe
        while True:
            ok = await self._check_health()
            if ok != ready and not self._busy:
                if ok:
                    # First successful probe settles into the idle owl; a later
                    # recovery earns a brief "connected" flash instead.
                    if ready is None:
                        self.set_state(mascot.State.WAITING)
                    else:
                        self.set_info("connected")
                else:
                    self.set_error("offline", "retrying")
                ready = ok
            await asyncio.sleep(HEALTH_RETRY_SECONDS)

    async def _check_health(self) -> bool:
        """Probe /health (and /props); update status. Returns True when ready."""
        health = None
        try:
            health = await self.client.health()
        except Exception:
            pass
        self.status.update_health(health)
        if health is not None:
            try:
                self.status.update_props(await self.client.props())
            except Exception:
                pass
        return bool(health and health.get("decoder_ready"))

    async def on_unmount(self) -> None:
        await self.client.aclose()

    # ------------------------------------------------------------------ #
    # Mascot state machine
    # ------------------------------------------------------------------ #
    def _reschedule_mascot(self) -> None:
        if self._mascot_timer is not None:
            self._mascot_timer.stop()
        self._mascot_timer = self.set_interval(self.mascot.interval, self.mascot.advance)

    def _apply_animation(self, animation: mascot.Animation) -> None:
        self.mascot.set_animation(animation)
        self._reschedule_mascot()

    def set_state(self, state: mascot.State) -> None:
        animation = mascot.ANIMATIONS.get(state.value)
        if animation is not None:
            self._apply_animation(animation)

    def begin_generation(self) -> None:
        self.status.reset_generation()

    def set_prefill(
        self, percent: float, processed: int | None = None, total: int | None = None
    ) -> None:
        # Real prefill % reported by the server (qw35_prefill SSE side-channel).
        self.status.update_prefill(total)
        self._apply_animation(mascot_states.prefill(percent))

    def add_reasoning_delta(self, text: str) -> None:
        self.status.update_reasoning(text)

    def set_usage(self, usage: dict, timings: dict | None = None) -> None:
        self.status.update_usage(usage)
        self.status.update_timings(timings)

    def set_error(self, code: str, message: str) -> None:
        self._apply_animation(mascot.error(code, message))

    def set_warning(self, message: str) -> None:
        """Surface a short application warning on the owl (not in the chat).

        The next generation state (prefill/inference) overwrites it on its own,
        so a transient warning shows briefly without lingering."""
        self._apply_animation(mascot.warn(message))

    def set_info(self, message: str) -> None:
        """Flash a short application status on the owl, then settle back to the
        idle WAITING state after ~1.5s if nothing else has taken over."""
        if self._notice_revert is not None:
            self._notice_revert.stop()
        self._apply_animation(mascot.info(message))
        self._notice_revert = self.set_timer(1.5, self._end_notice)

    def _end_notice(self) -> None:
        self._notice_revert = None
        if not self._busy:
            self.set_state(mascot.State.WAITING)

    def flash_copied(self) -> None:
        """Briefly show the green "✄ copied" owl, then revert after ~1s."""
        if self._copied_revert is not None:
            self._copied_revert.stop()
            self._copied_revert = None
        else:
            self._copied_prev = self.mascot.animation
        self._apply_animation(mascot.COPIED)
        self._copied_revert = self.set_timer(1.0, self._end_copied)

    def _end_copied(self) -> None:
        self._copied_revert = None
        if not self._busy:
            self.set_state(mascot.State.WAITING)
        elif self._copied_prev is not None:
            self._apply_animation(self._copied_prev)
        self._copied_prev = None

    # ------------------------------------------------------------------ #
    # Clipboard / text selection
    # ------------------------------------------------------------------ #
    def on_text_selected(self, event: events.TextSelected) -> None:
        text = self.screen.get_selected_text()
        if not text or not text.strip():
            return
        self._copy_to_clipboard(text)
        self.flash_copied()

    def _copy_to_clipboard(self, text: str) -> None:
        """Copy via OSC 52 (SSH / iTerm2) plus the native tool (Terminal.app,
        Linux, Windows). Both paths are guarded so a copy never crashes the UI."""
        try:
            self.copy_to_clipboard(text)
        except Exception:
            pass
        cmd = (
            ["pbcopy"] if sys.platform == "darwin"
            else ["clip"] if sys.platform == "win32"
            else ["xclip", "-selection", "clipboard"]
        )
        try:
            subprocess.run(
                cmd,
                input=text.encode(),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Input / turn handling
    # ------------------------------------------------------------------ #
    @on(PromptInput.Submitted)
    def _on_submit(self, event: PromptInput.Submitted) -> None:
        self._handle_submission(event.prompt, event.text)

    def _handle_submission(self, prompt: PromptInput, submitted_text: str) -> None:
        text = submitted_text.strip()
        if not text:
            return
        # Exact-match local commands are handled here and never reach the model
        # (nor the queue). Everything else is a normal turn/queued message.
        if self._dispatch_command(text):
            prompt.clear()
            return
        prompt.append_history(submitted_text)
        prompt.clear()
        if self._busy:
            self._enqueue_message(text)
            return
        self._busy = True
        self._run_turn(text)

    # ------------------------------------------------------------------ #
    # Local commands (typed into the prompt, matched exactly)
    # ------------------------------------------------------------------ #
    def _dispatch_command(self, text: str) -> bool:
        """Run an exact-match ``/``-command. Returns True if one was handled."""
        if text in ("/quit", "/exit", "/abort", "/close"):
            # ``App.action_quit`` is a coroutine (a bare call would never run);
            # ``exit`` is the synchronous shutdown entry point.
            self.exit()
            return True
        if text == "/clear":
            self._clear_conversation()
            return True
        if text == "/theme":
            self._open_theme_selector()
            return True
        return False

    def _clear_conversation(self) -> None:
        """``/clear``: reset the transcript and agent history (keep system prompt)."""
        self.agent.clear()
        self.chat.clear()
        self._queued_messages.clear()
        self._render_queue()
        self.set_info("cleared")

    @work(exclusive=True, group="theme")
    async def _open_theme_selector(self) -> None:
        """``/theme``: open the picker (live preview), commit or revert on close."""
        result = await self.push_screen_wait(
            ThemeSelector(self._theme_catalog.names, self._theme_name, self._theme_mode)
        )
        if result is not None:
            name, mode = result
            self.apply_theme_preview(name, mode)  # commit
            # apply_theme_preview snaps mode to an available one; persist that.
            theme_preference.save(self._theme_name, self._theme_mode)
        self.query_one(PromptInput).focus()

    def apply_theme_preview(self, name: str, mode: str) -> None:
        """Apply a theme+mode live: CSS restyle plus a refresh of Rich-drawn chrome."""
        available = self._theme_catalog.available_modes(name)
        if mode not in available:
            mode = available[0] if available else "dark"
        self._theme_name = name
        self._theme_mode = mode
        theme.set_active(self._theme_catalog.palette(name, mode))
        self.theme = self._theme_catalog.textual_name(name, mode)
        self._refresh_themed_widgets()

    def _refresh_themed_widgets(self) -> None:
        """Re-render widgets whose Rich content is cached; CSS-styled ones auto-update."""
        try:
            self.query_one("#footer", Static).update(self._footer_text())
        except Exception:
            pass
        self.status.refresh()
        self.mascot.refresh()
        self._render_queue()

    def _enqueue_message(self, text: str) -> None:
        self._queued_messages.append(text)
        self._render_queue()

    def pop_queued_user_batch(self) -> str | None:
        if not self._queued_messages:
            return None
        queued = list(self._queued_messages)
        self._queued_messages.clear()
        self._render_queue()
        return format_queued_user_batch(queued)

    def _render_queue(self) -> None:
        if not self._queued_messages:
            self.queue_panel.display = False
            self.queue_panel.update("")
            return

        lines = ["Queue"]
        visible = self._queued_messages[:QUEUE_DISPLAY_LIMIT]
        for message in visible:
            lines.append(_preview_queued_message(message))
        hidden = len(self._queued_messages) - len(visible)
        if hidden > 0:
            lines.append("... more")
        self.queue_panel.update(Text("\n".join(lines), style=theme.FG_DIM))
        self.queue_panel.display = True

    @work(exclusive=True, group="turn")
    async def _run_turn(self, text: str) -> None:
        self._set_busy(True)
        try:
            ok = await self.agent.run_turn(text)
        finally:
            self._set_busy(False)
        if ok:
            await asyncio.sleep(1.1)  # let the OK check linger
            if not self._busy:
                self.set_state(mascot.State.WAITING)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        prompt = self.query_one(PromptInput)
        prompt.disabled = False
        prompt.focus()

    def action_toggle_tools(self) -> None:
        self.chat.toggle_tools()

    # ------------------------------------------------------------------ #
    # Quit confirmation (Ctrl+C)
    # ------------------------------------------------------------------ #
    def action_request_quit(self) -> None:
        """Ctrl+C: ask before quitting. Ignore repeats while the modal is open."""
        if self._quit_pending:
            return
        self._prompt_quit()

    @work(group="quit-confirm")
    async def _prompt_quit(self) -> None:
        self._quit_pending = True
        try:
            confirmed = await self.push_screen_wait(QuitConfirm())
        finally:
            self._quit_pending = False
        if confirmed:
            self.exit()
        else:
            self.query_one(PromptInput).focus()

    # ------------------------------------------------------------------ #
    # Bash approval (called from the agent worker via the registry)
    # ------------------------------------------------------------------ #
    async def _confirm_bash(
        self, command: str, warnings: list[str], allowlist_info: str
    ) -> ApprovalDecision:
        return await self.push_screen_wait(ApprovalModal(command, warnings, allowlist_info))
