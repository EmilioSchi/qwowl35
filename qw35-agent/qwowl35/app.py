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
import time
from urllib.parse import urlsplit

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
from approval import ApprovalDecision
from client import Qw35Client
from config import Config, load_config
from modes import USER_MODES, Mode, next_mode
from orchestrator import Orchestrator
from sessions.replay import replay_into_chat
from sessions.restore import list_session_summaries, load_session
from sessions.store import SessionStore
from tools_registry import ToolRegistry
from agent import BudgetDecision
from agents.explorer import ExplorerBudgetContext
from widgets.approval_modal import ApprovalModal
from widgets.chat import BlockquoteFrame, ChatView, set_terminal_host
from widgets.command_palette import CommandPalette
from widgets.explorer_budget_modal import ExplorerBudgetModal
from widgets.mascot import MascotWidget
from widgets.plan_approval import PlanApprovalModal
from widgets.prompt_input import PromptInput
from widgets.question import QuestionModal
from widgets.quit_confirm import QuitConfirm
from widgets.session_selector import SessionSelector, TurnSelector
from widgets.status_bar import StatusBar, display_path
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
        max-height: 10;
        background: $bg-base;
        color: $fg-dim;
        padding: 0 1 1 1;
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
        # Shift+Tab arrives as the distinct backtab sequence (CSI Z) on
        # virtually every terminal; /mode covers the same cycle for the rest.
        # priority=True so it beats the focused prompt's own key handling.
        Binding("shift+tab", "cycle_mode", "Cycle mode", show=False, priority=True),
    ]

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config = config or load_config()
        try:  # LSP validation is optional; a broken install must not stop the TUI.
            from tools.lsp import configure as _configure_lsp

            _configure_lsp(self.config.lsp)
        except Exception:  # noqa: BLE001 - degrades to tree-sitter checks
            pass
        self._session_store = SessionStore()
        self.client = Qw35Client(
            self.config.base_url,
            timeout=self.config.request_timeout,
        )
        self.registry = ToolRegistry(
            approval=self._confirm_bash,
            restricted_bash=self.config.restricted_bash,
            compress=self.config.compress,
        )
        self.mascot = MascotWidget(info=display_path(os.getcwd()))
        self.status = StatusBar(
            base_url=self.config.base_url,
            think=self.config.think,
            effort=self.config.reasoning_effort,
            max_tokens=self.config.max_tokens,
        )
        # The mini terminal prompt names the qw35 server it talks to.
        set_terminal_host(urlsplit(self.config.base_url).hostname)
        self.chat = ChatView()
        self.queue_panel = Static("", id="queue-panel")
        # The slash-command palette (shown above the prompt when text starts with
        # "/"). ``_palette_suppressed`` latches it closed after Escape until the
        # text no longer starts with "/", so the user can still send a literal
        # "/word" to the model.
        self.command_palette = CommandPalette()
        self._palette_suppressed = False
        self._queued_messages: list[str] = []
        self._queue_last_at: str | None = None  # stamp of the latest enqueue
        # One agent for every mode: the orchestrator dispatches on the mode
        # the turn was sent under (NORMAL executor, PLAN pipeline, WEB, CHAT).
        # self.registry stays for widgets.
        self.agent = Orchestrator(
            self.client,
            self.config,
            self,
            approval=self._confirm_bash,
            restricted_bash=self.config.restricted_bash,
            session_store=self._session_store,
            question_callback=self._ask_user_questions,
            plan_callback=self._approve_plan,
            explorer_budget_callback=self._choose_explorer_budget,
        )
        # Theme catalog (built-in default + bundled opencode themes). The last
        # committed choice is persisted across launches (env override, then saved
        # file, then the built-in default); see ``theme.preference``.
        self._theme_catalog = theme_registry.load_catalog()
        self._theme_name, self._theme_mode = theme_preference.load(
            self._theme_catalog, default_name=theme_registry.BUILTIN_NAME
        )
        self._mascot_timer = None
        # Every new conversation starts in NORMAL; the mode is locked while a
        # turn runs (it can only change before sending a prompt).
        self.mode: Mode = Mode.NORMAL
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
            yield self.command_palette
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
            (
                ("Enter", "send/queue"),
                ("Ctrl+j", "newline"),
                ("Shift+Tab", "cycle mode"),
            )
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
        self.status.set_mode(self.mode)
        self._reschedule_mascot()
        self.set_state(mascot.State.WAITING)
        self.query_one(PromptInput).focus()
        self._monitor_health()
        await asyncio.to_thread(self._session_store.cleanup)

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
        try:  # stop any language servers; their __exit__ can block briefly.
            from tools.lsp import shutdown_all as _shutdown_lsp

            await asyncio.to_thread(_shutdown_lsp)
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass
        await asyncio.to_thread(self._session_store.cleanup)

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
        self,
        percent: float,
        processed: int | None = None,
        total: int | None = None,
        session_ctx: int | None = None,
    ) -> None:
        # Real prefill % reported by the server (qw35_prefill SSE side-channel),
        # with the serving session's live (growable) context ceiling.
        self.status.update_prefill(total, session_ctx)
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
    # Command palette (opens on a leading "/", keeps the prompt focused)
    # ------------------------------------------------------------------ #
    @on(PromptInput.Changed)
    def _on_prompt_changed(self, event: PromptInput.Changed) -> None:
        # TextArea.Changed also fires on programmatic edits (clear, paste,
        # history recall, our own accept), so re-derive the palette purely from
        # the current text rather than assuming a keystroke.
        if event.text_area is self.query_one(PromptInput):
            self._sync_palette()

    def _palette_query(self, text: str) -> str | None:
        """The text after a leading ``/`` while it is still a single token, or
        ``None`` when the palette should be closed. The Escape latch keeps it
        closed until the text stops starting with ``/``."""
        if self._palette_suppressed:
            if not text.startswith("/"):
                self._palette_suppressed = False
            return None
        if not text.startswith("/") or " " in text or "\n" in text:
            return None
        return text[1:]

    def _sync_palette(self) -> None:
        prompt = self.query_one(PromptInput)
        query = self._palette_query(prompt.text)
        if query is None:
            self._close_palette(prompt)
            return
        self.command_palette.update_query(query)
        self.command_palette.display = True
        prompt.palette_open = True

    def _close_palette(self, prompt: PromptInput | None = None) -> None:
        (prompt or self.query_one(PromptInput)).palette_open = False
        self.command_palette.display = False

    def _suppress_palette(self) -> None:
        self._palette_suppressed = True
        self._close_palette()

    @on(PromptInput.PaletteNavigate)
    def _on_palette_navigate(self, event: PromptInput.PaletteNavigate) -> None:
        self.command_palette.move(event.delta)

    @on(PromptInput.PaletteDismiss)
    def _on_palette_dismiss(self, event: PromptInput.PaletteDismiss) -> None:
        self._suppress_palette()

    @on(PromptInput.PaletteAccept)
    def _on_palette_accept(self, event: PromptInput.PaletteAccept) -> None:
        spec = self.command_palette.current_spec()
        if spec is None:  # empty state: nothing to accept
            return
        prompt = self.query_one(PromptInput)
        if spec.takes_args:
            # Never run a bare arg-command (e.g. /mode would silently cycle);
            # fill "/mode " so the user types the value (the space closes us).
            prompt.text = spec.name + " "
            prompt.move_cursor(prompt.document.end)
        elif event.complete_only:  # Tab: complete the text, keep the palette open
            prompt.text = spec.name
            prompt.move_cursor(prompt.document.end)
        else:  # Enter: run it via the pinned submit/dispatch path
            self._close_palette(prompt)
            self._handle_submission(prompt, spec.name)

    # ------------------------------------------------------------------ #
    # Local commands (typed into the prompt, matched exactly)
    # ------------------------------------------------------------------ #
    def _dispatch_command(self, text: str) -> bool:
        """Run a local ``/``-command. Returns True if one was handled."""
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
        if text == "/sessions":
            self._open_session_selector()
            return True
        parts = text.split()
        if parts and parts[0] == "/mode":
            self._mode_command(parts[1:])
            return True
        return False

    def _mode_command(self, args: list[str]) -> None:
        """``/mode`` cycles; ``/mode <name>`` selects directly. Locked while busy."""
        if len(args) > 1:
            self.set_warning("usage: /mode [normal|plan|web|chat]")
            return
        if args:
            wanted = args[0].lower()
            target = next((m for m in USER_MODES if m.value == wanted), None)
            if target is None:
                names = "|".join(m.value for m in USER_MODES)
                self.set_warning(f"unknown mode {args[0]!r} — one of: {names}")
                return
        else:
            target = next_mode(self.mode)
        self._select_mode(target)

    def _select_mode(self, target: Mode) -> None:
        if self._busy:
            self.set_warning("mode is locked while a turn runs")
            return
        self.mode = target
        self.status.set_mode(self.mode)

    def _clear_conversation(self) -> None:
        """``/clear``: reset the transcript and agent history (keep system prompt)."""
        self.agent.clear()
        self.chat.clear()
        # A cleared conversation is a NEW session: rotate the store so the
        # conversation just cleared becomes a restorable past session.
        self._session_store.rotate()
        self._queued_messages.clear()
        self._render_queue()
        # A cleared transcript is a new conversation: back to NORMAL.
        self.mode = Mode.NORMAL
        self.status.set_mode(self.mode)
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

    @work(exclusive=True, group="sessions")
    async def _open_session_selector(self) -> None:
        """``/sessions``: pick a past session, then the turn to restore up
        to (escape in the turn list steps back to the session list)."""
        if self._busy:
            self.set_warning("sessions are locked while a turn runs")
            return
        while True:
            summaries = await asyncio.to_thread(
                list_session_summaries,
                self._session_store.root,
                self._session_store.session_hash,
            )
            if not summaries:
                self.set_info("no past sessions yet — this one is being recorded")
                break
            chosen = await self.push_screen_wait(
                SessionSelector(
                    summaries,
                    load_turns=lambda h: load_session(self._session_store.root, h),
                )
            )
            if chosen is None:
                break
            turns = await asyncio.to_thread(
                load_session, self._session_store.root, chosen
            )
            if not turns:
                self.set_warning("session has no restorable turns")
                continue
            cut = await self.push_screen_wait(TurnSelector(turns))
            if cut is None:
                continue
            selected = turns[: cut + 1]
            if len(selected) < len(turns):
                # An earlier cut-off FORKS: the prefix turn dirs are copied
                # into a fresh self-contained session (stamped with its
                # origin) and the store attaches to it — the source session
                # is never modified, its tail stays restorable.
                await asyncio.to_thread(
                    self._session_store.fork,
                    chosen,
                    [turn.turn_dir for turn in selected],
                )
            else:
                self._session_store.attach(chosen)
            self._restore_session(selected)
            break
        self.query_one(PromptInput).focus()

    def _restore_session(self, turns: list) -> None:
        """Rehydrate a restored session prefix into the app: replay the
        display and rebuild the agent's turn log and CHAT conversation. The
        store was already pointed at the right session (attach or fork) by
        the caller; the server re-primes its KV cache on the first request
        (a normal full prefill)."""
        self.chat.clear()
        replay_into_chat(self.chat, turns)
        self.agent.restore(turns)
        self._queued_messages.clear()
        self._render_queue()
        # Resume in the mode the conversation ended with (NORMAL/PLAN/WEB/
        # CHAT), so the next prompt dispatches the way the session was being
        # driven. Unknown or missing modes fall back to NORMAL.
        last_mode = turns[-1].mode if turns else ""
        self.mode = next(
            (m for m in USER_MODES if m.value == last_mode), Mode.NORMAL
        )
        self.status.set_mode(self.mode)
        self.set_info("session restored")

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
        if self.command_palette.display:
            self.command_palette.repaint()
        self._render_queue()

    def _enqueue_message(self, text: str) -> None:
        self._queued_messages.append(text)
        self._queue_last_at = time.strftime("%H:%M:%S")
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
            self._queue_last_at = None
            return

        visible = self._queued_messages[:QUEUE_DISPLAY_LIMIT]
        body = Text()
        for i, message in enumerate(visible, start=1):
            if i > 1:
                body.append("\n")
            body.append(f"{i}. {_preview_queued_message(message)}", style=theme.FG_DIM)
        hidden = len(self._queued_messages) - len(visible)
        if hidden > 0:
            body.append("\n... more", style=theme.FG_FAINT)
        frame = BlockquoteFrame(
            body,
            title="Incoming Message",
            timestamp=self._queue_last_at,
            edge="ACCENT",
        )
        self.queue_panel.update(frame)
        self.queue_panel.display = True

    @work(exclusive=True, group="turn")
    async def _run_turn(self, text: str) -> None:
        self._set_busy(True)
        try:
            ok = await self.agent.run_turn(text, self.mode)
        finally:
            self._set_busy(False)
            # Restore the user's selection after any transient display modes
            # (VISUAL/INSERT/...) the turn pushed.
            self.status.set_mode(self.mode)
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

    def action_cycle_mode(self) -> None:
        """Shift+Tab: next user-selectable mode (NORMAL → PLAN → WEB → CHAT)."""
        self._select_mode(next_mode(self.mode))

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

    # ------------------------------------------------------------------ #
    # Smart-mode gates (called from the orchestrator via PlanTools)
    # ------------------------------------------------------------------ #
    async def _ask_user_questions(self, questions: list[dict]) -> dict:
        """One modal per planner question; a dismissed modal skips its question."""
        answers: dict = {}
        for index, question in enumerate(questions):
            answer = await self.push_screen_wait(QuestionModal(question))
            if answer is not None:
                answers[str(question.get("question", ""))] = answer
            # Grow the ask card's tree in the transcript (None = skipped).
            self.chat.note_question_answer(index, answer)
        return answers

    async def _approve_plan(self, plan: str):
        return await self.push_screen_wait(PlanApprovalModal(plan))

    async def _choose_explorer_budget(self, context: ExplorerBudgetContext) -> BudgetDecision:
        """The explorer's round budget ran out mid-run: ask what to do
        instead of the silent cutoff (see orchestrator._run_explorer)."""
        self.set_state(mascot.State.ASK)
        return await self.push_screen_wait(ExplorerBudgetModal(context))

    def set_mode(self, mode: Mode) -> None:
        """Show a (possibly transient) mode in the footer status bar.

        Called by the orchestrator during a turn (VISUAL while the explorer
        runs, INSERT while the editor runs, ...); the user's own selection is
        restored when the turn ends."""
        self.status.set_mode(mode)
