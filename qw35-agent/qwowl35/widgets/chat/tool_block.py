"""One tool call+result block in the transcript, including the mini terminal
window's click actions."""

from __future__ import annotations

import os
import socket
import time

from textual.widgets import Static

from widgets.chat import terminal_chrome
from widgets.chat.renderers.shell import _split_bash_advisories
from widgets.chat.tool_args import _SHELL_TOOL_NAMES, _command_from_args
from widgets.status_bar import display_path, rough_token_count


def _window_title(block: "ToolBlock") -> str:
    """The mini terminal's window name: not the wire tool name, but a stable
    per-window identity the user can track across the transcript."""
    return f"terminal #{block.term_hash}"


def _copy_payload(block: "ToolBlock") -> str:
    """What the mini terminal's copy button puts on the clipboard: the command
    alone while it still runs, command + output once the result is in
    (advisory blocks the agent appends for the model are stripped)."""
    command = _command_from_args(block.args_buf)
    if block.full_result is None:
        return command
    output, _ = _split_bash_advisories(block.full_result)
    output = output.strip("\n")
    return f"{command}\n{output}" if output else command


def _result_tokens(block: "ToolBlock") -> int | None:
    """Rough chars/4 estimate of what this call cost in prompt tokens (command
    + full result incl. advisories — that is what the model sees). None while
    the call still runs; the server never reports per-call usage."""
    if block.full_result is None:
        return None
    return rough_token_count(_command_from_args(block.args_buf) + block.full_result)


class ToolBlock(Static):
    """One tool call+result: grows as args stream, collapsible result.

    Shell calls render as a mini terminal window whose chrome controls
    (`- □ x`, copy) carry ``@click`` metas that Textual dispatches to the
    ``action_win_*`` methods below.
    """

    def __init__(self, name: str) -> None:
        super().__init__(classes="msg tool-pending")
        # Keep the chrome's own colors: Textual otherwise repaints every
        # @click segment with the theme's link-color, flattening the
        # traffic-light `- □ x` controls (click dispatch reads the style
        # meta directly, so it survives this).
        self.auto_links = False
        # Empty name = raw mode: the call streamed in before its function was
        # recognized; the box shows the raw XML growing until name_tool_call.
        self.tool_name = name
        self.args_buf = ""
        self.full_result: str | None = None
        self.is_error = False
        self.expanded = False
        self.args_dirty = False
        self.reveal = 0  # chars of the command/detail currently typed out
        self.result_ready = False  # result arrived but reveal still typing out
        self.stream_done = False  # final args arrived; reveal fast-forwards
        # ask_user_question card state: shimmer frame while the modals are up,
        # and question index -> answer (None = the modal was dismissed).
        self.anim_frame = 0
        self.ask_answers: dict[int, str | None] = {}
        # Delegator-edit Spawn card: the pre-edit file slice the Editor gets
        # as input, captured from disk once before the editor runs and frozen
        # (the disk content is post-edit by the time the result renders).
        self.spawn_snippet = None  # _SpawnSnippet | None
        self.spawn_snippet_tried = False  # one disk attempt, ever
        # Mini terminal window state (shell calls only).
        self.collapsed = False  # shrunk to the title bar via the `-` control
        self.term_hash = terminal_chrome._next_term_hash()  # names this window: terminal #79b1
        self.started_at = time.strftime("%H:%M:%S")
        # Prompt identity captured at construction: bash runs in this process's
        # cwd, and the cwd could change by the time the row repaints. The host
        # is the qw35 server's address when the app registered one.
        self.prompt_host = terminal_chrome._TERM_HOST or socket.gethostname().split(".")[0]
        self.prompt_path = display_path(os.getcwd(), max_len=30)

    def _is_shell(self) -> bool:
        return self.tool_name in _SHELL_TOOL_NAMES

    def _chat_view(self) -> "ChatView | None":
        # Lazy import: chat_view imports this module.
        from widgets.chat.chat_view import ChatView

        parent = self.parent
        return parent if isinstance(parent, ChatView) else None

    def action_win_collapse(self) -> None:
        if not self._is_shell():
            return
        self.collapsed = not self.collapsed
        chat = self._chat_view()
        if chat is not None:
            chat.repaint_block(self)

    def action_win_expand(self) -> None:
        if not self._is_shell():
            return
        # Lazy import: terminal_viewer imports the chat package's chrome helpers.
        from widgets.terminal_viewer import TerminalViewerModal

        output = ""
        if self.full_result is not None:
            output, _ = _split_bash_advisories(self.full_result)
        try:
            self.app.push_screen(
                TerminalViewerModal(
                    title=_window_title(self),
                    command=_command_from_args(self.args_buf),
                    output=output,
                    host=self.prompt_host,
                    path=self.prompt_path,
                    started_at=self.started_at,
                    tokens=_result_tokens(self),
                    is_error=self.is_error,
                )
            )
        except Exception:
            pass  # no running app (tests) — nothing to show the modal on

    def action_win_close(self) -> None:
        if not self._is_shell():
            return
        chat = self._chat_view()
        if chat is not None:
            chat.close_tool_block(self)
        else:
            self.remove()

    def action_win_copy(self) -> None:
        if not self._is_shell():
            return
        try:
            app = self.app
        except Exception:
            return
        copy = getattr(app, "_copy_to_clipboard", None)
        if not callable(copy):
            return
        copy(_copy_payload(self))
        flash = getattr(app, "flash_copied", None)
        if callable(flash):
            flash()
