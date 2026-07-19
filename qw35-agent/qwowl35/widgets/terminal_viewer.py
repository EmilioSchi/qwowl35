"""Full-screen viewer for a bash mini terminal window (the □ control).

The whole command and output, uncapped, in a scrollable full-screen panel
dressed in the same window chrome as the inline widget. Esc/q — or the title
bar's `x` — closes it; the footer's `copy` button works here too.
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

import theme
from widgets.chat.primitives import _FullWidthLines, _JustifiedRow, _click_span
from widgets.chat.renderers.shell import _command_rows, _output_rows
from widgets.chat.terminal_chrome import _prompt_text, _term_bg, _window_footer_row


class _ViewerBody(Static):
    """Renders the window; owns the chrome's ``action_win_*`` click targets."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Same as ToolBlock: keep the chrome's own colors instead of
        # Textual's link-color overlay on @click segments.
        self.auto_links = False

    def action_win_close(self) -> None:
        self.screen.dismiss(None)

    def action_win_copy(self) -> None:
        viewer = self.screen
        if not isinstance(viewer, TerminalViewerModal):
            return
        copy = getattr(self.app, "_copy_to_clipboard", None)
        if not callable(copy):
            return
        copy(viewer.copy_payload())
        flash = getattr(self.app, "flash_copied", None)
        if callable(flash):
            flash()


class TerminalViewerModal(ModalScreen[None]):
    # Transparent screen; the panel paints the whole viewport (full screen).
    DEFAULT_CSS = """
    TerminalViewerModal { align: center middle; background: transparent; }
    TerminalViewerModal #panel {
        width: 100%;
        height: 100%;
        background: $bg-base;
        border: none;
        padding: 1 2;
    }
    TerminalViewerModal #term-scroll {
        width: 1fr;
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
    TerminalViewerModal #term { width: 1fr; }
    """

    can_focus = True

    def __init__(
        self,
        *,
        title: str,
        command: str,
        output: str,
        host: str,
        path: str,
        started_at: str,
        tokens: int | None,
        is_error: bool = False,
    ) -> None:
        super().__init__()
        self._title = title
        self._command = command
        self._output = output
        self._host = host
        self._path = path
        self._started_at = started_at
        self._tokens = tokens
        self._is_error = is_error

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            with VerticalScroll(id="term-scroll"):
                yield _ViewerBody(self._window(), id="term")

    def on_mount(self) -> None:
        self.focus()

    def copy_payload(self) -> str:
        output = self._output.strip("\n")
        return f"{self._command}\n{output}" if output else self._command

    def _window(self) -> RenderableType:
        color = theme.ERROR_SOFT if self._is_error else theme.FG_MUTED
        left = Text()
        left.append(" ")
        left.append(self._title, style=color)
        # Only `x` up here: collapse/expand are meaningless in the full view.
        controls = Text()
        controls.append_text(_click_span(" x ", "win_close", theme.ERROR))
        bg = _term_bg()
        rows = _command_rows(
            self._command,
            cursor=False,
            first_prompt=_prompt_text(self._host, self._path),
            bg=bg,
        )
        if self._output:
            rows.extend(_output_rows(self._output, bg=bg))
        return Group(
            _JustifiedRow(left, controls, theme.BG_SURFACE),
            _FullWidthLines(rows, wrap=True),
            _window_footer_row(self._started_at, self._tokens),
        )

    def on_key(self, event) -> None:
        if event.key in ("escape", "q"):
            self.dismiss(None)
            event.stop()
