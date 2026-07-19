"""Shared helpers for the chat widget tests (not collected by pytest)."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from widgets.chat import ToolBlock  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def _plain(renderable) -> str:
    console = Console(width=100, record=True, file=StringIO())
    console.print(renderable)
    return console.export_text(styles=False)


def _ansi(renderable, width: int = 20) -> str:
    console = Console(
        width=width,
        record=True,
        force_terminal=True,
        color_system="truecolor",
        file=StringIO(),
    )
    console.print(renderable)
    return console.export_text(styles=True)


def _textual_content_height(renderable, width: int = 80) -> int:
    """Mimic Textual's ``RichVisual.get_height`` (textual/visual.py): the height it
    assigns an auto-height widget by COUNTING '\\n' in the rendered segments, not by
    counting logical lines. It then crops the widget's strips to that height. A box
    whose final row lacked a trailing newline measured one row short, so Textual
    cropped the last line — the bash output, a tool-call arg preview, or the final
    line of the approval command — and painted a blank in its place.
    """
    console = Console(width=width, file=StringIO())
    options = console.options.update_width(width).update(highlight=False)
    return sum(seg.text.count("\n") for seg in console.render(renderable, options))


def _fg_triplet(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"38;2;{int(h[0:2], 16)};{int(h[2:4], 16)};{int(h[4:6], 16)}"


def _shell_result_block(command: str = "date", output: str = "out\n") -> ToolBlock:
    block = ToolBlock("bash")
    block.args_buf = f'{{"command":"{command}"}}'
    block.full_result = output
    return block
