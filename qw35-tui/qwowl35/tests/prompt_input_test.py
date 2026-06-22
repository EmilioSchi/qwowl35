"""Tests for prompt input behavior."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.app import App, ComposeResult  # noqa: E402

from widgets.prompt_input import PromptInput  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def test_prompt_cursor_does_not_blink() -> None:
    prompt = PromptInput()

    assert_true(prompt.cursor_blink is False, "prompt cursor blink disabled")


class _Harness(App):
    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[str] = []

    def compose(self) -> ComposeResult:
        yield PromptInput(id="p")

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        self.submitted.append(event.text)


def test_ctrl_j_newlines_while_enter_submits() -> None:
    async def run() -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            app.query_one(PromptInput).focus()
            await pilot.press(*"hello")
            await pilot.press("ctrl+j")          # reliable newline on any terminal
            await pilot.press(*"world")
            await pilot.pause()
            prompt = app.query_one(PromptInput)
            assert_equal(prompt.text, "hello\nworld", "ctrl+j inserts a newline")
            assert_equal(app.submitted, [], "ctrl+j does not submit")
            await pilot.press("enter")
            await pilot.pause()
            assert_equal(app.submitted, ["hello\nworld"], "enter submits the multiline text")

    asyncio.run(run())


def main() -> None:
    test_prompt_cursor_does_not_blink()
    test_ctrl_j_newlines_while_enter_submits()
    print("prompt input tests passed")


if __name__ == "__main__":
    main()
