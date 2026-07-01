"""Tests for prompt input behavior."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.app import App, ComposeResult  # noqa: E402

from history import HistoryConfig, MessageHistory  # noqa: E402
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


class _HistoryHarness(App):
    def __init__(self, history: MessageHistory) -> None:
        super().__init__()
        self._history = history

    def compose(self) -> ComposeResult:
        yield PromptInput(id="p", history=self._history)

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        # Mirror app._handle_submission: persist, then clear the input.
        event.prompt.append_history(event.text)
        event.prompt.clear()


def test_history_up_down_recall() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hist = MessageHistory(HistoryConfig(file=Path(tmp) / "history"))
            hist.append("a")
            hist.append("b")
            app = _HistoryHarness(hist)
            async with app.run_test() as pilot:
                prompt = app.query_one(PromptInput)
                prompt.focus()
                await pilot.press(*"draft")
                await pilot.press("up")
                await pilot.pause()
                assert_equal(prompt.text, "b", "up recalls newest")
                await pilot.press("up")
                await pilot.pause()
                assert_equal(prompt.text, "a", "up again recalls older")
                await pilot.press("down")
                await pilot.pause()
                assert_equal(prompt.text, "b", "down recalls newer")
                await pilot.press("down")
                await pilot.pause()
                assert_equal(prompt.text, "draft", "down past newest restores draft")

    asyncio.run(run())


def test_submit_appends_and_clears_navigation() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hist = MessageHistory(HistoryConfig(file=Path(tmp) / "history"))
            app = _HistoryHarness(hist)
            async with app.run_test() as pilot:
                prompt = app.query_one(PromptInput)
                prompt.focus()
                await pilot.press(*"hello")
                await pilot.press("enter")
                await pilot.pause()
                assert_equal(hist.entries, ["hello"], "submit appends to history")
                # clear() ran on submit -> navigation reset, so next() is a no-op.
                assert_equal(hist.next(), None, "clear reset navigation")

    asyncio.run(run())


def test_up_mid_multiline_moves_cursor_not_recall() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hist = MessageHistory(HistoryConfig(file=Path(tmp) / "history"))
            hist.append("recalled")
            app = _HistoryHarness(hist)
            async with app.run_test() as pilot:
                prompt = app.query_one(PromptInput)
                prompt.focus()
                await pilot.press(*"one")
                await pilot.press("ctrl+j")
                await pilot.press(*"two")
                await pilot.pause()
                # Cursor is on the last line; Up should move it up a line, not recall.
                await pilot.press("up")
                await pilot.pause()
                assert_equal(prompt.text, "one\ntwo", "up mid-multiline does not recall")

    asyncio.run(run())


def main() -> None:
    test_prompt_cursor_does_not_blink()
    test_ctrl_j_newlines_while_enter_submits()
    test_history_up_down_recall()
    test_submit_appends_and_clears_navigation()
    test_up_mid_multiline_moves_cursor_not_recall()
    print("prompt input tests passed")


if __name__ == "__main__":
    main()
