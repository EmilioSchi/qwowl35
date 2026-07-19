"""Tests for the ask_user_question modal, including multiSelect."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.app import App  # noqa: E402

import theme  # noqa: E402
from widgets.question import QuestionModal  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def question(multi: bool = False) -> dict:
    q = {
        "question": "Which one?",
        "header": "Pick",
        "options": [
            {"label": "A", "description": "first"},
            {"label": "B", "description": "second"},
        ],
    }
    if multi:
        q["multiSelect"] = True
    return q


class _Harness(App):
    def get_theme_variable_defaults(self) -> dict[str, str]:
        # QuestionModal's CSS uses the app's custom theme variables
        # ($bg-base …); provide them so the widget resolves outside QwowlApp.
        return theme.to_css_variables(theme.DEFAULT)


def _run_modal(q: dict, keys: list[str]) -> tuple[str | None, bool]:
    """Drives the modal with keys; returns (answer, dismissed)."""
    answers: list[str | None] = []

    async def run() -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            app.push_screen(QuestionModal(q), callback=answers.append)
            await pilot.pause()
            for key in keys:
                await pilot.press(key)
            await pilot.pause()

    asyncio.run(run())
    return (answers[0] if answers else None, bool(answers))


def test_single_select_digit_confirms() -> None:
    answer, dismissed = _run_modal(question(), ["2"])
    assert_true(dismissed, "digit confirms immediately in single-select")
    assert_equal(answer, "B", "second option chosen")


def test_single_select_escape_dismisses_without_answer() -> None:
    answer, dismissed = _run_modal(question(), ["escape"])
    assert_true(dismissed, "escape closes the modal")
    assert_equal(answer, None, "no answer on escape")


def test_multi_select_toggles_then_enter_joins() -> None:
    # Space toggles the highlighted option, digits toggle by number.
    answer, dismissed = _run_modal(question(multi=True), ["space", "2", "enter"])
    assert_true(dismissed, "enter confirms the toggled set")
    assert_equal(answer, "A, B", "selected labels joined with ', '")


def test_multi_select_digit_retoggle_and_empty_enter_ignored() -> None:
    # Toggling the same option off leaves nothing selected; enter is a no-op,
    # so only escape closes the modal (without an answer).
    answer, dismissed = _run_modal(
        question(multi=True), ["1", "1", "enter", "escape"]
    )
    assert_true(dismissed, "escape still dismisses")
    assert_equal(answer, None, "empty selection cannot be confirmed")


def test_multi_select_other_appends_free_text() -> None:
    answer, dismissed = _run_modal(
        question(multi=True), ["1", "tab", *"custom", "enter"]
    )
    assert_true(dismissed, "other submission confirms")
    assert_equal(answer, "A, custom", "free text appended to toggled labels")


def main() -> None:
    test_single_select_digit_confirms()
    test_single_select_escape_dismisses_without_answer()
    test_multi_select_toggles_then_enter_joins()
    test_multi_select_digit_retoggle_and_empty_enter_ignored()
    test_multi_select_other_appends_free_text()
    print("question modal tests passed")


if __name__ == "__main__":
    main()
