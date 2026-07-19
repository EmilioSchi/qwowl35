"""Tests for the todo/plan cards."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chat_test_helpers import _ansi, _plain, assert_true  # noqa: E402


def test_parse_todo_result_rows_and_fallbacks() -> None:
    from widgets.chat.renderers.todo import _parse_todo_result

    # The `plan` result: rendered rows, then a blank line and the gate /
    # next-task sentence — the card parses the rows and stops at the blank.
    rendered = (
        "Todo list updated:\n"
        "[>] 1a2: Analyze cal output format\n"
        "[ ] 2b3: Fix header line formatting\n"
        "[x] 3c4: Write automated tests\n"
        "\n"
        "The user approved the plan. Execution will start now."
    )
    rows = _parse_todo_result(rendered)
    assert_true(
        rows
        == [
            ("in_progress", "1a2", "Analyze cal output format"),
            ("pending", "2b3", "Fix header line formatting"),
            ("completed", "3c4", "Write automated tests"),
        ],
        f"three statuses parsed, trailing sentence tolerated: {rows}",
    )
    # Anything unexpected falls back to the plain result box.
    assert_true(
        _parse_todo_result("Error: 'todos' must be a non-empty array.") is None,
        "error text is not a card",
    )
    assert_true(_parse_todo_result("some unrelated output") is None, "non-todo text rejected")
    assert_true(
        _parse_todo_result("Todo list updated:\ngarbage row without mark") is None,
        "malformed row rejects the whole card",
    )
    assert_true(_parse_todo_result("Todo list updated:") is None, "empty list rejected")


def test_todo_card_text_styles_and_progress() -> None:
    from widgets.chat.renderers.todo import _parse_todo_result, _todo_card_text

    rows = _parse_todo_result(
        "Todo list updated:\n"
        "[>] 1a2: Analyze output\n"
        "[ ] 2b3: Fix header\n"
        "[x] 3c4: Write tests"
    )
    card = _todo_card_text(rows)
    plain = _plain(card)
    assert_true("Todo list" in plain and "1/3 done" in plain, f"header with progress: {plain}")
    assert_true("▶ 1a2  Analyze output" in plain, f"in_progress row with glyph: {plain}")
    assert_true("○ 2b3  Fix header" in plain, "pending row with glyph")
    assert_true("✔ 3c4  Write tests" in plain, "completed row with glyph")
    ansi = _ansi(card, width=100)
    # Rich folds strike into a combined SGR (e.g. \x1b[9;38;2;...m).
    assert_true("\x1b[9;" in ansi or "\x1b[9m" in ansi, f"completed content struck through: {ansi!r}")


def test_plan_card_renders_markdown() -> None:
    from widgets.chat.renderers.todo import _plan_card

    plain = _plain(_plan_card("# Approach\n\nFix the *header* line.\n\n- keep tests green"))
    assert_true(plain.startswith("Plan"), f"header first: {plain}")
    assert_true("Approach" in plain and "# Approach" not in plain,
                f"heading rendered, not raw markdown: {plain}")
    assert_true("Fix the header line." in plain, f"emphasis markers stripped: {plain}")
    assert_true("• keep tests green" in plain, f"list bullet rendered: {plain}")


def main() -> None:
    test_parse_todo_result_rows_and_fallbacks()
    test_todo_card_text_styles_and_progress()
    test_plan_card_renders_markdown()
    print("todo renderer tests passed")


if __name__ == "__main__":
    main()
