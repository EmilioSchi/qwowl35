"""Tests for the plan-approval modal's plan-string rendering and layout.

Run directly: ``python qwowl35/tests/plan_approval_test.py``. Rendering tests
are pure-function level; layout tests mount the modal in a headless app.
"""

from __future__ import annotations

import asyncio
import sys
from io import StringIO
from pathlib import Path

from rich.console import Console
from textual.app import App

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import theme  # noqa: E402
from tools.plan import render_todos  # noqa: E402
from widgets.plan_approval import (  # noqa: E402
    PlanApprovalModal,
    _todo_card,
    split_plan_segments,
)


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def _plain(renderable) -> str:
    console = Console(width=100, record=True, file=StringIO())
    console.print(renderable)
    return console.export_text(styles=False)


def test_split_separates_prose_from_checklist() -> None:
    todos = ["Run cal to capture output", "Create the test file", "Implement cal.py"]
    plan = f"### Approach\nBuild it simply.\n\n{render_todos(todos, 0)}"
    segments = split_plan_segments(plan)
    assert_equal(
        [kind for kind, _ in segments], ["md", "todos"], "prose then checklist"
    )
    assert_true("### Approach" in segments[0][1], "markdown kept verbatim")
    rows = segments[1][1]
    assert_equal(len(rows), 3, "every todo line becomes a row")
    assert_equal(rows[0][0], " ", "pending mark parsed")
    assert_equal(rows[0][1], "1", "position parsed without the hash")
    assert_equal(rows[0][2], todos[0], "content parsed")


def test_split_handles_replan_shape_with_two_checklists() -> None:
    done = ["step one"]
    plan = (
        "Replan — reason: scope grew\n\n"
        f"Already completed:\n{render_todos(done, 1)}\n\n"
        f"Revised remaining steps:\nnew approach\n\n{render_todos(done + ['step two'], 1)}"
    )
    kinds = [kind for kind, _ in split_plan_segments(plan)]
    assert_equal(kinds, ["md", "todos", "md", "todos"], "checklists split twice")


def test_split_without_checklist_is_all_markdown() -> None:
    segments = split_plan_segments("just prose\n\nwith [brackets]: but no marks")
    assert_equal([kind for kind, _ in segments], ["md"], "no false todo matches")


def test_todo_card_renders_marks_positions_and_content() -> None:
    todos = ["capture output", "write tests", "implement"]
    rows = split_plan_segments(render_todos(todos, 1, active=True))[0][1]
    text = _plain(_todo_card(rows))
    assert_true("✔ 1. capture output" in text, "completed row glyph + number")
    assert_true("▶ 2. write tests" in text, "in-progress row glyph")
    assert_true("○ 3. implement" in text, "pending row glyph")
    assert_true("[ ]" not in text and "[x]" not in text, "raw marks not shown")


class _Harness(App):
    def get_theme_variable_defaults(self) -> dict[str, str]:
        # PlanApprovalModal's CSS uses the app's custom theme variables
        # ($bg-base …); provide them so the widget resolves outside QwowlApp.
        return theme.to_css_variables(theme.DEFAULT)


_LONG_PLAN = "\n\n".join(f"## Section {i}\n" + "line\n" * 6 for i in range(30))


def _on_screen(widget, height: int) -> bool:
    region = widget.region
    return region.height > 0 and region.y + region.height <= height


def test_long_plan_fills_tall_terminal() -> None:
    async def run() -> None:
        app = _Harness()
        async with app.run_test(size=(100, 40)) as pilot:
            modal = PlanApprovalModal(_LONG_PLAN)
            app.push_screen(modal)
            await pilot.pause()
            scroll = modal.query_one("#plan-scroll")
            assert_true(
                scroll.size.height >= 25,
                f"scroll uses the terminal height (got {scroll.size.height})",
            )
            assert_true(
                _on_screen(modal.query_one("#opt2"), 40),
                "options stay visible under a long plan",
            )

    asyncio.run(run())


def test_revise_input_visible_on_short_terminal() -> None:
    async def run() -> None:
        app = _Harness()
        async with app.run_test(size=(100, 24)) as pilot:
            modal = PlanApprovalModal(_LONG_PLAN)
            app.push_screen(modal)
            await pilot.pause()
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert_true(
                _on_screen(modal.query_one("#revise-input"), 24),
                "revise input stays visible on a 24-row terminal",
            )
            before = modal.query_one("#plan-scroll").size.height
            await pilot.press("escape")
            await pilot.pause()
            after = modal.query_one("#plan-scroll").size.height
            assert_true(after >= before, "scroll regains its CSS cap after revise")

    asyncio.run(run())


def main() -> None:
    test_split_separates_prose_from_checklist()
    test_split_handles_replan_shape_with_two_checklists()
    test_split_without_checklist_is_all_markdown()
    test_todo_card_renders_marks_positions_and_content()
    test_long_plan_fills_tall_terminal()
    test_revise_input_visible_on_short_terminal()
    print("plan approval rendering tests passed")


if __name__ == "__main__":
    main()
