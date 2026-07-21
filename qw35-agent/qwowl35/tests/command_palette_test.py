"""Tests for the slash-command palette: the catalog + filter (pure), the
renderer (pure), a drift guard proving the catalog stays in sync with the pinned
``_dispatch_command``, and the live palette behavior driven headless via Pilot.

Run directly: ``python qwowl35/tests/command_palette_test.py``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import QwowlApp  # noqa: E402
from commands import COMMANDS, filter_commands  # noqa: E402
from history import HistoryConfig, MessageHistory  # noqa: E402
from modes import Mode  # noqa: E402
from widgets.command_palette import render_palette  # noqa: E402
from widgets.prompt_input import PromptInput  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def _names(specs) -> list[str]:
    return [s.name for s in specs]


# --------------------------------------------------------------------------- #
# Catalog + filter (pure)
# --------------------------------------------------------------------------- #
def test_filter_empty_returns_all() -> None:
    assert_equal(_names(filter_commands("")), _names(COMMANDS), "empty query = full catalog in order")
    assert_equal(_names(filter_commands("  ")), _names(COMMANDS), "blank query = full catalog")


def test_filter_prefix_match() -> None:
    assert_equal(_names(filter_commands("cle")), ["/clear"], "cle -> /clear")
    assert_equal(_names(filter_commands("se")), ["/sessions"], "se -> /sessions")
    assert_equal(_names(filter_commands("the")), ["/theme"], "the -> /theme")
    assert_equal(_names(filter_commands("fo")), ["/fonts"], "fo -> /fonts")


def test_filter_prefix_can_match_several() -> None:
    # "cl" prefixes both /clear (name) and /quit (via its /close alias); catalog
    # order is preserved so /clear stays first (the default highlight).
    assert_equal(_names(filter_commands("cl")), ["/clear", "/quit"], "cl -> /clear then /quit")


def test_filter_no_match_returns_empty() -> None:
    assert_equal(filter_commands("ssn"), [], "non-prefix typo matches nothing")
    assert_equal(filter_commands("zzz"), [], "gibberish matches nothing")


def test_filter_matches_aliases() -> None:
    assert_equal(_names(filter_commands("exit")), ["/quit"], "exit alias -> /quit")
    assert_equal(_names(filter_commands("abort")), ["/quit"], "abort alias -> /quit")


def test_filter_ignores_leading_slash() -> None:
    assert_equal(_names(filter_commands("/the")), ["/theme"], "leading slash in query ignored")


# --------------------------------------------------------------------------- #
# Renderer (pure)
# --------------------------------------------------------------------------- #
def test_render_marks_selected_and_lists_all() -> None:
    specs = list(COMMANDS)
    lines = render_palette(specs, 1, "").plain.split("\n")
    assert_equal(len(lines), len(specs), "one row per command")
    assert_true(lines[1].startswith("› "), "selected row carries the marker")
    assert_true(not lines[0].startswith("› "), "unselected row has no marker")
    assert_true(specs[1].name in lines[1], "selected row shows its name")
    assert_true(all(s.description in "\n".join(lines) for s in specs), "descriptions shown")


def test_render_empty_state() -> None:
    assert_equal(render_palette([], 0, "zzz").plain, "no matching command", "empty-state line")


def test_render_aligns_descriptions() -> None:
    specs = list(COMMANDS)
    lines = render_palette(specs, 0, "").plain.split("\n")
    # Blank descriptions (e.g. /quit's " ") match any whitespace at index 0,
    # so only real descriptions can pin the column.
    starts = {
        spec.description: lines[i].index(spec.description)
        for i, spec in enumerate(specs)
        if spec.description.strip()
    }
    assert_equal(len(set(starts.values())), 1, "every description starts at the same column")


# --------------------------------------------------------------------------- #
# Drift guard: every catalog entry still routes through the pinned dispatcher
# --------------------------------------------------------------------------- #
def _dispatch_app():
    app = QwowlApp.__new__(QwowlApp)
    calls: list = []
    app.exit = lambda: calls.append("quit")
    app._clear_conversation = lambda: calls.append("clear")
    app._open_theme_selector = lambda: calls.append("theme")
    app._open_font_selector = lambda: calls.append("fonts")
    app._open_session_selector = lambda: calls.append("sessions")
    app._mode_command = lambda args: calls.append(("mode", args))
    return app, calls


def test_catalog_matches_dispatch() -> None:
    for spec in COMMANDS:
        for name in (spec.name, *spec.aliases):
            app, calls = _dispatch_app()
            assert_true(app._dispatch_command(name), f"{name} routes through dispatch")
            assert_true(calls, f"{name} fired a handler")
    # The one arg-command routes to the mode handler.
    app, calls = _dispatch_app()
    app._dispatch_command("/mode")
    assert_true(calls and calls[0][0] == "mode", "/mode routes to the mode handler")


# --------------------------------------------------------------------------- #
# Live palette behavior (Pilot integration)
# --------------------------------------------------------------------------- #
def _quiet_history(prompt: PromptInput) -> None:
    """Point the prompt at an in-memory history so seeding it never touches the
    user's real history file."""
    prompt._history = MessageHistory(HistoryConfig(file=Path("unused"), enabled=False))


def test_palette_opens_on_slash() -> None:
    async def run() -> None:
        app = QwowlApp()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            prompt.focus()
            await pilot.press("/")
            await pilot.pause()
            assert_true(app.command_palette.display, "palette shown on /")
            assert_true(prompt.palette_open, "palette_open set on the prompt")
            assert_true(app.command_palette.current_spec() is not None, "a command is highlighted")

    asyncio.run(run())


def test_palette_filters_live() -> None:
    async def run() -> None:
        app = QwowlApp()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            prompt.focus()
            await pilot.press("/")
            await pilot.pause()
            all_count = len(app.command_palette.matches)
            await pilot.press(*"the")
            await pilot.pause()
            assert_equal(app.command_palette.current_spec().name, "/theme", "narrowed to /theme")
            assert_true(len(app.command_palette.matches) < all_count, "match list narrowed")

    asyncio.run(run())


def test_arrow_moves_highlight_not_history() -> None:
    async def run() -> None:
        app = QwowlApp()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            _quiet_history(prompt)
            prompt.append_history("earlier message")  # would be recalled if not intercepted
            prompt.focus()
            await pilot.press("/")
            await pilot.pause()
            before = app.command_palette.selected
            await pilot.press("up")  # normally recalls the previous history entry
            await pilot.pause()
            assert_true(app.command_palette.selected != before, "highlight moved")
            assert_equal(prompt.text, "/", "history was NOT recalled while the palette is open")

    asyncio.run(run())


def test_enter_runs_highlighted_command() -> None:
    async def run() -> None:
        app = QwowlApp()
        async with app.run_test() as pilot:
            app.chat.add_user("hello there")
            await pilot.pause()
            assert_true(len(app.chat.children) > 0, "chat seeded with a message")
            prompt = app.query_one(PromptInput)
            prompt.focus()
            await pilot.press(*"/cl")
            await pilot.pause()
            assert_equal(app.command_palette.current_spec().name, "/clear", "/clear highlighted")
            await pilot.press("enter")
            await pilot.pause()
            assert_equal(len(app.chat.children), 0, "enter ran /clear (transcript emptied)")
            assert_equal(prompt.text, "", "input cleared after running")
            assert_true(not app.command_palette.display, "palette closed after running")

    asyncio.run(run())


def test_enter_on_arg_command_completes_not_runs() -> None:
    async def run() -> None:
        app = QwowlApp()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            prompt.focus()
            mode_before = app.mode
            await pilot.press(*"/mo")
            await pilot.pause()
            assert_equal(app.command_palette.current_spec().name, "/mode", "/mode highlighted")
            await pilot.press("enter")
            await pilot.pause()
            assert_equal(prompt.text, "/mode ", "input completed to '/mode ' for the argument")
            assert_equal(app.mode, mode_before, "bare /mode was NOT run")
            assert_true(not app.command_palette.display, "trailing space closed the palette")

    asyncio.run(run())


def test_tab_completes_noarg_and_keeps_open() -> None:
    async def run() -> None:
        app = QwowlApp()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            prompt.focus()
            await pilot.press(*"/cl")
            await pilot.pause()
            await pilot.press("tab")
            await pilot.pause()
            assert_equal(prompt.text, "/clear", "tab completed the text")
            assert_true(app.command_palette.display, "palette stays open after tab")
            assert_true(len(app.chat.children) == 0, "tab did not run anything")

    asyncio.run(run())


def test_escape_dismisses_and_keeps_text() -> None:
    async def run() -> None:
        app = QwowlApp()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            prompt.focus()
            await pilot.press(*"/cl")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert_true(not app.command_palette.display, "palette closed on escape")
            assert_true(not prompt.palette_open, "palette_open cleared on escape")
            assert_equal(prompt.text, "/cl", "typed text kept after escape")

    asyncio.run(run())


def test_typing_space_closes_palette() -> None:
    async def run() -> None:
        app = QwowlApp()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            prompt.focus()
            await pilot.press(*"/clear")
            await pilot.pause()
            assert_true(app.command_palette.display, "open while typing /clear")
            await pilot.press("space")
            await pilot.pause()
            assert_true(not app.command_palette.display, "a space (into args) closes the palette")

    asyncio.run(run())


def test_plain_text_never_opens() -> None:
    async def run() -> None:
        app = QwowlApp()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            prompt.focus()
            await pilot.press(*"hello")
            await pilot.pause()
            assert_true(not app.command_palette.display, "no palette for non-slash text")
            assert_true(not prompt.palette_open, "palette_open stays False")

    asyncio.run(run())


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("command palette tests passed")


if __name__ == "__main__":
    main()
