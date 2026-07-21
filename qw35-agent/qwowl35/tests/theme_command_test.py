"""Tests for local ``/``-commands: dispatch, ``/clear``, ``/mode``, the mode
cycle, and the theme picker.

Run directly: ``python qwowl35/tests/theme_command_test.py``. The dispatch/clear
tests use fakes (no TUI); the picker and mode-cycle tests drive the real app
headless via Textual's Pilot.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import Agent  # noqa: E402
from app import QwowlApp  # noqa: E402
from modes import Mode  # noqa: E402
from widgets.theme_selector import ThemeSelector  # noqa: E402


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


# --------------------------------------------------------------------------- #
# Command dispatch (unit)
# --------------------------------------------------------------------------- #
def _dispatch_app():
    app = QwowlApp.__new__(QwowlApp)
    calls: list[str] = []
    app.exit = lambda: calls.append("quit")  # quit/exit/abort/close -> App.exit()
    app._clear_conversation = lambda: calls.append("clear")
    app._open_theme_selector = lambda: calls.append("theme")
    app._open_font_selector = lambda: calls.append("fonts")
    return app, calls


def test_dispatch_routes_exact_commands() -> None:
    for cmd, expected in [
        ("/quit", "quit"),
        ("/exit", "quit"),
        ("/abort", "quit"),
        ("/close", "quit"),
        ("/clear", "clear"),
        ("/theme", "theme"),
        ("/fonts", "fonts"),
    ]:
        app, calls = _dispatch_app()
        assert_true(app._dispatch_command(cmd), f"{cmd} handled")
        assert_equal(calls, [expected], f"{cmd} routed")


def test_dispatch_ignores_non_commands() -> None:
    app, calls = _dispatch_app()
    for text in ("hello", "quit", "/quitx", "/ quit", "/THEME", "/clear now", "/", "\\quit"):
        assert_true(not app._dispatch_command(text), f"{text!r} not a command")
    assert_equal(calls, [], "no handlers fired for non-commands")


# --------------------------------------------------------------------------- #
# \clear conversation reset (unit)
# --------------------------------------------------------------------------- #
class _FakeAgent:
    def __init__(self) -> None:
        self.cleared = False

    def clear(self) -> None:
        self.cleared = True


class _FakeChat:
    def __init__(self) -> None:
        self.cleared = False

    def clear(self) -> None:
        self.cleared = True


class _FakePanel:
    def __init__(self) -> None:
        self.display = True
        self.content = "stuff"

    def update(self, content) -> None:
        self.content = content


class _FakeStatus:
    def __init__(self) -> None:
        self.mode: Mode | None = None

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode


class _FakeSessionStore:
    def __init__(self) -> None:
        self.rotated = False

    def rotate(self) -> str:
        self.rotated = True
        return "new-hash"


def test_clear_conversation_resets_everything() -> None:
    app = QwowlApp.__new__(QwowlApp)
    app.agent = _FakeAgent()
    app.chat = _FakeChat()
    app.queue_panel = _FakePanel()
    app.status = _FakeStatus()
    app._session_store = _FakeSessionStore()
    app.mode = Mode.CHAT
    app._queued_messages = ["queued"]
    infos: list[str] = []
    app.set_info = infos.append

    app._clear_conversation()

    assert_true(app.agent.cleared, "agent history cleared")
    assert_true(app.chat.cleared, "chat transcript cleared")
    assert_true(app._session_store.rotated,
                "cleared conversation becomes a restorable past session")
    assert_equal(app._queued_messages, [], "queue emptied")
    assert_true(not app.queue_panel.display, "queue panel hidden")
    assert_equal(app.mode, Mode.NORMAL, "a cleared conversation restarts in NORMAL")
    assert_equal(app.status.mode, Mode.NORMAL, "status bar box follows the reset")
    assert_equal(infos, ["cleared"], "mascot flashed 'cleared'")


def test_dispatch_mode_command_cycles_selects_and_locks() -> None:
    app = QwowlApp.__new__(QwowlApp)
    app._busy = False
    app.mode = Mode.NORMAL
    app.status = _FakeStatus()
    warnings: list[str] = []
    app.set_warning = warnings.append

    assert_true(app._dispatch_command("/mode"), "/mode handled")
    assert_equal(app.mode, Mode.PLAN, "bare /mode cycles NORMAL -> PLAN")
    assert_equal(app.status.mode, Mode.PLAN, "status bar box follows")

    assert_true(app._dispatch_command("/mode chat"), "/mode chat handled")
    assert_equal(app.mode, Mode.CHAT, "named mode selected directly")

    assert_true(app._dispatch_command("/mode visual"), "/mode visual handled")
    assert_equal(app.mode, Mode.CHAT, "display modes are not selectable")
    assert_true(warnings and "unknown mode" in warnings[-1], "unknown mode warned")

    app._busy = True
    assert_true(app._dispatch_command("/mode"), "handled while busy")
    assert_equal(app.mode, Mode.CHAT, "mode locked while a turn runs")
    assert_true("locked" in warnings[-1], "lock warned")


# --------------------------------------------------------------------------- #
# Agent.clear (unit) — preserve system message, reset guards
# --------------------------------------------------------------------------- #
def test_agent_clear_keeps_system_and_resets_guards() -> None:
    agent = Agent.__new__(Agent)
    agent.registry = None  # only needed if the system message is missing
    agent.messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    agent._last_signatures = {"bash": "sig"}
    agent._last_repeat_msg = "again"
    agent._bash_rewrite_counts = {"a.py": 3}
    agent._last_rewrite_advice = "nudge"

    agent.clear()

    assert_equal(agent.messages, [{"role": "system", "content": "SYS"}], "only system kept")
    assert_equal(agent._last_signatures, {}, "tool signatures reset")
    assert_equal(agent._last_repeat_msg, None, "repeat msg reset")
    assert_equal(agent._bash_rewrite_counts, {}, "rewrite counts reset")
    assert_equal(agent._last_rewrite_advice, None, "rewrite advice reset")


# --------------------------------------------------------------------------- #
# Theme picker + live preview (Pilot integration)
# --------------------------------------------------------------------------- #
def test_theme_picker_preview_commit_and_revert() -> None:
    async def scenario() -> None:
        app = QwowlApp()
        async with app.run_test() as pilot:
            names = app._theme_catalog.names
            assert_true(len(names) >= 2, "need at least two themes to navigate")
            app.apply_theme_preview(theme_default := names[0], "dark")

            # /theme opens the picker (real worker + push_screen_wait flow)
            app._dispatch_command("/theme")
            await pilot.pause()
            await pilot.pause()
            assert_true(isinstance(app.screen, ThemeSelector), "picker opened")

            # Down previews the next theme live (Textual theme actually changes).
            await pilot.press("down")
            await pilot.pause()
            assert_equal(app._theme_name, names[1], "preview selected next theme")
            assert_equal(
                app.theme, app._theme_catalog.textual_name(names[1], "dark"), "textual theme applied"
            )

            # Escape reverts to the theme active when the picker opened.
            await pilot.press("escape")
            await pilot.pause()
            await pilot.pause()
            assert_true(not isinstance(app.screen, ThemeSelector), "picker dismissed")
            assert_equal(app._theme_name, theme_default, "escape reverted theme")

    asyncio.run(scenario())


def test_clear_command_empties_chat_keeps_system() -> None:
    # /clear resets the orchestrator's session state.
    async def scenario() -> None:
        app = QwowlApp()
        async with app.run_test() as pilot:
            app.chat.add_user("hello there")
            app.agent.turn_log.append(("hello there", "answered"))
            app.agent.messages.append({"role": "system", "content": "=== stage: execute ==="})
            await pilot.pause()
            assert_true(len(app.chat.children) > 0, "chat has a message")

            app._dispatch_command("/clear")
            await pilot.pause()

            assert_equal(len(app.chat.children), 0, "transcript emptied")
            assert_equal(app.agent.turn_log, [], "session log cleared")
            assert_equal(app.agent.messages, [], "debug transcript cleared")

    asyncio.run(scenario())


def test_clear_command_resets_chat_lineage_state() -> None:
    # /clear resets the chat lineage, session log, and the debug transcript.
    async def scenario() -> None:
        app = QwowlApp()
        async with app.run_test() as pilot:
            app.chat.add_user("hello there")
            app.agent.chat_messages.append({"role": "user", "content": "hello there"})
            app.agent.turn_log.append(("hello there", "answered"))
            app.agent.messages.append({"role": "system", "content": "=== stage: chat ==="})
            await pilot.pause()

            app._dispatch_command("/clear")
            await pilot.pause()

            assert_equal(len(app.chat.children), 0, "transcript emptied")
            assert_equal(
                [m["role"] for m in app.agent.chat_messages], ["system"],
                "chat lineage reset to its system prompt",
            )
            assert_equal(app.agent.turn_log, [], "session log cleared")
            assert_equal(app.agent.messages, [], "debug transcript cleared")

    asyncio.run(scenario())


def test_app_binds_interactive_plan_callbacks_at_construction() -> None:
    # Regression guard for "quiz never shown / plan approved without me":
    # a constructed app must have the plan gate and question quiz bound to
    # its modal methods — never the silent auto fallbacks.
    app = QwowlApp()
    assert_true(
        app.agent.registry.plan._plan_callback == app._approve_plan,
        "plan gate bound to the approval modal",
    )
    assert_true(
        app.agent.registry.plan._question_callback == app._ask_user_questions,
        "questions bound to the quiz modal",
    )
    assert_true(
        app.agent.registry.plan.notify is not None,
        "fallbacks would announce themselves",
    )


def test_shift_tab_cycles_user_modes_and_locks_while_busy() -> None:
    async def scenario() -> None:
        app = QwowlApp()
        async with app.run_test() as pilot:
            assert_equal(app.mode, Mode.NORMAL, "every conversation starts in NORMAL")
            assert_equal(app.status.state.mode, Mode.NORMAL, "box shows NORMAL")

            expected_cycle = [Mode.PLAN, Mode.WEB, Mode.CHAT, Mode.NORMAL]
            for expected in expected_cycle:
                await pilot.press("shift+tab")
                await pilot.pause()
                assert_equal(app.mode, expected, f"cycle reaches {expected.value}")
                assert_equal(
                    app.status.state.mode, expected, "status bar box tracks the cycle"
                )

            # The mode is locked once inference runs: cycling is refused.
            app._busy = True
            await pilot.press("shift+tab")
            await pilot.pause()
            assert_equal(app.mode, Mode.NORMAL, "mode locked while a turn runs")

    asyncio.run(scenario())


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("theme command tests passed")


if __name__ == "__main__":
    main()
