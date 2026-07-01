"""Tests for local ``\\``-commands: dispatch, ``\\clear``, and the theme picker.

Run directly: ``python qwowl35/tests/theme_command_test.py``. The dispatch/clear
tests use fakes (no TUI); the picker test drives the real app headless via
Textual's Pilot.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import Agent  # noqa: E402
from app import QwowlApp  # noqa: E402
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
    app.action_quit = lambda: calls.append("quit")
    app._clear_conversation = lambda: calls.append("clear")
    app._open_theme_selector = lambda: calls.append("theme")
    return app, calls


def test_dispatch_routes_exact_commands() -> None:
    for cmd, expected in [
        ("\\quit", "quit"),
        ("\\abort", "quit"),
        ("\\exit", "quit"),
        ("\\clear", "clear"),
        ("\\theme", "theme"),
    ]:
        app, calls = _dispatch_app()
        assert_true(app._dispatch_command(cmd), f"{cmd} handled")
        assert_equal(calls, [expected], f"{cmd} routed")


def test_dispatch_ignores_non_commands() -> None:
    app, calls = _dispatch_app()
    for text in ("hello", "quit", "\\quitx", "\\ quit", "\\THEME", "\\clear now", "\\"):
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


def test_clear_conversation_resets_everything() -> None:
    app = QwowlApp.__new__(QwowlApp)
    app.agent = _FakeAgent()
    app.chat = _FakeChat()
    app.queue_panel = _FakePanel()
    app._queued_messages = ["queued"]
    infos: list[str] = []
    app.set_info = infos.append

    app._clear_conversation()

    assert_true(app.agent.cleared, "agent history cleared")
    assert_true(app.chat.cleared, "chat transcript cleared")
    assert_equal(app._queued_messages, [], "queue emptied")
    assert_true(not app.queue_panel.display, "queue panel hidden")
    assert_equal(infos, ["cleared"], "mascot flashed 'cleared'")


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
    agent._last_tool_signature = "sig"
    agent._last_repeat_msg = "again"
    agent._bash_rewrite_counts = {"a.py": 3}
    agent._last_rewrite_advice = "nudge"

    agent.clear()

    assert_equal(agent.messages, [{"role": "system", "content": "SYS"}], "only system kept")
    assert_equal(agent._last_tool_signature, None, "tool signature reset")
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

            # \theme opens the picker (real worker + push_screen_wait flow)
            app._dispatch_command("\\theme")
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
    async def scenario() -> None:
        app = QwowlApp()
        async with app.run_test() as pilot:
            app.chat.add_user("hello there")
            app.agent.messages.append({"role": "user", "content": "hello there"})
            await pilot.pause()
            assert_true(len(app.chat.children) > 0, "chat has a message")

            app._dispatch_command("\\clear")
            await pilot.pause()

            assert_equal(len(app.chat.children), 0, "transcript emptied")
            assert_equal([m["role"] for m in app.agent.messages], ["system"], "only system kept")

    asyncio.run(scenario())


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("theme command tests passed")


if __name__ == "__main__":
    main()
