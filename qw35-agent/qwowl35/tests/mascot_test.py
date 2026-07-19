"""Tests for the owl mascot's per-tool-family running animations."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mascot  # noqa: E402
from mascot import State  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


# The family each tool call is expected to animate as (mirrors the plan's
# family map, independent of the module's own dict so a wrong remap is caught).
_EXPECTED = {
    "run_shell_command": State.BASH,
    "bash": State.BASH,
    "inspect_file": State.READ,
    "beginTransaction": State.READ,
    "grep_search": State.SEARCH,
    "glob": State.SEARCH,
    "list_directory": State.SEARCH,
    "edit": State.EDIT,
    "insert": State.EDIT,
    "delete": State.EDIT,
    "web_fetch": State.WEB,
    "plan": State.PLAN,
    "ask_user_question": State.ASK,
    "explore": State.SEARCH,
    "resume": State.JUDGE,
}


def test_state_for_tool_maps_each_family() -> None:
    for name, expected in _EXPECTED.items():
        got = mascot.state_for_tool(name)
        assert_true(got is expected, f"{name} -> {expected} (got {got})")


def test_unknown_tool_falls_back_to_edit() -> None:
    assert_true(mascot.state_for_tool("no_such_tool") is State.EDIT, "unknown -> EDIT")
    assert_true(mascot.state_for_tool("") is State.EDIT, "empty -> EDIT")


def test_every_running_state_has_a_registered_animation() -> None:
    # set_state looks the animation up by State.value in ANIMATIONS; a state
    # returned by state_for_tool with no entry would silently show nothing.
    for name in _EXPECTED:
        state = mascot.state_for_tool(name)
        assert_true(
            state.value in mascot.ANIMATIONS,
            f"{state} has an animation registered",
        )


def test_new_animations_render_without_error() -> None:
    # Each new family animation must produce a 4-line owl for every frame.
    for state in (State.READ, State.SEARCH, State.WEB, State.PLAN, State.ASK, State.JUDGE):
        anim = mascot.ANIMATIONS[state.value]
        assert_true(len(anim.frames) >= 1, f"{state} has frames")
        for tick in range(len(anim.frames)):
            rendered = mascot.render(anim.frame(tick), info="/tmp/demo")
            assert_true(rendered.count("\n") == 3, f"{state} frame {tick} is 4 rows")


def main() -> None:
    test_state_for_tool_maps_each_family()
    test_unknown_tool_falls_back_to_edit()
    test_every_running_state_has_a_registered_animation()
    test_new_animations_render_without_error()
    print("mascot tests passed")


if __name__ == "__main__":
    main()
