"""Tests for the editor spawn's background block (agents/spawn_context).

The block hands the editor a compact view of the delegating agent's plan,
recent activity, and reasoning — with every tool call obfuscated into plain
markdown so the editor is never shown tool-call syntax for tools it lacks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.spawn_context import (  # noqa: E402
    BACKGROUND_HEADER,
    PLAN_HEAD_CHARS,
    REASONING_TAIL_CHARS,
    RESULT_LINES,
    TOTAL_CAP_CHARS,
    build_editor_background,
    clip_result,
    obfuscate_call,
)
from tools.files.adapter import TOOL_ATTENTION_MARKER  # noqa: E402


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def _assistant(content: str = "", calls: list | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for call_id, name, args in calls
        ]
    return message


def _tool(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_bash_call_renders_as_markdown_fence() -> None:
    lines = obfuscate_call("run_shell_command", {"command": "ls -la | grep foo"})
    assert_equal(lines, ["```bash", "ls -la | grep foo", "```"], "bash call is a fence")


def test_delegator_edit_renders_one_line() -> None:
    lines = obfuscate_call(
        "edit", {"filename": "a.py", "line_ranges": "3-9", "instructions": "rename x to y"}
    )
    assert_equal(lines, ["- edit `a.py` (lines 3-9): rename x to y"], "delegator edit line")


def test_main_field_generic_and_plan_calls() -> None:
    assert_equal(
        obfuscate_call("explore", {"task": "find the retry helper"}),
        ["- explore: find the retry helper"],
        "explore call",
    )
    assert_equal(
        obfuscate_call("plan", {"todos": ["a", "b", "c"]}),
        ["- plan: proposed 3 todo(s)"],
        "plan present call",
    )
    [generic] = obfuscate_call("lsp", {"operation": "hover", "file": "a.py"})
    assert_true(
        generic.startswith("- lsp: ") and "a.py" in generic and "hover" in generic,
        f"generic key=value summary: {generic}",
    )
    assert_equal(obfuscate_call("mystery", {}), ["- mystery"], "argless unknown call")


def test_clip_result_truncates_and_strips_marker() -> None:
    text = TOOL_ATTENTION_MARKER + "\n".join(f"line{i}" for i in range(10))
    lines = clip_result(text)
    assert_equal(lines[0], "```", "plain fence opens")
    assert_equal(lines[1:1 + RESULT_LINES], [f"line{i}" for i in range(RESULT_LINES)], "first lines kept")
    assert_equal(lines[-1], "… (output truncated)", "truncation note present")
    assert_true(all(TOOL_ATTENTION_MARKER not in line for line in lines), "marker stripped")
    assert_equal(clip_result("   \n  "), [], "blank result renders nothing")


def test_background_full_shape_never_leaks_tool_call_syntax() -> None:
    history = [
        {"role": "user", "content": "add retries"},
        _assistant(
            "The helper exists; checking the call site.",
            [("c1", "run_shell_command", {"command": "grep -n fetch_page util.py"})],
        ),
        _tool("c1", "42:def fetch_page(url):\n43:    return get(url)\nmore\nmore\nmore\nmore"),
        _assistant("", [("c2", "edit", {"filename": "util.py", "line_ranges": "43", "instructions": "wrap in retry"})]),
    ]
    block = build_editor_background(
        plan_markdown="# Add retries\n1. helper\n2. wire in",
        current_todo="(2/2): wire in",
        history=history,
        reasoning="the timeout kwarg must be preserved",
    )
    assert_true(block.startswith(BACKGROUND_HEADER), "header opens the block")
    assert_true("Approved plan (excerpt):" in block, "plan section present")
    assert_true("Current todo (2/2): wire in" in block, "current todo present")
    assert_true("```bash" in block and "grep -n fetch_page util.py" in block, "bash fence")
    assert_true("42:def fetch_page(url):" in block, "clipped result present")
    assert_true("… (output truncated)" in block, "result truncation noted")
    assert_true("the timeout kwarg must be preserved" in block, "reasoning tail present")
    assert_true("The helper exists" in block, "earlier visible text present")
    # The final turn ISSUED the edit call: it must not be re-rendered (it
    # duplicates File/Instructions), and no tool-call syntax may survive.
    assert_true("- edit `util.py`" not in block, "spawning edit call skipped")
    assert_true("<tool_call" not in block, "no tool_call XML")
    assert_true('"arguments"' not in block, "no tool_call JSON")
    assert_true("run_shell_command" not in block, "no wire tool name for bash")


def test_caps_apply_to_plan_and_reasoning() -> None:
    block = build_editor_background(
        plan_markdown="P" * (PLAN_HEAD_CHARS * 2),
        current_todo=None,
        history=[],
        reasoning="R" * (REASONING_TAIL_CHARS * 2),
    )
    assert_true("… (plan truncated)" in block, "plan head capped")
    assert_true("…" + "R" * 10 in block, "reasoning kept as a tail with ellipsis")
    assert_true("R" * (REASONING_TAIL_CHARS + 1) not in block, "reasoning tail capped")


def test_total_cap_bounds_the_block() -> None:
    history = []
    for i in range(30):
        history.append(
            _assistant(
                f"step {i} " + "x" * 400,
                [(f"c{i}", "run_shell_command", {"command": f"echo {i} " + "y" * 300})],
            )
        )
        history.append(_tool(f"c{i}", "\n".join("z" * 200 for _ in range(8))))
    history.append(_assistant("", [("cf", "edit", {"filename": "a", "line_ranges": "1", "instructions": "x"})]))
    block = build_editor_background(None, None, history, "r" * 5000)
    assert_true(
        len(block) <= TOTAL_CAP_CHARS + len("\n… (background truncated)"),
        f"hard cap respected: {len(block)}",
    )
    assert_true(block.endswith("… (background truncated)"), "truncation note closes the block")


def test_empty_inputs_yield_empty_block() -> None:
    assert_equal(build_editor_background(None, None, [], ""), "", "nothing to tell")
    # Only the spawning turn, with no visible text and no reasoning: still
    # nothing worth a background section.
    history = [
        {"role": "user", "content": "fix it"},
        _assistant("", [("c1", "edit", {"filename": "a.py", "line_ranges": "1", "instructions": "fix"})]),
    ]
    assert_equal(build_editor_background(None, None, history, ""), "", "spawn-only history is empty")


def test_normal_mode_has_no_plan_section() -> None:
    history = [
        {"role": "user", "content": "task"},
        _assistant("did a thing", [("c1", "run_shell_command", {"command": "ls"})]),
        _tool("c1", "a.py"),
        _assistant("", [("c2", "edit", {"filename": "a.py", "line_ranges": "1", "instructions": "x"})]),
    ]
    block = build_editor_background(None, None, history, "")
    assert_true("Approved plan" not in block and "Current todo" not in block, "no plan section")
    assert_true("Recent activity of the delegating agent:" in block, "activity still present")


def main() -> None:
    test_bash_call_renders_as_markdown_fence()
    test_delegator_edit_renders_one_line()
    test_main_field_generic_and_plan_calls()
    test_clip_result_truncates_and_strips_marker()
    test_background_full_shape_never_leaks_tool_call_syntax()
    test_caps_apply_to_plan_and_reasoning()
    test_total_cap_bounds_the_block()
    test_empty_inputs_yield_empty_block()
    test_normal_mode_has_no_plan_section()
    print("spawn context tests passed")


if __name__ == "__main__":
    main()
