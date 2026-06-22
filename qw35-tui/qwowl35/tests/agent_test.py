"""Tests for the agent's repeated-tool-call guard.

Run directly: ``python qwowl35/tests/agent_test.py``. These never touch the network
or a real shell — they drive ``Agent.run_turn`` with a fake stream, registry,
and UI to assert the dedup behaviour.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# The package modules import each other by bare name (flat sys.path quirk), so
# put the qwowl35 dir itself on the path and import flat.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import (  # noqa: E402
    CONTINUATION_FEEDBACK,
    CONTINUATION_MAX_NUDGES,
    REPEATED_TOOL_NOTES,
    REWRITE_ADVICE_NOTES,
    Agent,
    _authored_write_targets,
    _bash_syntax_warning,
    build_auto_read_block,
    repeated_tool_message,
    rewrite_advice_message,
)
from client import AssistantTurn, ToolCall  # noqa: E402
from tools.files.adapter import HashlineTools  # noqa: E402


def _denial_pool(name: str) -> list[str]:
    return [note.format(name=name) for note in REPEATED_TOOL_NOTES]


def _advice_pool(file: str) -> list[str]:
    return [note.format(file=file) for note in REWRITE_ADVICE_NOTES]


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


class FakeChat:
    def __init__(self) -> None:
        self.tool_results: list[tuple[str, str, bool]] = []
        self.warnings: list[str] = []
        self.users: list[str] = []

    def add_user(self, text: str) -> None:
        self.users.append(text)

    def flush_reasoning(self) -> None: ...
    def flush_assistant(self) -> None: ...
    def add_error(self, *a, **k) -> None: ...

    def add_tool_result(self, index, name, result, is_error=False) -> None:
        self.tool_results.append((name, result, is_error))

    def add_warning(self, text: str) -> None:
        self.warnings.append(text)


class FakeUI:
    def __init__(self, queued_batches: list[str] | None = None) -> None:
        self.chat = FakeChat()
        self.queued_batches = list(queued_batches or [])
        self.popped_batches: list[str] = []
        self.warnings: list[str] = []

    def set_state(self, *a, **k) -> None: ...
    def begin_generation(self, *a, **k) -> None: ...
    def set_prefill(self, *a, **k) -> None: ...
    def add_reasoning_delta(self, *a, **k) -> None: ...
    def set_usage(self, *a, **k) -> None: ...
    def set_error(self, *a, **k) -> None: ...
    def set_info(self, *a, **k) -> None: ...

    def set_warning(self, message: str) -> None:
        self.warnings.append(message)

    def pop_queued_user_batch(self) -> str | None:
        if not self.queued_batches:
            return None
        batch = self.queued_batches.pop(0)
        self.popped_batches.append(batch)
        return batch


class FakeRegistry:
    """Records every executed call and returns a canned result."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict]] = []

    async def execute(self, name: str, arguments: dict) -> str:
        self.executed.append((name, arguments))
        return f"ran {name}"


def make_agent(turns: list[AssistantTurn], queued_batches: list[str] | None = None):
    """Build an Agent whose _stream_assistant yields the given turns in order."""
    registry = FakeRegistry()
    ui = FakeUI(queued_batches)
    # Config/client/prompts are untouched because _stream_assistant is replaced.
    agent = Agent.__new__(Agent)
    agent.client = None
    agent.registry = registry
    agent.config = None
    agent.ui = ui
    agent.messages = []
    agent._last_tool_signature = None
    agent._bash_rewrite_counts = {}
    agent._last_rewrite_advice = None

    pending = list(turns)
    stream_messages: list[list[dict]] = []

    async def fake_stream() -> AssistantTurn:
        stream_messages.append(json.loads(json.dumps(agent.messages)))
        return pending.pop(0)

    agent._stream_assistant = fake_stream  # type: ignore[assignment]
    agent.stream_messages = stream_messages
    # Skip the wake animation sleep.
    import agent as agent_mod

    agent_mod.WAKEUP_HOLD_SECONDS = 0
    return agent, registry, ui


def call(name: str, arguments: dict, index: int = 0) -> ToolCall:
    return ToolCall(id=f"c{index}", name=name, arguments=arguments, index=index)


def test_identical_consecutive_call_is_denied() -> None:
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": "ls"})]),
        AssistantTurn(tool_calls=[call("bash", {"command": "ls"})]),  # repeat
        AssistantTurn(content="done"),  # no tool calls → ends the turn
    ]
    agent, registry, ui = make_agent(turns)
    asyncio.run(agent.run_turn("hi"))

    assert_equal(registry.executed, [("bash", {"command": "ls"})], "executed once")
    denied = [r for _, r, is_err in ui.chat.tool_results if is_err]
    assert_equal(len(denied), 1, "one denial reported")
    # The denial is one of the randomly-picked notes from the pool.
    assert_true(denied[0] in _denial_pool("bash"), "denial drawn from the note pool")


def test_changed_arguments_clear_the_guard() -> None:
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": "ls"})]),
        AssistantTurn(tool_calls=[call("bash", {"command": "pwd"})]),  # different args
        AssistantTurn(content="done"),
    ]
    agent, registry, _ = make_agent(turns)
    asyncio.run(agent.run_turn("hi"))

    assert_equal(
        registry.executed,
        [("bash", {"command": "ls"}), ("bash", {"command": "pwd"})],
        "both ran",
    )


def test_third_identical_call_still_denied() -> None:
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": "ls"})]),
        AssistantTurn(tool_calls=[call("bash", {"command": "ls"})]),
        AssistantTurn(tool_calls=[call("bash", {"command": "ls"})]),
        AssistantTurn(content="done"),
    ]
    agent, registry, ui = make_agent(turns)
    asyncio.run(agent.run_turn("hi"))

    assert_equal(len(registry.executed), 1, "executed exactly once")
    denied = [r for _, r, is_err in ui.chat.tool_results if is_err]
    assert_equal(len(denied), 2, "two denials")
    # Both denials come from the pool, and consecutive ones never read the same.
    pool = _denial_pool("bash")
    assert_true(all(d in pool for d in denied), "denials drawn from the note pool")
    assert_true(denied[0] != denied[1], "consecutive denials are worded differently")


def test_denial_notes_vary_and_give_no_direction() -> None:
    # exclude= must never return the excluded note, so consecutive denials differ.
    pool = _denial_pool("bash")
    for note in pool:
        for _ in range(20):
            assert_true(repeated_tool_message("bash", exclude=note) != note, "exclude honored")
    # No note prescribes a specific next action (it may invite carrying on, but
    # never names a tool or tells the model what to do instead).
    banned = ("instead", "try ", "use the file", "use bash", "switch", "you should")
    for m in pool:
        low = m.lower()
        for word in banned:
            assert_true(word not in low, f"message must give no direction (found {word!r})")


def test_argument_key_order_does_not_matter() -> None:
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"a": 1, "b": 2})]),
        AssistantTurn(tool_calls=[call("bash", {"b": 2, "a": 1})]),  # same, reordered
        AssistantTurn(content="done"),
    ]
    agent, registry, _ = make_agent(turns)
    asyncio.run(agent.run_turn("hi"))

    assert_equal(len(registry.executed), 1, "reordered args treated as identical (bash)")


def test_invalid_json_markers_are_not_deduped() -> None:
    invalid = {"_invalid_json": True, "_json_error": "Expecting ':'"}
    turns = [
        AssistantTurn(tool_calls=[call("bash", invalid)]),
        AssistantTurn(tool_calls=[call("bash", invalid)]),
        AssistantTurn(content="done"),
    ]
    agent, registry, ui = make_agent(turns)
    asyncio.run(agent.run_turn("hi"))

    assert_equal(
        registry.executed,
        [("bash", invalid), ("bash", invalid)],
        "invalid json calls are not hidden by repeat guard",
    )
    denied = [r for _, r, is_err in ui.chat.tool_results if is_err]
    assert_equal(denied, [], "no repeat denial for invalid json")


def test_file_edits_are_not_deduped() -> None:
    # Edit tools must always re-run: a content-free "already did that" denial can
    # mislead the model into thinking a failed edit landed.
    turns = [
        AssistantTurn(tool_calls=[call("edit", {"path": "a.py", "anchor": "1:af", "text": "x"})]),
        AssistantTurn(tool_calls=[call("edit", {"path": "a.py", "anchor": "1:af", "text": "x"})]),  # identical
        AssistantTurn(content="done"),
    ]
    agent, registry, _ = make_agent(turns)
    asyncio.run(agent.run_turn("hi"))
    assert_equal(len(registry.executed), 2, "identical file edits both execute (no dedup)")


def test_new_user_turn_resets_guard() -> None:
    agent, registry, _ = make_agent(
        [AssistantTurn(tool_calls=[call("bash", {"command": "ls"})]), AssistantTurn(content="ok")]
    )
    asyncio.run(agent.run_turn("first"))
    # Second turn repeats the same call — allowed because the guard reset.
    agent._stream_assistant = _replayer(  # type: ignore[assignment]
        [AssistantTurn(tool_calls=[call("bash", {"command": "ls"})]), AssistantTurn(content="ok")]
    )
    asyncio.run(agent.run_turn("second"))

    assert_equal(len(registry.executed), 2, "same call runs again in a new turn")


def test_queued_batch_after_tool_result_is_sent_before_next_stream() -> None:
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": "ls"})]),
        AssistantTurn(content="queued answer"),
    ]
    agent, registry, ui = make_agent(turns, queued_batches=["queued follow-up"])
    ok = asyncio.run(agent.run_turn("hi"))

    assert_true(ok, "turn completes")
    assert_equal(registry.executed, [("bash", {"command": "ls"})], "tool ran")
    assert_equal(ui.popped_batches, ["queued follow-up"], "queue popped once")
    assert_equal(len(agent.stream_messages), 2, "queued input triggered second stream")
    second_stream = agent.stream_messages[1]
    assert_equal(second_stream[-2]["role"], "tool", "tool result precedes queued user")
    assert_equal(
        second_stream[-1],
        {"role": "user", "content": "queued follow-up"},
        "queued user is in context before next stream",
    )
    assert_equal(ui.chat.users, ["hi", "queued follow-up"], "queued user shown in chat")


def test_queued_batch_at_final_response_continues_turn() -> None:
    turns = [
        AssistantTurn(content="first final"),
        AssistantTurn(content="queued final"),
    ]
    agent, _, ui = make_agent(turns, queued_batches=["queued after final"])
    ok = asyncio.run(agent.run_turn("hi"))

    assert_true(ok, "turn completes after queued follow-up")
    assert_equal(ui.popped_batches, ["queued after final"], "queue popped at final boundary")
    assert_equal(len(agent.stream_messages), 2, "final boundary continued into a second stream")
    assert_equal(
        agent.stream_messages[1][-1],
        {"role": "user", "content": "queued after final"},
        "queued final-boundary message is sent before second stream",
    )
    assert_equal(ui.chat.users, ["hi", "queued after final"], "queued message rendered")


def test_completed_tool_history_is_compacted() -> None:
    huge = "x" * 2000
    agent = Agent.__new__(Agent)
    agent.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c0",
                    "type": "function",
                    "function": {
                        "name": "edit",
                        "arguments": json.dumps(
                            {
                                "file": "m.py",
                                "anchor": "2:aa",
                                "content": huge,
                            }
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c0", "content": "Edited lines 2-2.\n" + huge},
        {"role": "assistant", "content": "done"},
    ]

    agent._compact_completed_history()

    serialized = json.dumps(agent.messages)
    assert_true(huge not in serialized, "large payload removed")
    assert_true("tool_calls" not in serialized, "tool-call objects removed")
    assert_true("edit on m.py at 2:aa" in serialized, "summary keeps locator")
    assert_equal(agent.messages[-1]["content"], "done", "final assistant text kept")


def test_invalid_json_marker_is_not_serialized_to_history() -> None:
    message = Agent._assistant_message(
        AssistantTurn(
            tool_calls=[
                call(
                    "bash",
                    {
                        "_invalid_json": True,
                        "_json_error": "Expecting ',' delimiter at line 1 column 35",
                    },
                )
            ]
        )
    )

    serialized = json.dumps(message)
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert_equal(args, {}, "invalid marker hidden from history")
    assert_true("_invalid_json" not in serialized, "marker not serialized")


def test_malformed_tool_call_text_gets_repair_turn() -> None:
    malformed = (
        "<tool_call><function=edit>"
        "<parameter=file>solve.py</parameter><parameter=anchor></parameter>"
        '<parameter=content>#!/usr/bin/env python3\\n"""bad raw xml"""'
    )
    turns = [
        AssistantTurn(content=malformed),
        AssistantTurn(content="done"),
    ]
    agent, registry, ui = make_agent(turns)
    ok = asyncio.run(agent.run_turn("fix file"))

    assert_true(ok, "turn continues after malformed tool call")
    assert_equal(registry.executed, [], "malformed text is not executed as a tool")
    assert_equal(len(ui.warnings), 1, "warning shown on the mascot")
    repair_messages = [
        msg for msg in agent.messages if msg.get("role") == "user" and "malformed <tool_call>" in msg.get("content", "")
    ]
    assert_equal(len(repair_messages), 1, "repair instruction sent to model")
    assert_true("send only the corrected tool call" in repair_messages[0]["content"], "repair is actionable")
    assert_true("valid nested XML" in repair_messages[0]["content"], "repair requests nested xml")
    assert_true("<function=tool_name>" in repair_messages[0]["content"], "repair names function form")


def test_malformed_tool_call_retries_are_capped() -> None:
    malformed = "<tool_call><function=edit><parameter=file>a.py</parameter>"
    turns = [AssistantTurn(content=malformed) for _ in range(4)]
    agent, _, ui = make_agent(turns)
    ok = asyncio.run(agent.run_turn("fix file"))

    assert_true(not ok, "repeated malformed calls fail the turn")
    assert_equal(len(ui.warnings), 4, "each malformed output is surfaced")


def _bash_results(ui: FakeUI) -> list[str]:
    return [r for name, r, _ in ui.chat.tool_results if name == "bash"]


def test_first_full_rewrite_silent_then_advises() -> None:
    # Two different commands that both wholesale-rewrite app.py. The first write
    # is silent; the second earns a nudge toward the anchored edit tools.
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": "printf 'a\\n' > app.py"})]),
        AssistantTurn(tool_calls=[call("bash", {"command": "printf 'b\\n' > app.py"})]),
        AssistantTurn(content="done"),
    ]
    agent, registry, ui = make_agent(turns)
    asyncio.run(agent.run_turn("hi"))

    assert_equal(len(registry.executed), 2, "both writes execute (not deduped)")
    results = _bash_results(ui)
    assert_equal(results[0], "ran bash", "first wholesale write is silent")
    assert_true(results[1].startswith("ran bash\n\n"), "second write carries advice")
    advice = results[1].split("\n\n", 1)[1]
    assert_true(advice in _advice_pool("app.py"), "advice drawn from the note pool")


def test_repeated_append_advises() -> None:
    # `>>` appends to the same authored file are the slower escape hatch, so a
    # repeated append now earns the same nudge as a repeated `>` rewrite.
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": "echo a >> notes.py"})]),
        AssistantTurn(tool_calls=[call("bash", {"command": "echo b >> notes.py"})]),
        AssistantTurn(content="done"),
    ]
    agent, _, ui = make_agent(turns)
    asyncio.run(agent.run_turn("hi"))

    results = _bash_results(ui)
    assert_equal(results[0], "ran bash", "first append silent")
    assert_true(results[1].startswith("ran bash\n\n"), "second append advised")
    assert_true(results[1].split("\n\n", 1)[1] in _advice_pool("notes.py"),
                "append advice drawn from the soft pool")


def test_scratch_redirect_never_advises() -> None:
    # Writing program output to /tmp is not authoring a source file, so repeated
    # scratch redirects draw no rewrite advice.
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": "python3 cal.py > /tmp/out.txt"})]),
        AssistantTurn(tool_calls=[call("bash", {"command": "python3 cal.py 2>&1 > /tmp/out.txt"})]),
        AssistantTurn(content="done"),
    ]
    agent, _, ui = make_agent(turns)
    asyncio.run(agent.run_turn("hi"))

    assert_equal(_bash_results(ui), ["ran bash", "ran bash"], "scratch /tmp writes draw no advice")


def test_distinct_files_each_written_once_are_silent() -> None:
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": "echo a > app.py"})]),
        AssistantTurn(tool_calls=[call("bash", {"command": "echo b > util.py"})]),
        AssistantTurn(content="done"),
    ]
    agent, _, ui = make_agent(turns)
    asyncio.run(agent.run_turn("hi"))

    assert_equal(_bash_results(ui), ["ran bash", "ran bash"], "distinct first writes are silent")


def test_repeated_rewrite_escalates() -> None:
    # Four rewrites of the same file: 1st silent, 2nd a soft pool nudge, 3rd and
    # 4th a firm fixed escalation (is_error) that names the edit tools.
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": "printf '1' > app.py"})]),
        AssistantTurn(tool_calls=[call("bash", {"command": "printf '2' > app.py"})]),
        AssistantTurn(tool_calls=[call("bash", {"command": "printf '3' > app.py"})]),
        AssistantTurn(tool_calls=[call("bash", {"command": "printf '4' > app.py"})]),
        AssistantTurn(content="done"),
    ]
    agent, _, ui = make_agent(turns)
    asyncio.run(agent.run_turn("hi"))

    results = _bash_results(ui)
    pool = _advice_pool("app.py")
    assert_equal(results[0], "ran bash", "first write silent")
    soft = results[1].split("\n\n", 1)[1]
    assert_true(soft in pool, "second write gets a soft pool nudge")
    for i in (2, 3):
        firm = results[i].split("\n\n", 1)[1]
        assert_true(firm not in pool, "third+ write escalates beyond the soft pool")
        assert_true("STOP" in firm, "escalation is firm")
        assert_true("read" in firm and "edit" in firm,
                    "escalation names the edit tools")
    # The escalated rewrites are flagged as errors (not appended to a success).
    bash_errors = [is_err for name, _, is_err in ui.chat.tool_results if name == "bash"]
    assert_equal(bash_errors, [False, False, True, True], "3rd+ rewrites flagged is_error")


def test_rewrite_advice_excludes_previous_wording() -> None:
    pool = _advice_pool("app.py")
    for note in pool:
        for _ in range(20):
            assert_true(
                rewrite_advice_message("app.py", exclude=note) != note,
                "exclude honored",
            )


def test_rewrite_memory_persists_across_turns() -> None:
    agent, registry, ui = make_agent(
        [
            AssistantTurn(tool_calls=[call("bash", {"command": "echo a > app.py"})]),
            AssistantTurn(content="ok"),
        ]
    )
    asyncio.run(agent.run_turn("first"))
    # A new turn that rewrites the same file still gets advice — the memory of
    # written files is not reset per turn (unlike the repeat guard).
    agent._stream_assistant = _replayer(  # type: ignore[assignment]
        [
            AssistantTurn(tool_calls=[call("bash", {"command": "echo b > app.py"})]),
            AssistantTurn(content="ok"),
        ]
    )
    asyncio.run(agent.run_turn("second"))

    results = _bash_results(ui)
    assert_equal(results[0], "ran bash", "first-turn write silent")
    assert_true(results[1].startswith("ran bash\n\n"), "second-turn rewrite advised")


def test_authored_write_targets_selects_authored_files() -> None:
    assert_equal(_authored_write_targets("cat > app.py <<EOF\nx\nEOF"), ["app.py"], "heredoc write")
    assert_equal(_authored_write_targets("echo a >> notes.py"), ["notes.py"], "append write")
    assert_equal(_authored_write_targets("prog > /tmp/out.txt"), [], "scratch /tmp skipped")
    assert_equal(_authored_write_targets("ls -la && cat app.py"), [], "read-only command")
    assert_equal(_authored_write_targets("echo a > f.py && echo b >> f.py"), ["f.py"], "deduped")


def test_build_auto_read_block_shows_anchors_for_written_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            tools = HashlineTools()
            block = build_auto_read_block(tools, "cat > app.py <<EOF\n...\nEOF")
            assert_true("You just wrote `app.py`" in block, f"names the file: {block}")
            assert_true("|def f():" in block, f"includes anchor lines: {block}")
            # No anchors for non-write commands, missing files, or empty writes.
            assert_equal(build_auto_read_block(tools, "ls -la"), "", "non-write → empty")
            assert_equal(build_auto_read_block(tools, "cat > gone.py <<EOF\nEOF"), "", "missing file → empty")
        finally:
            os.chdir(cwd)


def test_build_auto_read_block_caps_file_count() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            for name in ("a.py", "b.py", "c.py", "d.py"):
                Path(name).write_text("x = 1\n", encoding="utf-8")
            tools = HashlineTools()
            command = "echo x > a.py && echo x > b.py && echo x > c.py && echo x > d.py"
            block = build_auto_read_block(tools, command)
            assert_true("`a.py`" in block and "`c.py`" in block, "first targets shown")
            assert_true("`d.py`" not in block, "fourth target capped out")
            assert_true("+1 more written file" in block, f"notes the overflow: {block}")
        finally:
            os.chdir(cwd)


def test_bash_write_appends_anchors_to_result() -> None:
    # Drive the real run_turn: a write-target command's result should carry the
    # freshly-written file's anchors so the model can edit without a read.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            turns = [
                AssistantTurn(tool_calls=[call("bash", {"command": "cat > app.py <<EOF\n...\nEOF"})]),
                AssistantTurn(content="done"),
            ]
            agent, registry, ui = make_agent(turns)
            registry.files = HashlineTools()  # the fake registry gains a real files tool
            asyncio.run(agent.run_turn("hi"))
            result = _bash_results(ui)[0]
            assert_true(result.startswith("ran bash\n\n"), f"anchors appended after result: {result}")
            assert_true("You just wrote `app.py`" in result, f"anchor block present: {result}")
            assert_true("|def f():" in result, "anchor lines present")
        finally:
            os.chdir(cwd)


def test_auto_read_suppressed_while_anchors_current_reshown_when_changed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            tools = HashlineTools()
            cmd = "cat > app.py <<EOF\n...\nEOF"
            first = build_auto_read_block(tools, cmd)
            assert_true("You just wrote `app.py`" in first, "first write shows anchors")
            # File unchanged since → model still holds current anchors → suppress.
            assert_equal(build_auto_read_block(tools, cmd), "", "suppressed while anchors current")
            # File changed out from under the model (external editor / other process)
            # → anchors are stale → re-show fresh ones.
            Path("app.py").write_text("def f():\n    return 99\n", encoding="utf-8")
            again = build_auto_read_block(tools, cmd)
            assert_true("You just wrote `app.py`" in again, "re-shown after external change")
            assert_true("|    return 99" in again, "re-shown anchors reflect new content")
        finally:
            os.chdir(cwd)


def test_model_sees_raw_tool_result_not_the_rendered_view() -> None:
    # The model must receive the RAW tool result; the file-view gutter, the
    # "Model also received" preview label, badges and ANSI exist only in the TUI.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            turns = [
                AssistantTurn(tool_calls=[call("bash", {"command": "cat > app.py <<EOF\nx\nEOF"})]),
                AssistantTurn(content="done"),
            ]
            agent, registry, ui = make_agent(turns)
            registry.files = HashlineTools()
            asyncio.run(agent.run_turn("hi"))

            # agent.py appends ONE `result` to both the model history and the UI, so
            # the model-facing content is byte-identical to what the renderer was given.
            tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
            model_content = tool_msgs[-1]["content"]
            _ui_name, ui_result, _is_err = ui.chat.tool_results[-1]
            assert_equal(model_content, ui_result, "model content == the UI's source string")

            # The model sees the raw advisory text and raw pipe anchors...
            assert_true("You just wrote `app.py`" in model_content, "advisory text is model-facing")
            assert_true(re.search(r"\d+:[0-9a-f]{2}\|def f", model_content), "raw pipe anchors present")
            # ...and NONE of the render-only decoration.
            assert_true("Model also received" not in model_content, "preview label must not reach the model")
            assert_true(">_" not in model_content and "<>" not in model_content, "badges must not reach the model")
            assert_true("\x1b[" not in model_content, "no ANSI in the model content")
            assert_true(re.search(r"\d+:[0-9a-f]{2}  ", model_content) is None, "no file-view gutter in the model content")
        finally:
            os.chdir(cwd)


def test_bash_syntax_warning_is_errors_only_never_ok() -> None:
    # Positive "OK" confirmations are for code files only; a clean bash command
    # gets nothing (its validity is not worth a line), only a real error does.
    assert_equal(_bash_syntax_warning({"command": "ls -la && echo hi"}), "", "clean bash → no block")
    bad = _bash_syntax_warning({"command": 'echo "unterminated'})
    assert_true("Syntax check (bash)" in bad and "issue(s)" in bad, "broken bash → issue list")
    assert_true("OK" not in bad, "bash never emits an OK confirmation")


def test_auto_read_suppressed_after_model_read_unchanged_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("util.py").write_text("x = 1\n", encoding="utf-8")
            tools = HashlineTools()
            tools.execute("read", {"file": "util.py"})  # model already has anchors
            # The file is unchanged, so a write command that does not alter it is
            # suppressed (model already holds the current anchors).
            assert_equal(
                build_auto_read_block(tools, "cat > util.py <<EOF\nx = 1\nEOF"),
                "",
                "suppressed: model already holds current anchors",
            )
        finally:
            os.chdir(cwd)


def test_attention_marker_is_stripped_and_flags_error() -> None:
    # A file tool returns a successful-but-needs-attention result with an in-band
    # marker; the loop strips it, flags is_error=True, and the marker never reaches
    # the UI or the model history.
    from tools.files.adapter import TOOL_ATTENTION_MARKER

    class MarkerRegistry:
        def __init__(self) -> None:
            self.executed: list[tuple[str, dict]] = []

        async def execute(self, name: str, arguments: dict) -> str:
            self.executed.append((name, arguments))
            return TOOL_ATTENTION_MARKER + "Edited line 1.\n\nSyntax check (python) — 1 issue(s):"

    turns = [
        AssistantTurn(tool_calls=[call("edit", {"file": "a.py", "anchor": "1:af", "content": "x"})]),
        AssistantTurn(content="done"),
    ]
    agent, _registry, ui = make_agent(turns)
    agent.registry = MarkerRegistry()
    asyncio.run(agent.run_turn("hi"))

    _name, result, is_err = ui.chat.tool_results[-1]
    assert_true(is_err, "attention marker maps to is_error=True")
    assert_true(TOOL_ATTENTION_MARKER not in result, "marker stripped from the UI result")
    assert_true(result.startswith("Edited line 1."), f"clean text preserved: {result!r}")
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert_true(TOOL_ATTENTION_MARKER not in tool_msgs[-1]["content"], "marker absent from history")


def _replayer(turns: list[AssistantTurn]):
    pending = list(turns)

    async def fake_stream() -> AssistantTurn:
        return pending.pop(0)

    return fake_stream


def _continuations(agent) -> int:
    return sum(
        1 for m in agent.messages
        if m.get("role") == "user" and m.get("content") == CONTINUATION_FEEDBACK
    )


def test_unfinished_turn_gets_continuation() -> None:
    # A no-tool-call turn that trails off is nudged to continue; the next turn ends.
    turns = [
        AssistantTurn(content="The output is wrong. Let me analyze:"),
        AssistantTurn(content="The calendar is correct now."),
    ]
    agent, _, _ = make_agent(turns)
    ok = asyncio.run(agent.run_turn("hi"))
    assert_true(ok, "turn completes")
    assert_equal(_continuations(agent), 1, "exactly one continuation nudge injected")


def test_done_turn_no_continuation() -> None:
    turns = [AssistantTurn(content="The task is complete and all tests pass.")]
    agent, _, _ = make_agent(turns)
    ok = asyncio.run(agent.run_turn("hi"))
    assert_true(ok, "turn completes")
    assert_equal(_continuations(agent), 0, "a finished turn is not nudged")


def test_continuation_is_capped() -> None:
    # A model that keeps trailing off is nudged at most CONTINUATION_MAX_NUDGES
    # times, then the turn ends — no infinite loop.
    turns = [
        AssistantTurn(content=f"Step {i}. Let me keep going:")
        for i in range(CONTINUATION_MAX_NUDGES + 3)
    ]
    agent, _, _ = make_agent(turns)
    ok = asyncio.run(agent.run_turn("hi"))
    assert_true(ok, "turn terminates despite persistent trailing-off")
    assert_equal(_continuations(agent), CONTINUATION_MAX_NUDGES, "continuation nudges capped")


def test_tool_progress_refreshes_continuation_budget() -> None:
    # An unfinished text turn, then a tool call (progress), then more unfinished
    # turns: the budget resets after the tool call so the cap is per-stall, and the
    # run still terminates.
    turns = [
        AssistantTurn(content="Let me start:"),
        AssistantTurn(tool_calls=[call("bash", {"command": "echo hi"})]),
        AssistantTurn(content="Now let me verify:"),
        AssistantTurn(content="All set, the task is complete."),
    ]
    agent, registry, _ = make_agent(turns)
    ok = asyncio.run(agent.run_turn("hi"))
    assert_true(ok, "turn completes")
    assert_equal(len(registry.executed), 1, "the tool call ran")
    assert_equal(_continuations(agent), 2, "one nudge before the tool, one after")


def main() -> None:
    test_identical_consecutive_call_is_denied()
    test_changed_arguments_clear_the_guard()
    test_third_identical_call_still_denied()
    test_denial_notes_vary_and_give_no_direction()
    test_argument_key_order_does_not_matter()
    test_invalid_json_markers_are_not_deduped()
    test_file_edits_are_not_deduped()
    test_new_user_turn_resets_guard()
    test_queued_batch_after_tool_result_is_sent_before_next_stream()
    test_queued_batch_at_final_response_continues_turn()
    test_completed_tool_history_is_compacted()
    test_invalid_json_marker_is_not_serialized_to_history()
    test_malformed_tool_call_text_gets_repair_turn()
    test_malformed_tool_call_retries_are_capped()
    test_first_full_rewrite_silent_then_advises()
    test_repeated_append_advises()
    test_scratch_redirect_never_advises()
    test_distinct_files_each_written_once_are_silent()
    test_repeated_rewrite_escalates()
    test_rewrite_advice_excludes_previous_wording()
    test_rewrite_memory_persists_across_turns()
    test_authored_write_targets_selects_authored_files()
    test_build_auto_read_block_shows_anchors_for_written_file()
    test_build_auto_read_block_caps_file_count()
    test_bash_write_appends_anchors_to_result()
    test_auto_read_suppressed_while_anchors_current_reshown_when_changed()
    test_auto_read_suppressed_after_model_read_unchanged_file()
    test_model_sees_raw_tool_result_not_the_rendered_view()
    test_bash_syntax_warning_is_errors_only_never_ok()
    test_attention_marker_is_stripped_and_flags_error()
    test_unfinished_turn_gets_continuation()
    test_done_turn_no_continuation()
    test_continuation_is_capped()
    test_tool_progress_refreshes_continuation_budget()
    print("agent guard tests passed")


if __name__ == "__main__":
    main()
