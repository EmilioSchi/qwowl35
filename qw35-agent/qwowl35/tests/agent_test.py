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
    BudgetDecision,
    TurnRunner,
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


def make_agent(
    turns: list[AssistantTurn],
    queued_batches: list[str] | None = None,
    registry=None,
):
    """Build an Agent whose _stream_assistant yields the given turns in order."""
    registry = registry if registry is not None else FakeRegistry()
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
        AssistantTurn(tool_calls=[call("replace", {"file": "a.py", "id": "1af", "content": "x"})]),
        AssistantTurn(tool_calls=[call("replace", {"file": "a.py", "id": "1af", "content": "x"})]),  # identical
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
                                "id": "2aa",
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
    assert_true("edit on m.py at 2aa" in serialized, "summary keeps locator")
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
        assert_true("`edit`" in firm and "read_file" not in firm,
                    "escalation names the edit tool and nothing absent")
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


def test_build_auto_read_block_marks_attention_on_syntax_error() -> None:
    # A bash-written file goes through the same syntax/LSP validation as an
    # edit: errors re-mark the whole block so the loop can flag is_error.
    from tools.files.adapter import TOOL_ATTENTION_MARKER

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("broken.py").write_text("def f()\n    return 1\n", encoding="utf-8")
            Path("clean.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            tools = HashlineTools()
            block = build_auto_read_block(tools, "cat > broken.py <<EOF\n...\nEOF")
            assert_true(block.startswith(TOOL_ATTENTION_MARKER), "syntax error → marked block")
            assert_true("Syntax check (python" in block, f"syntax block present: {block}")
            assert_true("replace id:" in block, "error rows carry edit anchors")
            clean = build_auto_read_block(tools, "cat > clean.py <<EOF\n...\nEOF")
            assert_true(not clean.startswith(TOOL_ATTENTION_MARKER), "clean file → unmarked")
            assert_true("You just wrote `clean.py`" in clean, "clean file still auto-read")
        finally:
            os.chdir(cwd)


def test_build_auto_read_block_marks_attention_on_any_broken_target() -> None:
    # One clean + one broken file in the same command: the joined block is marked.
    from tools.files.adapter import TOOL_ATTENTION_MARKER

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("ok.py").write_text("x = 1\n", encoding="utf-8")
            Path("bad.py").write_text("def f(:\n", encoding="utf-8")
            tools = HashlineTools()
            block = build_auto_read_block(tools, "echo x > ok.py && echo y > bad.py")
            assert_true(block.startswith(TOOL_ATTENTION_MARKER), "any broken target marks the block")
            assert_true("`ok.py`" in block and "`bad.py`" in block, "both targets auto-read")
        finally:
            os.chdir(cwd)


def test_bash_write_syntax_error_flags_result_is_error() -> None:
    # run_turn-level: a bash command writing a file with syntax errors surfaces
    # the syntax block AND flags the tool result is_error, like an edit would.
    from tools.files.adapter import TOOL_ATTENTION_MARKER

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("broken.py").write_text("def f()\n    return 1\n", encoding="utf-8")
            turns = [
                AssistantTurn(tool_calls=[call("bash", {"command": "cat > broken.py <<EOF\n...\nEOF"})]),
                AssistantTurn(content="done"),
            ]
            agent, registry, ui = make_agent(turns)
            registry.files = HashlineTools()
            asyncio.run(agent.run_turn("hi"))
            name, result, is_err = ui.chat.tool_results[0]
            assert_equal(name, "bash", "bash result inspected")
            assert_true(is_err, "written-file syntax error flags the bash result")
            assert_true("Syntax check (python" in result, f"syntax block present: {result}")
            assert_true(TOOL_ATTENTION_MARKER not in result, "marker stripped from the UI result")
            tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
            assert_true(TOOL_ATTENTION_MARKER not in tool_msgs[-1]["content"], "marker absent from history")
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


def test_build_post_write_report_clean_file_one_liner() -> None:
    # The non-hashline dialect: a clean write earns ONE confirmation line —
    # never anchor ids, never read_file (the freestyle executor does not
    # speak hashline; its content is already verbatim in context).
    from agent import build_post_write_report

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            block = build_post_write_report("cat > app.py <<EOF\n...\nEOF")
            assert_true(block.startswith("Wrote `app.py` (2 lines)."), f"one-liner: {block}")
            assert_true("Syntax check (python" in block, f"validation present: {block}")
            assert_true(": OK" in block, f"clean verdict: {block}")
            assert_true("read_file" not in block, "no hashline tool reference")
            assert_true("line ids" not in block, "no hashline preamble")
            assert_true("|def f():" not in block, "no hashline anchor rows")
            assert_equal(build_post_write_report("ls -la"), "", "non-write → empty")
            assert_equal(
                build_post_write_report("cat > gone.py <<EOF\nEOF"), "", "missing file → empty"
            )
        finally:
            os.chdir(cwd)


def test_build_post_write_report_errors_use_plain_line_rows() -> None:
    from agent import build_post_write_report
    from tools.files.adapter import TOOL_ATTENTION_MARKER

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("broken.py").write_text("def f()\n    return 1\n", encoding="utf-8")
            block = build_post_write_report("cat > broken.py <<EOF\n...\nEOF")
            assert_true(block.startswith(TOOL_ATTENTION_MARKER), "errors mark the block")
            assert_true("You just wrote `broken.py` — it has problems." in block, f"header: {block}")
            assert_true("Syntax check (python" in block, f"label present: {block}")
            assert_true("\n  line 1: def f()" in block, f"plain-numbered content row: {block}")
            assert_true("replace id:" not in block, "no hashline anchors")
            assert_true("read_file" not in block, "no hashline tool reference")
            assert_true("`edit` tool" in block, f"names the edit delegator: {block}")
            plain = build_post_write_report("cat > broken.py <<EOF\n...\nEOF", edit_hint=False)
            assert_true("`edit`" not in plain, f"report dialect drops the edit hint: {plain}")
            assert_true("issue(s)" in plain, f"report dialect keeps the issues: {plain}")
        finally:
            os.chdir(cwd)


def test_build_post_write_report_caps_file_count() -> None:
    from agent import build_post_write_report

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            for name in ("a.py", "b.py", "c.py", "d.py"):
                Path(name).write_text("x = 1\n", encoding="utf-8")
            command = "echo x > a.py && echo x > b.py && echo x > c.py && echo x > d.py"
            block = build_post_write_report(command)
            assert_true("`a.py`" in block and "`c.py`" in block, "first targets reported")
            assert_true("`d.py`" not in block, "fourth target capped out")
            assert_true("+1 more written file(s) not shown." in block, f"overflow line: {block}")
            assert_true("read_file" not in block, "overflow avoids hashline terms")
        finally:
            os.chdir(cwd)


def test_subedit_write_feedback_skips_hashline_and_flags_errors() -> None:
    # run_turn-level: with write_feedback="subedit" (the freestyle executor),
    # the bash result carries the plain report — registry.files is never needed
    # — and a broken written file still flags is_error with the marker stripped.
    from tools.files.adapter import TOOL_ATTENTION_MARKER

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            Path("broken.py").write_text("def f()\n    return 1\n", encoding="utf-8")
            turns = [
                AssistantTurn(tool_calls=[call("bash", {"command": "cat > app.py <<EOF\n...\nEOF"})]),
                AssistantTurn(tool_calls=[call("bash", {"command": "cat > broken.py <<EOF\n...\nEOF"}, index=1)]),
                AssistantTurn(content="done"),
            ]
            agent, registry, ui = make_agent(turns)
            agent.write_feedback = "subedit"
            asyncio.run(agent.run_turn("hi"))
            (name1, clean_result, clean_err), (name2, broken_result, broken_err) = (
                ui.chat.tool_results[0],
                ui.chat.tool_results[1],
            )
            assert_true("Wrote `app.py` (2 lines)." in clean_result, f"clean report: {clean_result}")
            assert_true(not clean_err, "clean write is not an error")
            assert_true("You just wrote `broken.py` — it has problems." in broken_result, f"broken report: {broken_result}")
            assert_true(broken_err, "broken write flags is_error")
            for result in (clean_result, broken_result):
                assert_true(TOOL_ATTENTION_MARKER not in result, "marker stripped from UI result")
                assert_true("read_file" not in result, "no hashline terms on this surface")
                assert_true("line ids" not in result, "no hashline preamble on this surface")
            tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
            assert_true(
                all(TOOL_ATTENTION_MARKER not in m["content"] for m in tool_msgs),
                "marker absent from history",
            )
        finally:
            os.chdir(cwd)


def test_post_write_report_dedups_repeat_rows_per_conversation() -> None:
    # The freestyle conversation is ONE context: re-writing the same broken
    # file lists its rows once; the repeat keeps the flag and the honest
    # count but not the rows. A conversation reset re-arms everything.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("broken.py").write_text("def f()\n    return 1\n", encoding="utf-8")
            # Two DISTINCT commands (the consecutive-repeat guard would deny an
            # identical re-run) writing the SAME broken content, so the second
            # validation reports the exact rows the first already showed.
            turns = [
                AssistantTurn(tool_calls=[call("bash", {"command": "cat > broken.py <<EOF\ndef f(\nEOF"})]),
                AssistantTurn(tool_calls=[call("bash", {"command": "cat >broken.py <<EOF\ndef f(\nEOF"}, index=1)]),
                AssistantTurn(content="done"),
            ]
            agent, registry, ui = make_agent(turns)
            agent.write_feedback = "subedit"
            # Real registries carry the conversation's diagnostics memory; the
            # FakeRegistry does not, so attach one (its absence = no dedup).
            from tools.diagnostics import DiagnosticsMemory

            registry.diag_memory = DiagnosticsMemory()
            asyncio.run(agent.run_turn("hi"))
            (_n1, first, first_err), (_n2, second, second_err) = (
                ui.chat.tool_results[0],
                ui.chat.tool_results[1],
            )
            assert_true("- line 1" in first, f"first report lists the row: {first}")
            assert_true(first_err and second_err, "both broken writes flag is_error")
            assert_true(
                "it still has problems" in second
                and "all unchanged and already reported above" in second,
                f"repeat collapses to the honest one-liner: {second}",
            )
            assert_true("- line 1" not in second, f"row not repeated: {second}")
            # A conversation reset (the /clear seam) forgets the seen rows.
            agent.reset_conversation_guards()
            sifted = registry.diag_memory.sift(
                "broken.py",
                type("V", (), {"errors": [(1, 1, "line 1, col 1: x")], "warnings": []})(),
                "def f()\n",
            )
            assert_true(sifted.errors, "cleared conversation sees rows as new again")
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
            assert_true(re.search(r"\d+[0-9a-f]{2}\|def f", model_content), "raw pipe ids present")
            # ...and NONE of the render-only decoration.
            assert_true("Model also received" not in model_content, "preview label must not reach the model")
            # Badge words are render-only; they must not appear as standalone
            # badge rows (word + two spaces at line start) in model content.
            assert_true(not re.search(r"^(?:Sh|Read|View|Grep|Glob|List|Add|Del|Edit|Fetch|Search|Plan|Ask|Expl|Resm)  ", model_content, re.MULTILINE),
                        "badges must not reach the model")
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
            tools.execute("read_file", {"file_path": os.path.abspath("util.py")})  # model already has ids
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
        AssistantTurn(tool_calls=[call("replace", {"file": "a.py", "id": "1af", "content": "x"})]),
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


def test_interleaved_identical_calls_are_still_denied() -> None:
    # Regression: a planner looped identical `plan` calls interleaved
    # with ask_user_question; the old single-slot guard was cleared by the
    # interleaving. The guard is per tool name now.
    plan_args = {"plan": "p", "todos": ["x"]}
    turns = [
        AssistantTurn(tool_calls=[call("plan", plan_args)]),
        AssistantTurn(tool_calls=[call("ask_user_question", {"questions": []})]),
        AssistantTurn(tool_calls=[call("plan", plan_args)]),  # identical → denied
        AssistantTurn(content="done"),
    ]
    agent, registry, ui = make_agent(turns)
    asyncio.run(agent.run_turn("hi"))

    executed_names = [name for name, _ in registry.executed]
    assert_equal(
        executed_names,
        ["plan", "ask_user_question"],
        "the laundered repeat did not execute",
    )
    denied = [r for n, r, err in ui.chat.tool_results if n == "plan" and err]
    assert_equal(len(denied), 1, "the repeat was denied")
    assert_true(denied[0] in _denial_pool("plan"), "denial from the note pool")


def test_max_rounds_stops_the_loop_cleanly() -> None:
    # A stage round budget is a machine transition: after max_rounds streams
    # the loop returns True with whatever was done; the remaining script is
    # never consumed.
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": f"echo {i}"})])
        for i in range(10)
    ]
    agent, registry, ui = make_agent(turns)
    agent.max_rounds = 3
    ok = asyncio.run(agent.run_turn("hi"))

    assert_true(ok, "budgeted stop is a success, not an error")
    assert_equal(len(registry.executed), 3, "exactly max_rounds streams ran")
    assert_equal(len(agent.stream_messages), 3, "no further streams requested")


def test_budget_stop_decision_matches_default_behavior() -> None:
    # An explicit "stop" decision from on_round_budget_reached must behave
    # exactly like today's silent cutoff — just with the driver asked once.
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": f"echo {i}"})])
        for i in range(5)
    ]
    agent, registry, ui = make_agent(turns)
    agent.max_rounds = 3
    asked = {"n": 0}

    async def on_budget_reached() -> BudgetDecision:
        asked["n"] += 1
        return BudgetDecision(kind="stop")

    agent.on_round_budget_reached = on_budget_reached
    ok = asyncio.run(agent.run_turn("hi"))

    assert_true(ok, "user-chosen stop is a success, not an error")
    assert_equal(len(registry.executed), 3, "loop stopped at the original budget")
    assert_equal(asked["n"], 1, "the driver was asked exactly once")


def test_budget_grow_decision_raises_max_rounds_and_continues() -> None:
    # "grow" raises max_rounds and the loop keeps running — a second budget
    # hit at the grown ceiling asks the driver again.
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": f"echo {i}"})])
        for i in range(6)
    ]
    agent, registry, ui = make_agent(turns)
    agent.max_rounds = 3
    grown = {"done": False}

    async def on_budget_reached() -> BudgetDecision:
        if grown["done"]:
            return BudgetDecision(kind="stop")
        grown["done"] = True
        return BudgetDecision(kind="grow", max_rounds=5)

    agent.on_round_budget_reached = on_budget_reached
    ok = asyncio.run(agent.run_turn("hi"))

    assert_true(ok, "budgeted stop after growth is still a success")
    assert_equal(agent.max_rounds, 5, "max_rounds was raised to the grown value")
    assert_equal(len(registry.executed), 5, "loop ran to the grown budget, not the original")


def test_budget_force_decision_forces_a_tool_and_asks_only_once() -> None:
    # "force" keeps the loop going past the budget instead of stopping; once
    # the forced tool actually satisfies stop_when, the loop ends cleanly and
    # the driver was never asked a second time.
    state = {"done": False}

    class ResumeRegistry(FakeRegistry):
        async def execute(self, name, arguments):
            if name == "resume":
                state["done"] = True
            return await super().execute(name, arguments)

    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": f"echo {i}"})])
        for i in range(3)
    ] + [AssistantTurn(tool_calls=[call("resume", {"summary": "done"})])]
    agent, registry, ui = make_agent(turns, registry=ResumeRegistry())
    agent.max_rounds = 3
    asked = {"n": 0}

    async def on_budget_reached() -> BudgetDecision:
        asked["n"] += 1
        return BudgetDecision(kind="force", forced_tool="resume")

    agent.on_round_budget_reached = on_budget_reached
    agent.stop_when = lambda: state["done"]
    ok = asyncio.run(agent.run_turn("hi"))

    assert_true(ok, "forced resume ends the loop cleanly")
    assert_equal(asked["n"], 1, "the driver is asked exactly once")
    assert_equal(len(agent.stream_messages), 4, "one extra forced stream ran")
    assert_equal(
        [name for name, _ in registry.executed][-1], "resume", "the forced call executed"
    )


def test_pending_tool_choice_is_sent_once_and_cleared() -> None:
    # The actual wire-level mechanism behind "force": a pending tool_choice
    # reaches exactly the next stream_chat call and is cleared afterward —
    # this is what makes qw35-server seed <tool_call><function=resume> via
    # its existing forced_call_prefix machinery (no server changes needed).
    class FakeStreamClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def stream_chat(self, **kwargs):
            self.calls.append(kwargs)

            async def _empty():
                return
                yield  # pragma: no cover - never reached, makes this a generator

            return _empty()

    class FakeConfig:
        def gen_params(self) -> dict:
            return {}

    class FakeStreamRegistry:
        def schemas(self) -> list:
            return []

    runner = TurnRunner.__new__(TurnRunner)
    runner.client = FakeStreamClient()
    runner.registry = FakeStreamRegistry()
    runner.config = FakeConfig()
    runner.ui = FakeUI()
    runner.messages = []
    runner.request_overrides = None
    runner._pending_tool_choice = {"type": "function", "function": {"name": "resume"}}

    asyncio.run(runner._stream_assistant())
    asyncio.run(runner._stream_assistant())

    calls = runner.client.calls
    assert_equal(len(calls), 2, "two streams were issued")
    assert_equal(
        calls[0].get("tool_choice"),
        {"type": "function", "function": {"name": "resume"}},
        "the forced tool_choice reached the wire params on the first stream",
    )
    assert_true(
        "tool_choice" not in calls[1],
        "the forced tool_choice is one-shot — cleared after the first stream",
    )


def test_same_command_after_other_work_is_allowed() -> None:
    # The shell guard is strictly consecutive: re-running the same command
    # after ANY other call executed (edit -> re-run the tests) is
    # legitimate; only an immediate identical repeat is denied.
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": "pytest -q"})]),
        AssistantTurn(
            tool_calls=[
                call("edit", {"filename": "a.py", "line_ranges": "1", "instructions": "fix"})
            ]
        ),
        AssistantTurn(tool_calls=[call("bash", {"command": "pytest -q"})]),  # re-run: allowed
        AssistantTurn(tool_calls=[call("bash", {"command": "pytest -q"})]),  # immediate: denied
        AssistantTurn(content="done"),
    ]
    agent, registry, ui = make_agent(turns)
    asyncio.run(agent.run_turn("hi"))

    executed = [name for name, _ in registry.executed]
    assert_equal(executed, ["bash", "edit", "bash"], "the re-run after the edit executed")
    denied = [r for n, r, err in ui.chat.tool_results if n == "bash" and err]
    assert_equal(len(denied), 1, "only the immediate repeat was denied")


def test_stop_when_predicate_ends_loop_cleanly() -> None:
    # The driver's state predicate ends the loop right after the turn that
    # made it true; the rest of the script is never consumed.
    state = {"done": False}

    class FlagRegistry(FakeRegistry):
        async def execute(self, name, arguments):
            if arguments.get("command") == "make done":
                state["done"] = True
            return await super().execute(name, arguments)

    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": "prep"})]),
        AssistantTurn(tool_calls=[call("bash", {"command": "make done"})]),
        AssistantTurn(tool_calls=[call("bash", {"command": "never runs"})]),
    ]
    agent, registry, ui = make_agent(turns)
    flag_registry = FlagRegistry()
    agent.registry = flag_registry
    agent.stop_when = lambda: state["done"]
    ok = asyncio.run(agent.run_turn("hi"))

    assert_true(ok, "predicate stop is a success")
    assert_equal(
        [args["command"] for _, args in flag_registry.executed],
        ["prep", "make done"],
        "loop ended at the turn that satisfied the predicate",
    )
    assert_equal(len(agent.stream_messages), 2, "no further streams requested")


def test_stop_on_stall_ends_loop_after_denied_only_turn() -> None:
    # After real work, a turn of nothing but repeat-denials is a stall: the
    # loop transitions instead of spinning (and never consumes the rest of
    # the script).
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": "echo hi"})]),
        AssistantTurn(tool_calls=[call("bash", {"command": "echo hi"})]),  # denied
        AssistantTurn(tool_calls=[call("bash", {"command": "echo other"})]),  # never reached
    ]
    agent, registry, ui = make_agent(turns)
    agent.stop_on_stall = True
    ok = asyncio.run(agent.run_turn("hi"))

    assert_true(ok, "stall exit is a success")
    assert_equal(len(registry.executed), 1, "only the first call executed")
    assert_equal(len(agent.stream_messages), 2, "loop ended at the denied turn")


def test_stop_on_stall_skips_continuation_nudge_after_work() -> None:
    # Once a tool has run, an unfinished-looking closing text ends the loop
    # instead of drawing a continuation nudge; before any work, the nudge
    # still applies (a planner's opening narration is legitimate).
    turns = [
        AssistantTurn(content="First I will analyze the task:"),  # nudged (no work yet)
        AssistantTurn(tool_calls=[call("bash", {"command": "echo hi"})]),
        AssistantTurn(content="Now let me implement the solution:"),  # would nudge; stalls instead
        AssistantTurn(content="never reached"),
    ]
    agent, registry, ui = make_agent(turns)
    agent.stop_on_stall = True
    ok = asyncio.run(agent.run_turn("hi"))

    assert_true(ok, "loop ended cleanly")
    assert_equal(len(agent.stream_messages), 3, "one nudge before work, none after")
    nudges = [
        m for m in agent.messages
        if m["role"] == "user" and "You ended your turn without calling a tool" in str(m["content"])
    ]
    assert_equal(len(nudges), 1, "only the pre-work narration was nudged")


def test_out_of_stage_tool_call_is_denied_without_executing() -> None:
    # Stage discipline (smart mode): a call outside allowed_tools is answered
    # with an error tool-result and never reaches the registry. None (the
    # freestyle default) is covered by every other test in this file.
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": "ls"})]),
        AssistantTurn(content="done"),
    ]
    agent, registry, ui = make_agent(turns)
    agent.allowed_tools = frozenset({"edit", "insert"})
    ok = asyncio.run(agent.run_turn("hi"))

    assert_true(ok, "turn completes")
    assert_equal(registry.executed, [], "out-of-stage call never executes")
    name, result, is_err = ui.chat.tool_results[-1]
    assert_true(is_err, "stage violation flagged as error")
    assert_true("not available in this stage" in result, f"violation text: {result!r}")
    assert_true("edit" in result and "insert" in result, "denial lists the stage's tools")
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert_equal(len(tool_msgs), 1, "violation fed back as a tool result")


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


def test_appended_tool_message_is_compressed() -> None:
    # End to end through a REAL registry: the {"role": "tool"} message the model
    # sees is the compressed text, and the UI card shows exactly the same string.
    from tools_registry import ToolRegistry

    command = 'for i in $(seq 1 300); do echo "the same log line over and over"; done'
    turns = [
        AssistantTurn(tool_calls=[call("bash", {"command": command})]),
        AssistantTurn(content="done"),
    ]
    agent, _, ui = make_agent(turns, registry=ToolRegistry())
    asyncio.run(agent.run_turn("hi"))

    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert_equal(len(tool_messages), 1, "one tool result appended")
    content = tool_messages[0]["content"]
    assert_true("[compressed:" in content, f"appended message compressed: {content[-200:]!r}")
    assert_true("repeated × 300" in content, "repeat count in appended message")
    shown = [r for name, r, _ in ui.chat.tool_results if name == "bash"]
    assert_equal(len(shown), 1, "one tool card shown")
    assert_equal(shown[0], content, "UI card and model message are the same string")


def test_inspect_file_lsp_section_reaches_model_verbatim() -> None:
    # The LSP diagnostics block appended by inspect_file is model-facing
    # context, not TUI decoration: the exact registry result string must land
    # in the next request's tool message (and the chat view shows the same).
    section = (
        "def f():\n    return 1\n\n"
        "LSP diagnostics (python, lsp) — 1 error(s), 1 warning(s):\n"
        "- line 2, col 12: [undefined-variable] Undefined variable 'x' (pylint)\n"
        "Warnings (not blocking):\n"
        "- line 1, col 1: [unused-import] Unused import os (pylint)"
    )

    class SectionRegistry(FakeRegistry):
        async def execute(self, name: str, arguments: dict) -> str:
            self.executed.append((name, arguments))
            return section

    turns = [
        AssistantTurn(tool_calls=[call("inspect_file", {"file_path": "/w/f.py"})]),
        AssistantTurn(content="done"),
    ]
    agent, registry, ui = make_agent(turns, registry=SectionRegistry())
    asyncio.run(agent.run_turn("read it"))

    # stream_messages[1] is the request payload for the model call AFTER the
    # tool ran — exactly what the server (and so the model) receives.
    sent = agent.stream_messages[1]
    tool_msgs = [m for m in sent if m.get("role") == "tool"]
    assert_equal(len(tool_msgs), 1, "one tool message in the model request")
    assert_equal(
        tool_msgs[0]["content"], section, "model receives the LSP section verbatim"
    )
    assert_equal(
        ui.chat.tool_results,
        [("inspect_file", section, False)],
        "chat view shows the same string it sent to the model",
    )


def test_last_reasoning_tracks_issuing_turn_mid_dispatch() -> None:
    # The orchestrator reads runner.last_reasoning while a tool call executes
    # (the editor spawn's background block); it must be the reasoning of the
    # SAME turn that issued the call, refreshed before each round's dispatch,
    # and reset with the turn guards.
    class SnappingRegistry(FakeRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.agent = None
            self.seen: list[str] = []

        async def execute(self, name: str, arguments: dict) -> str:
            self.seen.append(self.agent.last_reasoning)
            return await super().execute(name, arguments)

    turns = [
        AssistantTurn(reasoning="think-one", tool_calls=[call("bash", {"command": "ls"})]),
        AssistantTurn(reasoning="think-two", tool_calls=[call("bash", {"command": "pwd"})]),
        AssistantTurn(content="done"),
    ]
    snapping = SnappingRegistry()
    agent, _registry, _ui = make_agent(turns, registry=snapping)
    snapping.agent = agent
    asyncio.run(agent.run_turn("hi"))

    assert_equal(
        snapping.seen, ["think-one", "think-two"],
        "each dispatch sees its issuing turn's reasoning",
    )
    assert_equal(agent.last_reasoning, "", "reasoning-free closing turn clears it")
    agent.last_reasoning = "stale"
    agent.reset_turn_guards()
    assert_equal(agent.last_reasoning, "", "reset_turn_guards clears the capture")


# --- coalescing parallel edit calls into one batch --------------------------


class FakeBatchRegistry(FakeRegistry):
    """A registry that also records batch dispatches, so tests can tell the
    coalesced path from the per-call path."""

    def __init__(self) -> None:
        super().__init__()
        self.batched: list[list[tuple[str, dict]]] = []

    async def execute_batch(self, ops: list[tuple[str, dict]]) -> list[str]:
        self.batched.append([(name, args) for name, args in ops])
        return [f"batched {name}" for name, _ in ops]


def test_parallel_edits_coalesced_into_one_batch() -> None:
    # Several edit calls on the same file in ONE turn go through execute_batch
    # once (one write/diff/syntax pass), not per-call execute.
    turns = [
        AssistantTurn(tool_calls=[
            call("replace", {"file": "a.py", "id": "1af", "content": "x"}, index=0),
            call("replace", {"file": "a.py", "id": "2bd", "content": "y"}, index=1),
            call("replace", {"file": "a.py", "id": "3ce", "content": "z"}, index=2),
        ]),
        AssistantTurn(content="done"),
    ]
    reg = FakeBatchRegistry()
    agent, _reg, ui = make_agent(turns, registry=reg)
    asyncio.run(agent.run_turn("hi"))

    assert_equal(len(reg.batched), 1, "exactly one batch dispatched")
    assert_equal(
        [name for name, _ in reg.batched[0]], ["replace", "replace", "replace"],
        "all three edits in the batch",
    )
    assert_equal(reg.executed, [], "no per-op execute calls")
    assert_equal(len(ui.chat.tool_results), 3, "one tool result per call id")


def test_single_edit_uses_single_call_path() -> None:
    # A lone edit never reaches execute_batch — identical to today's behavior.
    turns = [
        AssistantTurn(tool_calls=[call("replace", {"file": "a.py", "id": "1af", "content": "x"})]),
        AssistantTurn(content="done"),
    ]
    reg = FakeBatchRegistry()
    agent, _reg, _ui = make_agent(turns, registry=reg)
    asyncio.run(agent.run_turn("hi"))

    assert_equal(reg.batched, [], "no batch for a lone edit")
    assert_equal(
        reg.executed, [("replace", {"file": "a.py", "id": "1af", "content": "x"})],
        "single-call path used",
    )


def test_edits_split_by_file_into_separate_groups() -> None:
    # Contiguous same-file edits coalesce; a different-file edit is its own unit.
    turns = [
        AssistantTurn(tool_calls=[
            call("replace", {"file": "a.py", "id": "1af", "content": "x"}, index=0),
            call("replace", {"file": "a.py", "id": "2bd", "content": "y"}, index=1),
            call("replace", {"file": "b.py", "id": "1aa", "content": "z"}, index=2),
        ]),
        AssistantTurn(content="done"),
    ]
    reg = FakeBatchRegistry()
    agent, _reg, _ui = make_agent(turns, registry=reg)
    asyncio.run(agent.run_turn("hi"))

    assert_equal(len(reg.batched), 1, "one batch (a.py's two edits)")
    assert_equal(len(reg.batched[0]), 2, "a.py batch has two ops")
    assert_equal(
        reg.executed, [("replace", {"file": "b.py", "id": "1aa", "content": "z"})],
        "the lone b.py edit runs on the single-call path",
    )


def test_non_edit_between_edits_breaks_the_group() -> None:
    # A non-edit call between two edits ends the contiguous run, so nothing
    # coalesces (each unit is size 1).
    turns = [
        AssistantTurn(tool_calls=[
            call("replace", {"file": "a.py", "id": "1af", "content": "x"}, index=0),
            call("read_file", {"file_path": "/tmp/a.py"}, index=1),
            call("replace", {"file": "a.py", "id": "2bd", "content": "y"}, index=2),
        ]),
        AssistantTurn(content="done"),
    ]
    reg = FakeBatchRegistry()
    agent, _reg, _ui = make_agent(turns, registry=reg)
    asyncio.run(agent.run_turn("hi"))

    assert_equal(reg.batched, [], "no coalescing across a non-edit call")
    assert_equal(len(reg.executed), 3, "all three run on the single-call path")


def test_registry_without_execute_batch_falls_back() -> None:
    # A registry lacking execute_batch (e.g. the planner) runs edits one by one.
    turns = [
        AssistantTurn(tool_calls=[
            call("replace", {"file": "a.py", "id": "1af", "content": "x"}, index=0),
            call("replace", {"file": "a.py", "id": "2bd", "content": "y"}, index=1),
        ]),
        AssistantTurn(content="done"),
    ]
    reg = FakeRegistry()  # no execute_batch attribute
    agent, _reg, _ui = make_agent(turns, registry=reg)
    asyncio.run(agent.run_turn("hi"))

    assert_equal(len(reg.executed), 2, "both edits run sequentially without a batch path")


def main() -> None:
    test_inspect_file_lsp_section_reaches_model_verbatim()
    test_last_reasoning_tracks_issuing_turn_mid_dispatch()
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
    test_build_auto_read_block_marks_attention_on_syntax_error()
    test_build_auto_read_block_marks_attention_on_any_broken_target()
    test_bash_write_syntax_error_flags_result_is_error()
    test_auto_read_suppressed_while_anchors_current_reshown_when_changed()
    test_auto_read_suppressed_after_model_read_unchanged_file()
    test_build_post_write_report_clean_file_one_liner()
    test_build_post_write_report_errors_use_plain_line_rows()
    test_build_post_write_report_caps_file_count()
    test_subedit_write_feedback_skips_hashline_and_flags_errors()
    test_post_write_report_dedups_repeat_rows_per_conversation()
    test_model_sees_raw_tool_result_not_the_rendered_view()
    test_bash_syntax_warning_is_errors_only_never_ok()
    test_attention_marker_is_stripped_and_flags_error()
    test_unfinished_turn_gets_continuation()
    test_done_turn_no_continuation()
    test_continuation_is_capped()
    test_tool_progress_refreshes_continuation_budget()
    test_interleaved_identical_calls_are_still_denied()
    test_max_rounds_stops_the_loop_cleanly()
    test_budget_stop_decision_matches_default_behavior()
    test_budget_grow_decision_raises_max_rounds_and_continues()
    test_budget_force_decision_forces_a_tool_and_asks_only_once()
    test_pending_tool_choice_is_sent_once_and_cleared()
    test_same_command_after_other_work_is_allowed()
    test_stop_when_predicate_ends_loop_cleanly()
    test_stop_on_stall_ends_loop_after_denied_only_turn()
    test_stop_on_stall_skips_continuation_nudge_after_work()
    test_out_of_stage_tool_call_is_denied_without_executing()
    test_appended_tool_message_is_compressed()
    test_parallel_edits_coalesced_into_one_batch()
    test_single_edit_uses_single_call_path()
    test_edits_split_by_file_into_separate_groups()
    test_non_edit_between_edits_breaks_the_group()
    test_registry_without_execute_batch_falls_back()
    print("agent guard tests passed")


if __name__ == "__main__":
    main()
