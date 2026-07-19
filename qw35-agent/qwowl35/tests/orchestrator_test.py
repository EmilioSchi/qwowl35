"""Tests for the mode-dispatch orchestrator: one agent per user-selected TUI
mode, strict per-stage tool segregation, the planner's stateless explore/resume
sub-agent, persistent executor context inheritance, and the editor sub-loop.

Run directly: ``python qwowl35/tests/orchestrator_test.py``. No network: the
model is scripted by patching TurnRunner._stream_assistant at class level, so
every runner the orchestrator spawns (stages, explorer, editor) consumes the
same ordered script of AssistantTurns.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent as agent_mod  # noqa: E402
from agent import BudgetDecision, TurnRunner  # noqa: E402
from agents.pipeline import ExplorerRegistry, PipelineRegistry  # noqa: E402
from agents.spawn_context import BACKGROUND_HEADER  # noqa: E402
from client import AssistantTurn, Qw35Error, ToolCall  # noqa: E402
from config import Config  # noqa: E402
from modes import Mode  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from tools.plan import todo_ref  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


class FakeChat:
    def __init__(self) -> None:
        self.tool_results: list[tuple[str, str, bool]] = []
        self.users: list[str] = []
        self.system_notes: list[str] = []

    def add_user(self, text: str) -> None:
        self.users.append(text)

    def flush_reasoning(self) -> None: ...
    def flush_assistant(self) -> None: ...

    def add_tool_result(self, index, name, result, is_error=False) -> None:
        self.tool_results.append((name, result, is_error))

    def add_system(self, text: str) -> None:
        self.system_notes.append(text)


class FakeUI:
    def __init__(self) -> None:
        self.chat = FakeChat()
        self.modes: list[Mode] = []
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

    def set_mode(self, mode: Mode) -> None:
        self.modes.append(mode)

    def pop_queued_user_batch(self) -> str | None:
        return None


def call(name: str, arguments: dict, index: int = 0) -> ToolCall:
    return ToolCall(id=f"c{index}", name=name, arguments=arguments, index=index)


class Script:
    """Scripted model: each _stream_assistant call pops the next turn and
    snapshots the exact messages (and wire tools) that request would send."""

    def __init__(self, turns: list[AssistantTurn], eval_count: int = 0) -> None:
        self.pending = list(turns)
        self.requests: list[dict] = []
        self.eval_count = eval_count

    def install(self):
        script = self

        async def fake_stream(runner_self) -> AssistantTurn:
            script.requests.append(
                {
                    "messages": json.loads(json.dumps(runner_self.messages)),
                    "tools": json.loads(json.dumps(runner_self.registry.schemas())),
                    "overrides": dict(runner_self.request_overrides or {}),
                }
            )
            runner_self.last_timings = {"eval_count": script.eval_count}
            if not script.pending:
                raise AssertionError("script exhausted: model asked for another turn")
            turn = script.pending.pop(0)
            if turn == "ERROR":  # sentinel: simulate a stream failure
                raise Qw35Error("bad_re", "simulated stream failure")
            return turn

        self._original = TurnRunner._stream_assistant
        TurnRunner._stream_assistant = fake_stream
        return self

    def uninstall(self) -> None:
        TurnRunner._stream_assistant = self._original


def run_orchestrated(
    turns: list[AssistantTurn],
    goal: str = "do the thing",
    config: Config | None = None,
    eval_count: int = 0,
    mode: Mode = Mode.NORMAL,
):
    ui = FakeUI()
    orch = Orchestrator(client=None, config=config or Config(), ui=ui)
    script = Script(turns, eval_count=eval_count).install()
    agent_mod.WAKEUP_HOLD_SECONDS = 0
    try:
        ok = asyncio.run(orch.run_turn(goal, mode))
    finally:
        script.uninstall()
    return ok, orch, ui, script


def tool_names(request: dict) -> list[str]:
    return [t["function"]["name"] for t in request["tools"]]


def system_of(request: dict) -> str:
    return request["messages"][0]["content"]


def last_user_of(request: dict) -> str:
    return [m for m in request["messages"] if m.get("role") == "user"][-1]["content"]


def exec_requests(script: Script) -> list[dict]:
    return [r for r in script.requests if tool_names(r) == ["run_shell_command", "edit"]]


def explorer_requests(script: Script) -> list[dict]:
    return [r for r in script.requests if "resume" in tool_names(r)]


PLANNER_TOOLS = ["plan", "ask_user_question", "explore"]


def planner_requests(script: Script) -> list[dict]:
    return [r for r in script.requests if tool_names(r) == PLANNER_TOOLS]


def test_rewrite_advice_never_names_absent_tools() -> None:
    # The executor toolset is run_shell_command + edit; rewrite-advice texts
    # must never name the internal hashline tools (observed live: "editing it
    # in place with `read_file`" advised to an executor that has no
    # such tool).
    from agent import (
        REWRITE_ADVICE_NOTES,
        escalated_rewrite_message,
        rewrite_advice_message,
    )

    absent = ("read_file", "insert", "delete")
    for note in REWRITE_ADVICE_NOTES:
        for tool in absent:
            assert_true(tool not in note, f"advice names absent tool {tool!r}: {note!r}")
    escalation = escalated_rewrite_message("cal.py", 3)
    for tool in absent:
        assert_true(tool not in escalation, f"escalation names absent tool {tool!r}")
    advice = rewrite_advice_message("cal.py")
    assert_true("`edit`" in advice, f"advice points at the real tool: {advice!r}")


def test_normal_mode_runs_direct_executor() -> None:
    # NORMAL mode = the freestyle executor spawned directly: one stream, no
    # pipeline stages, the executor toolset on the wire.
    turns = [AssistantTurn(content="Did the thing.")]
    ok, orch, ui, script = run_orchestrated(turns, goal="just do it")
    assert_true(ok, "turn completes")
    assert_equal(len(script.requests), 1, "exactly one stream: the executor")
    req = script.requests[0]
    assert_equal(
        tool_names(req), ["run_shell_command", "edit"],
        "the direct executor advertises the executor toolset",
    )
    assert_true("just do it" in json.dumps(req["messages"]), "goal reaches the executor")
    assert_equal(ui.modes, [Mode.NORMAL], "the mode box shows NORMAL")


def test_display_modes_clamp_to_normal_dispatch() -> None:
    # VISUAL/INSERT are display-only: run_turn defensively treats them as
    # NORMAL rather than dispatching to a sub-agent.
    turns = [AssistantTurn(content="Did the thing.")]
    ok, orch, ui, script = run_orchestrated(turns, mode=Mode.VISUAL)
    assert_true(ok, "turn completes")
    assert_equal(
        tool_names(script.requests[0]), ["run_shell_command", "edit"],
        "a display mode dispatches as NORMAL",
    )


def test_mode_dispatch_selects_the_right_agent() -> None:
    # Each user-selectable mode runs its own agent: own system prompt, own
    # wire toolset.
    cases = [
        (Mode.NORMAL, "qwowl35, a coding agent", ["run_shell_command", "edit"]),
        (Mode.PLAN, "qwowl35 planner", PLANNER_TOOLS),
        (Mode.WEB, "qwowl35 web agent", ["search_engine", "web_fetch"]),
        (Mode.CHAT, "general-purpose assistant", []),
    ]
    for mode, prompt_marker, tools in cases:
        ok, orch, ui, script = run_orchestrated(
            [AssistantTurn(content="All done here.")], mode=mode
        )
        req = script.requests[0]
        assert_true(
            prompt_marker in system_of(req),
            f"{mode.value}: system prompt is the agent's own",
        )
        assert_equal(tool_names(req), tools, f"{mode.value}: wire toolset")
        assert_equal(ui.modes[0], mode, f"{mode.value}: mode box notified")


def test_stage_toolsets_are_segregated() -> None:
    registry = PipelineRegistry()
    registry.set_stage("planner", frozenset())
    assert_equal(
        [t["function"]["name"] for t in registry.schemas()],
        PLANNER_TOOLS,
        "planner advertises the plan tools + explore",
    )
    registry.set_stage("execute", frozenset())
    assert_equal(
        [t["function"]["name"] for t in registry.schemas()],
        ["run_shell_command", "edit"],
        "execute advertises only the shell + edit",
    )
    registry.set_stage("web", frozenset())
    assert_equal(
        [t["function"]["name"] for t in registry.schemas()],
        ["search_engine", "web_fetch"],
        "web advertises only search_engine + web_fetch",
    )
    registry.set_stage("chat", frozenset())
    assert_equal(registry.schemas(), [], "chat advertises no tools")
    # The explorer sub-agent's toolset lives in its own registry.
    explorer = ExplorerRegistry(registry)
    assert_equal(
        [t["function"]["name"] for t in explorer.schemas()],
        ["list_directory", "glob", "grep_search", "inspect_file", "lsp", "run_shell_command", "resume"],
        "the explorer advertises its search tools + lsp + shell + resume",
    )


def test_begin_stage_sets_write_feedback_dialect() -> None:
    # The shared TurnRunner defaults to the NORMAL agent's hashline dialect;
    # pointing it at a stage must adopt that stage's post-write dialect, so the
    # execute stage never receives hashline anchors it cannot use.
    from agents import chat as chat_agent
    from agents import freestyle as freestyle_agent

    assert_equal(TurnRunner.write_feedback, "hashline", "class default is the NORMAL dialect")
    ui = FakeUI()
    orch = Orchestrator(client=None, config=Config(), ui=ui)
    orch._begin_stage(freestyle_agent.SPEC, [])
    assert_equal(orch.runner.write_feedback, "subedit", "execute stage speaks subedit")
    orch._begin_stage(chat_agent.SPEC, [])
    assert_equal(orch.runner.write_feedback, "report", "other stages get the plain report")


def test_explorer_shell_is_restricted_under_the_hood() -> None:
    # The explorer's run_shell_command is the same wire tool the executor
    # gets, but the ExplorerRegistry silently routes it through `bash -r`:
    # observation is possible, mutation paths (cd, redirects) are not.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            pipeline = PipelineRegistry()
            explorer = ExplorerRegistry(pipeline)
            blocked = asyncio.run(
                explorer.execute(
                    "run_shell_command",
                    {"command": "echo x > out.txt", "is_background": False},
                )
            )
            assert_true("restricted" in blocked and "Exit code" in blocked, blocked)
            assert_true(not Path("out.txt").exists(), "the write never happened")
            observe = asyncio.run(
                explorer.execute(
                    "run_shell_command", {"command": "echo observed", "is_background": False}
                )
            )
            assert_equal(observe.strip(), "observed", "observation commands still run")
            # The executor stage keeps the full shell.
            pipeline.set_stage("execute", frozenset({"run_shell_command"}))
            unrestricted = asyncio.run(
                pipeline.execute(
                    "run_shell_command",
                    {"command": "echo x > out.txt", "is_background": False},
                )
            )
            assert_true("restricted" not in unrestricted, f"execute unrestricted: {unrestricted!r}")
            assert_true(Path("out.txt").exists(), "executor's shell can write")
        finally:
            os.chdir(cwd)


def test_conversation_routes_to_chat_and_keeps_history() -> None:
    ok, orch, ui, script = run_orchestrated(
        [AssistantTurn(content="It's a prefix cache.")],
        goal="what is the session cache?",
        mode=Mode.CHAT,
    )
    assert_true(ok, "turn succeeds")
    assert_equal(
        orch.chat_messages[-1]["content"],
        "It's a prefix cache.",
        "chat answer stays in the persistent chat lineage",
    )
    chat_req = script.requests[-1]
    assert_equal(chat_req["tools"], [], "chat request advertises no tools")
    assert_true("general-purpose" in system_of(chat_req), "chat has its own prompt")
    assert_true(
        "run_shell_command" not in system_of(chat_req),
        "chat prompt never mentions the shell tool",
    )


def test_web_mode_answers_with_web_tools_only() -> None:
    ok, orch, ui, script = run_orchestrated(
        [AssistantTurn(content="Web findings: the docs say X (https://ex.am/ple).")],
        goal="look this up",
        mode=Mode.WEB,
    )
    assert_true(ok, "turn succeeds")
    req = script.requests[0]
    assert_equal(
        tool_names(req),
        ["search_engine", "web_fetch"],
        "web agent sees only the web tools",
    )
    assert_true(
        "the docs say X" in orch.turn_log[-1][1],
        "the web answer is the recorded outcome",
    )


def test_easy_task_runs_executor_with_only_bash_and_edit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            ok, orch, ui, script = run_orchestrated(
                [
                    AssistantTurn(tool_calls=[call("run_shell_command", {"command": "echo hi > out.txt", "is_background": False})]),
                    AssistantTurn(content="Created out.txt with the greeting."),
                ]
            )
        finally:
            os.chdir(cwd)
    assert_true(ok, "turn succeeds")
    for req in script.requests:
        assert_equal(tool_names(req), ["run_shell_command", "edit"], "executor sees exactly the shell + edit")
        assert_true("list_directory" not in system_of(req), "no explorer tools in its prompt")
    assert_equal(
        orch.turn_log[-1][1],
        "Created out.txt with the greeting.",
        "outcome recorded in the session log",
    )


def test_execute_stage_bash_write_gets_anchors_and_syntax_check() -> None:
    # The execute stage (the freestyle agent) creates files with the shell, so
    # its bash results must carry post-write validation with is_error flagged
    # when the written file is broken — but in ITS dialect (the plain report
    # naming the `edit` delegator, write_feedback="subedit"), never hashline
    # anchors or read_file, which only the NORMAL agent understands.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            ok, orch, ui, script = run_orchestrated(
                [
                    AssistantTurn(
                        tool_calls=[
                            call(
                                "run_shell_command",
                                {"command": "cat > broken.py <<EOF\ndef f(\nEOF", "is_background": False},
                            )
                        ]
                    ),
                    AssistantTurn(content="Wrote broken.py."),
                ]
            )
        finally:
            os.chdir(cwd)
    assert_true(ok, "turn succeeds")
    results = [(r, e) for n, r, e in ui.chat.tool_results if n == "run_shell_command"]
    assert_true(results, "shell result captured")
    result, is_err = results[0]
    assert_true("You just wrote `broken.py`" in result, f"report appended: {result}")
    assert_true("Syntax check (python" in result, f"syntax block appended: {result}")
    assert_true(is_err, "a broken written file flags the execute-stage result as an error")
    assert_true("\x00" not in result, "no attention-marker bytes leak to the UI")
    # The execute stage does not speak hashline: no anchor ids, no
    # read_file — plain `line N:` rows for its `edit` delegator instead.
    assert_true("read_file" not in result, f"no hashline tool reference: {result}")
    assert_true("line ids" not in result, f"no hashline preamble: {result}")
    assert_true("replace id:" not in result, f"no hashline anchor rows: {result}")
    assert_true("`edit` tool" in result, f"names the edit delegator: {result}")


def test_editor_leaving_file_broken_reports_syntax_to_executor() -> None:
    # When the editor session ends with the file malformed, the executor must
    # receive WHAT is wrong (the syntax block), not just a bare error flag.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("hello.py").write_text('def f():\n    return "hi"\n', encoding="utf-8")
            from tools.files.adapter import HashlineTools

            probe = HashlineTools().execute(
                "read_file", {"file_path": "hello.py", "_force": True}
            )
            first_line_id = probe.splitlines()[1].split("|", 1)[0]
            turns = [
                AssistantTurn(
                    tool_calls=[
                        call(
                            "edit",
                            {"filename": "hello.py", "line_ranges": "1", "instructions": "drop the colon"},
                        )
                    ]
                ),
                AssistantTurn(
                    tool_calls=[call("replace", {"file":"hello.py", "id": first_line_id, "content": "def f()"})]
                ),
                AssistantTurn(content="Removed the colon."),
                AssistantTurn(content="Done."),
            ]
            ok, orch, ui, script = run_orchestrated(turns, goal="tweak the greeting")
        finally:
            os.chdir(cwd)
    reports = [(r, e) for n, r, e in ui.chat.tool_results if "Editor result for hello.py" in r]
    assert_true(reports, "executor received the editor report")
    report, is_err = reports[0]
    assert_true(is_err, "a still-broken file flags the executor's edit result")
    assert_true("Syntax check (python" in report, f"syntax block included: {report}")
    assert_true("issue(s)" in report, f"the error list is included: {report}")
    assert_true("\x00" not in report, "no marker bytes leak to the UI")


def test_editor_spawn_diag_memory_is_fresh_and_executor_dedup_persists() -> None:
    # Diagnostics memory follows the agent seams: every editor spawn is a
    # fresh context, so its OPENING view lists the broken file's rows in full
    # both times — while the executor, whose one context receives both editor
    # reports, gets the full block once and an honest "unchanged" one-liner
    # the second time.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("broken.py").write_text(
                "a = 1\nb = 2\ndef f()\n    return 1\n", encoding="utf-8"
            )
            from tools.files.adapter import HashlineTools

            probe = HashlineTools().execute(
                "read_file", {"file_path": "broken.py", "_force": True}
            )
            id_a1 = probe.splitlines()[1].split("|", 1)[0]
            # The second editor edits the same line 1 after it became "a = 9";
            # ids are line+content hashes, so probe the future state.
            Path("probe.py").write_text(
                "a = 9\nb = 2\ndef f()\n    return 1\n", encoding="utf-8"
            )
            probe2 = HashlineTools().execute(
                "read_file", {"file_path": "probe.py", "_force": True}
            )
            id_a9 = probe2.splitlines()[1].split("|", 1)[0]
            turns = [
                AssistantTurn(
                    tool_calls=[
                        call(
                            "edit",
                            {"filename": "broken.py", "line_ranges": "1", "instructions": "bump a"},
                        )
                    ]
                ),
                AssistantTurn(
                    tool_calls=[call("replace", {"file":"broken.py", "id": id_a1, "content": "a = 9"})]
                ),
                AssistantTurn(content="Bumped a."),
                AssistantTurn(
                    tool_calls=[
                        call(
                            "edit",
                            {"filename": "broken.py", "line_ranges": "1", "instructions": "bump a again"},
                        )
                    ]
                ),
                AssistantTurn(
                    tool_calls=[call("replace", {"file":"broken.py", "id": id_a9, "content": "a = 8"})]
                ),
                AssistantTurn(content="Bumped a again."),
                AssistantTurn(content="Done."),
            ]
            ok, orch, ui, script = run_orchestrated(turns, goal="bump a twice")
        finally:
            os.chdir(cwd)
    reports = [(r, e) for n, r, e in ui.chat.tool_results if "Editor result for broken.py" in r]
    assert_equal(len(reports), 2, "both editor reports reached the executor")
    (first, first_err), (second, second_err) = reports
    assert_true(first_err and second_err, "the still-broken file flags both")
    assert_true("Syntax check (python" in first, f"first report carries the block: {first}")
    assert_true("- line 3" in first, f"first report lists the row: {first}")
    assert_true(
        "all unchanged and already reported above" in second,
        f"second report collapses for the executor: {second}",
    )
    assert_true("- line 3" not in second, f"row not repeated to the executor: {second}")
    # The first editor spawn opens with the full anchored rows; the second
    # spawn continues the same persistent conversation (its continuation
    # message is appended to the same list, not logged as a separate stage).
    # Only the first spawn is logged to avoid quadratic duplication.
    editor_stage_texts: list[str] = []
    collecting = False
    for message in orch.messages:
        content = message.get("content", "")
        if content.startswith("=== stage: "):
            collecting = content == "=== stage: editor ==="
            if collecting:
                editor_stage_texts.append("")
            continue
        if collecting and message.get("role") == "user":
            editor_stage_texts[-1] += content
    assert_equal(len(editor_stage_texts), 1, "one editor stage logged (first spawn only)")
    assert_true(
        "replace id:" in editor_stage_texts[0],
        f"first editor spawn opened with the full anchored rows: {editor_stage_texts[0][:400]}",
    )
    # The second spawn's continuation message is in THE persisted editor
    # conversation (one conversation for the whole turn, not per file).
    editor_messages = orch._editor_messages
    assert_true(editor_messages, "the editor conversation persisted")
    # System prompt + first user message + ... + continuation user message.
    user_texts = [
        m["content"] for m in editor_messages if m.get("role") == "user"
    ]
    assert_true(
        len(user_texts) >= 2,
        f"at least two user messages in persisted conversation: {len(user_texts)}",
    )
    assert_true(
        "Previous edit complete" in user_texts[-1],
        f"last user message is a continuation: {user_texts[-1][:200]}",
    )


def test_editor_conversation_spans_files() -> None:
    # The editor is ONE persistent conversation per turn, like the planner
    # and the executor: an edit to a second file continues the same context
    # with a slim continuation message instead of opening a fresh spawn.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("a.py").write_text("x = 1\n", encoding="utf-8")
            Path("b.py").write_text("y = 2\n", encoding="utf-8")
            from tools.files.adapter import HashlineTools

            probe_a = HashlineTools().execute(
                "read_file", {"file_path": "a.py", "_force": True}
            )
            id_a = probe_a.splitlines()[1].split("|", 1)[0]
            probe_b = HashlineTools().execute(
                "read_file", {"file_path": "b.py", "_force": True}
            )
            id_b = probe_b.splitlines()[1].split("|", 1)[0]
            turns = [
                AssistantTurn(
                    tool_calls=[
                        call(
                            "edit",
                            {"filename": "a.py", "line_ranges": "1", "instructions": "bump x"},
                        )
                    ]
                ),
                AssistantTurn(
                    tool_calls=[call("replace", {"file": "a.py", "id": id_a, "content": "x = 9"})]
                ),
                AssistantTurn(content="Bumped x."),
                AssistantTurn(
                    tool_calls=[
                        call(
                            "edit",
                            {"filename": "b.py", "line_ranges": "1", "instructions": "bump y"},
                        )
                    ]
                ),
                AssistantTurn(
                    tool_calls=[call("replace", {"file": "b.py", "id": id_b, "content": "y = 9"})]
                ),
                AssistantTurn(content="Bumped y."),
                AssistantTurn(content="Done."),
            ]
            ok, orch, ui, script = run_orchestrated(turns, goal="bump both")
        finally:
            os.chdir(cwd)
    assert_true(ok, "turn completed")
    editor_reqs = [r for r in script.requests if "replace" in tool_names(r)]
    assert_true(len(editor_reqs) >= 3, f"editor requests recorded: {len(editor_reqs)}")
    # The b.py spawn's first request carries the WHOLE inherited conversation:
    # one system prompt, the a.py opener, its dialogue, then the continuation.
    b_request = editor_reqs[2]
    systems = [m for m in b_request["messages"] if m.get("role") == "system"]
    assert_equal(len(systems), 1, "one system prompt in the continued conversation")
    user_texts = [m["content"] for m in b_request["messages"] if m.get("role") == "user"]
    assert_true("a.py" in user_texts[0], f"a.py opener inherited: {user_texts[0][:120]}")
    assert_true(
        "Previous edit complete" in user_texts[-1] and "b.py" in user_texts[-1],
        f"b.py lands as a continuation: {user_texts[-1][:200]}",
    )
    both = [(r, e) for n, r, e in ui.chat.tool_results if "Editor result for" in r]
    assert_equal(len(both), 2, "both editor reports reached the executor")


def test_editor_break_then_fix_does_not_flag_executor() -> None:
    # saw_attention is sticky across the session, but an interim break the
    # editor itself fixed must not surface as an error to the executor.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("hello.py").write_text('def f():\n    return "hi"\n', encoding="utf-8")
            from tools.files.adapter import HashlineTools

            probe = HashlineTools().execute(
                "read_file", {"file_path": "hello.py", "_force": True}
            )
            good_id = probe.splitlines()[1].split("|", 1)[0]
            # The broken line's id (same line number + content hash rules).
            Path("probe.py").write_text('def f()\n    return "hi"\n', encoding="utf-8")
            broken_probe = HashlineTools().execute(
                "read_file", {"file_path": "probe.py", "_force": True}
            )
            broken_id = broken_probe.splitlines()[1].split("|", 1)[0]
            turns = [
                AssistantTurn(
                    tool_calls=[
                        call(
                            "edit",
                            {"filename": "hello.py", "line_ranges": "1", "instructions": "rename f"},
                        )
                    ]
                ),
                AssistantTurn(
                    tool_calls=[call("replace", {"file":"hello.py", "id": good_id, "content": "def f()"})]
                ),
                AssistantTurn(
                    tool_calls=[call("replace", {"file":"hello.py", "id": broken_id, "content": "def g():"})]
                ),
                AssistantTurn(content="Renamed f to g."),
                AssistantTurn(content="Done."),
            ]
            ok, orch, ui, script = run_orchestrated(turns, goal="rename the function")
        finally:
            os.chdir(cwd)
    reports = [(r, e) for n, r, e in ui.chat.tool_results if "Editor result for hello.py" in r]
    assert_true(reports, "executor received the editor report")
    report, is_err = reports[0]
    assert_true(not is_err, f"an interim break the editor fixed is not an error: {report}")


def test_plan_pipeline_hands_over_data_not_conversation() -> None:
    # PLAN mode: the planner explores on demand through a STATELESS explorer
    # sub-agent, receives only its resume summary, and the executors continue
    # one persistent conversation that never contains the explorer's turns.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("lib.py").write_text("def add(a, b):\n    return a + b\n")
            turns = [
                # planner round 1: spawn the explorer.
                AssistantTurn(
                    tool_calls=[call("explore", {"task": "Read lib.py and report what add() does."})]
                ),
                # the explorer's own (interleaved) turns: search, then resume.
                AssistantTurn(
                    tool_calls=[call("inspect_file", {"file_path": str(Path(tmp) / "lib.py")})]
                ),
                AssistantTurn(
                    tool_calls=[
                        call("resume", {"summary": "lib.py holds add(); it returns a + b."})
                    ]
                ),
                # planner round 2: one unified plan call — the (auto-approved)
                # gate is atomic with it and ends planning (stop_when).
                AssistantTurn(
                    tool_calls=[
                        call(
                            "plan",
                            {
                                "plan": "1. add 2. test",
                                "todos": ["add subtract()", "test subtract()"],
                            },
                        )
                    ]
                ),
                # ping-pong: task 1, planner review (guidance, no plan call:
                # the machine fallback completes T1), task 2, final review.
                AssistantTurn(content="Added subtract() to lib.py."),
                AssistantTurn(content="Now verify it with the tests."),
                AssistantTurn(content="Tested subtract() successfully."),
                AssistantTurn(content="Goal reached."),
            ]
            ok, orch, ui, script = run_orchestrated(
                turns, goal="add subtract to lib", mode=Mode.PLAN
            )
        finally:
            os.chdir(cwd)

    assert_true(ok, "pipeline completes")
    explore_reqs = explorer_requests(script)
    plan_reqs = planner_requests(script)
    exec_reqs = exec_requests(script)
    assert_true(explore_reqs and plan_reqs and exec_reqs, "all three agents ran")

    # The explorer is STATELESS: its first request is exactly its own system
    # prompt plus the planner's task text, verbatim — no goal framing, no
    # session notes, no planner conversation.
    first = explore_reqs[0]
    assert_equal(len(first["messages"]), 2, "explorer starts with system + task only")
    assert_true("qwowl35 explorer" in system_of(first), "explorer has its own prompt")
    assert_equal(
        first["messages"][1]["content"],
        "Read lib.py and report what add() does.",
        "the planner's task string is the explorer's user prompt, verbatim",
    )
    assert_equal(
        first["overrides"].get("qw35_session"), "scratch", "explorer on the scratch session"
    )
    for req in explore_reqs:
        assert_true(
            "edit" not in tool_names(req) and "plan" not in tool_names(req),
            "no other agent's tools leak into the explorer",
        )

    # The handoff is DATA: the planner sees only the resume summary as the
    # explore call's tool result.
    summary_results = [
        m for r in plan_reqs for m in r["messages"]
        if m.get("role") == "tool" and str(m.get("content", "")).startswith("Exploration findings:")
    ]
    assert_true(summary_results, "the resume summary returns as the explore result")
    assert_true(
        "it returns a + b" in summary_results[0]["content"],
        "the summary text reaches the planner",
    )
    plan_context = json.dumps(plan_reqs[0]["messages"])
    assert_true(
        "return a + b\n" not in plan_context,
        "the explorer's raw tool output never enters the planner's context",
    )
    assert_equal(
        len(plan_reqs[0]["messages"]), 2,
        "planner starts fresh: its system prompt + one task message",
    )
    for req in plan_reqs:
        assert_true("qwowl35 planner" in system_of(req), "planner has its own prompt")
    assert_true(
        all(r["overrides"].get("qw35_session") == "plan" for r in plan_reqs),
        "planner and review pings run on the plan session",
    )

    # Executors CONTINUE one conversation: task 2's request extends task 1's
    # (full context inheritance — the plan block is not re-sent, and task 1's
    # result rides along as the conversation itself, not a handoff block).
    task1, task2 = exec_reqs[0], exec_reqs[-1]
    assert_equal(len(task1["messages"]), 2, "first executor: system + opener")
    opener = task1["messages"][1]["content"]
    assert_true("Overall goal:" in opener and "Approved plan:" in opener, "opener carries goal + plan")
    assert_true("Your task (1/2)" in opener, "task 1 scoped to one todo")
    assert_equal(
        task2["messages"][: len(task1["messages"])],
        task1["messages"],
        "task 2's context starts with task 1's as a strict prefix",
    )
    directive = task2["messages"][-1]["content"]
    assert_true("Your next task (2/2)" in directive, "task 2 gets a slim continuation directive")
    assert_true("Approved plan:" not in directive, "the plan block is not re-sent")
    assert_true(
        "Planner guidance for this task:\nNow verify it with the tests." in directive,
        "review guidance reaches the next directive",
    )
    assert_true("Already completed:" not in directive, "no completed-results block: the conversation carries it")
    assert_true(
        any(
            m.get("role") == "assistant" and "Added subtract()" in str(m.get("content", ""))
            for m in task2["messages"]
        ),
        "task 1's result is inherited as conversation",
    )
    assert_true(
        "resume" not in json.dumps([tool_names(r) for r in exec_reqs]),
        "explorer tools never reach the executors",
    )
    assert_true(
        all("inspect_file" not in json.dumps(r["messages"]) for r in exec_reqs),
        "the explorer's conversation never reaches the executors",
    )

    # Review pings carry the persistent plan context, the report, and the
    # SAME wire tools as planning (prefix stability).
    review_reqs = [
        r for r in plan_reqs
        if any("Executor report:" in str(m.get("content", "")) for m in r["messages"])
    ]
    assert_true(len(review_reqs) >= 2, "one review ping per task")
    review_text = json.dumps(review_reqs[0]["messages"])
    assert_true(
        "Current todo list:" in review_text and "[>]" in review_text,
        "the review ping shows the CURRENT rendered list (machine marks included)",
    )
    assert_equal(
        tool_names(review_reqs[-1]),
        PLANNER_TOOLS,
        "review pings keep the planner stage's wire tools (prefix stability)",
    )
    state = orch.registry.plan.state
    assert_equal(state.progress, len(state.todos), f"todos completed: {state.todos}")


def test_explore_spawns_stateless_explorer_each_time() -> None:
    # Two explore calls in one planning stage: each spawn starts with ZERO
    # history (system + task only) on the scratch session — nothing from the
    # first exploration leaks into the second.
    turns = [
        AssistantTurn(tool_calls=[call("explore", {"task": "Map the parser module."})]),
        AssistantTurn(tool_calls=[call("resume", {"summary": "parser.py: one class."})]),
        AssistantTurn(tool_calls=[call("explore", {"task": "Map the lexer module."})]),
        AssistantTurn(tool_calls=[call("resume", {"summary": "lexer.py: two functions."})]),
        AssistantTurn(content="I could not finish planning."),
    ]
    ok, orch, ui, script = run_orchestrated(turns, goal="map it", mode=Mode.PLAN)
    explore_reqs = explorer_requests(script)
    assert_equal(len(explore_reqs), 2, "two explorer spawns, one stream each")
    for i, req in enumerate(explore_reqs):
        assert_equal(len(req["messages"]), 2, f"spawn {i + 1} starts clean: system + task")
        assert_equal(
            req["overrides"], {"qw35_session": "scratch"},
            f"spawn {i + 1} rides the scratch session",
        )
        names = tool_names(req)
        assert_true("resume" in names, f"spawn {i + 1} can resume")
        assert_true(
            "plan" not in names and "edit" not in names,
            f"spawn {i + 1} never sees other agents' tools",
        )
    assert_true(
        "parser.py" not in json.dumps(explore_reqs[1]["messages"]),
        "the first exploration leaves no trace in the second",
    )


def test_resume_summary_returns_to_planner() -> None:
    turns = [
        AssistantTurn(tool_calls=[call("explore", {"task": "Find the config loader."})]),
        AssistantTurn(
            tool_calls=[call("resume", {"summary": "config.py:139 load_config builds the Config."})]
        ),
        AssistantTurn(content="Enough planning for today."),
    ]
    ok, orch, ui, script = run_orchestrated(turns, goal="find it", mode=Mode.PLAN)
    follow_up = [r for r in planner_requests(script) if len(r["messages"]) > 2]
    assert_true(follow_up, "the planner streamed again after the explore call")
    tool_msgs = [
        m for m in follow_up[0]["messages"]
        if m.get("role") == "tool"
    ]
    assert_true(tool_msgs, "the explore call produced a tool result")
    content = str(tool_msgs[-1]["content"])
    assert_true(content.startswith("Exploration findings:"), f"labeled handoff: {content[:40]}")
    assert_true("load_config builds the Config" in content, "summary text delivered")


def test_explorer_budget_callback_binds_at_construction() -> None:
    # Same fail-open convention as plan_callback/question_callback: the
    # callback passed to Orchestrator(...) is what the explorer's runner
    # actually consults when its round budget runs out.
    async def budget_cb(context):
        return BudgetDecision(kind="stop")

    ui = FakeUI()
    orch = Orchestrator(client=None, config=Config(), ui=ui, explorer_budget_callback=budget_cb)
    assert_true(orch.explorer_budget_callback is budget_cb, "explorer budget callback bound")


def test_explorer_budget_force_choice_drives_a_resume_call() -> None:
    # A "force" decision keeps the explorer going past its exhausted budget
    # and drives a real `resume` call — real findings, not a fallback note.
    seen = []

    async def budget_cb(context):
        seen.append(context)
        return BudgetDecision(kind="force", forced_tool="resume")

    with tempfile.TemporaryDirectory() as tmp:
        list_call = call("list_directory", {"path": tmp})
        turns = [
            AssistantTurn(tool_calls=[call("explore", {"task": "Survey the repo."})]),
            *[AssistantTurn(tool_calls=[list_call]) for _ in range(6)],  # exhausts low (6)
            AssistantTurn(tool_calls=[call("resume", {"summary": "Repo surveyed."})]),
            AssistantTurn(content="Planning stops here."),
        ]
        ui = FakeUI()
        orch = Orchestrator(
            client=None, config=Config(reasoning_effort="low"), ui=ui,
            explorer_budget_callback=budget_cb,
        )
        script = Script(turns).install()
        agent_mod.WAKEUP_HOLD_SECONDS = 0
        try:
            # The scripted planner never calls `plan`, so the overall turn is
            # not `ok` (see test_planning_without_a_plan_call_fails_cleanly) —
            # this test is only about what the explorer sub-agent produced.
            asyncio.run(orch.run_turn("survey", Mode.PLAN))
        finally:
            script.uninstall()

    assert_equal(len(seen), 1, "the driver is asked exactly once")
    assert_equal(seen[0].max_rounds, 6, "context reports the exhausted budget")
    explore_results = [r for n, r, _ in ui.chat.tool_results if n == "explore"]
    assert_equal(len(explore_results), 1, "one explore result returned")
    assert_true(
        explore_results[0].startswith("Exploration findings:"),
        f"forced resume produced real findings, not a fallback: {explore_results[0][:60]}",
    )
    assert_true("Repo surveyed." in explore_results[0], "the forced resume's summary is delivered")


def test_explorer_budget_stop_choice_notes_user_stopped_early() -> None:
    # A "stop" decision behaves like today's cutoff, but the fallback report
    # says a human made the call, not just that the budget ran out.
    async def budget_cb(context):
        return BudgetDecision(kind="stop")

    with tempfile.TemporaryDirectory() as tmp:
        list_call = call("list_directory", {"path": tmp})
        turns = [
            AssistantTurn(tool_calls=[call("explore", {"task": "Survey the repo."})]),
            *[AssistantTurn(tool_calls=[list_call]) for _ in range(6)],
            AssistantTurn(content="Planning stops here."),
        ]
        ui = FakeUI()
        orch = Orchestrator(
            client=None, config=Config(reasoning_effort="low"), ui=ui,
            explorer_budget_callback=budget_cb,
        )
        script = Script(turns).install()
        agent_mod.WAKEUP_HOLD_SECONDS = 0
        try:
            asyncio.run(orch.run_turn("survey", Mode.PLAN))
        finally:
            script.uninstall()

    explore_results = [r for n, r, _ in ui.chat.tool_results if n == "explore"]
    assert_equal(len(explore_results), 1, "one explore result returned")
    assert_true(
        "user chose to stop this exploration early" in explore_results[0],
        f"the stop choice is noted: {explore_results[0]}",
    )


def test_explorer_budget_grow_choice_extends_the_budget_then_resumes() -> None:
    # A "grow" decision raises the round budget instead of cutting the
    # explorer off; it goes on to finish normally with real findings.
    calls = []

    async def budget_cb(context):
        calls.append(context)
        assert_equal(context.next_tier, ("medium", 10), "next tier above low is medium/10")
        return BudgetDecision(kind="grow", max_rounds=context.next_tier[1])

    with tempfile.TemporaryDirectory() as tmp:
        list_call = call("list_directory", {"path": tmp})
        turns = [
            AssistantTurn(tool_calls=[call("explore", {"task": "Survey the repo."})]),
            *[AssistantTurn(tool_calls=[list_call]) for _ in range(6)],  # exhausts low (6)
            AssistantTurn(tool_calls=[call("resume", {"summary": "Repo surveyed after growth."})]),
            AssistantTurn(content="Planning stops here."),
        ]
        ui = FakeUI()
        orch = Orchestrator(
            client=None, config=Config(reasoning_effort="low"), ui=ui,
            explorer_budget_callback=budget_cb,
        )
        script = Script(turns).install()
        agent_mod.WAKEUP_HOLD_SECONDS = 0
        try:
            # Same note as the "force" test above: no `plan` call is
            # scripted, so the overall turn is not `ok` — only the
            # explorer's own outcome is under test here.
            asyncio.run(orch.run_turn("survey", Mode.PLAN))
        finally:
            script.uninstall()

    assert_equal(len(calls), 1, "the driver is asked exactly once")
    explore_results = [r for n, r, _ in ui.chat.tool_results if n == "explore"]
    assert_equal(len(explore_results), 1, "one explore result returned")
    assert_true(
        explore_results[0].startswith("Exploration findings:"),
        f"growth let the explorer finish normally: {explore_results[0][:60]}",
    )
    assert_true(
        "Repo surveyed after growth." in explore_results[0],
        "the summary after growth is delivered",
    )


def test_explorer_without_resume_falls_back_to_its_notes() -> None:
    # An explorer that ends without a `resume` call still hands SOMETHING
    # back — its last notes, clearly labeled — so the planner can decide to
    # re-explore or continue.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("a.py").write_text("alpha = 1\n")
            turns = [
                AssistantTurn(tool_calls=[call("explore", {"task": "Check a.py."})]),
                AssistantTurn(
                    tool_calls=[call("inspect_file", {"file_path": str(Path(tmp) / "a.py")})]
                ),
                AssistantTurn(content="a.py only defines the alpha constant."),
                AssistantTurn(content="Planning stops here."),
            ]
            ok, orch, ui, script = run_orchestrated(turns, goal="check", mode=Mode.PLAN)
        finally:
            os.chdir(cwd)
    explore_results = [r for n, r, _ in ui.chat.tool_results if n == "explore"]
    assert_equal(len(explore_results), 1, "one explore result returned")
    assert_true(
        "finished without calling `resume`" in explore_results[0],
        f"fallback is labeled: {explore_results[0][:80]}",
    )
    assert_true(
        "alpha constant" in explore_results[0],
        "the explorer's last notes are handed back",
    )


def test_repeated_identical_explore_call_is_denied() -> None:
    # The per-tool repeat guard covers `explore` like `plan`: an identical
    # re-spawn would burn a whole sub-agent run for a report the planner
    # already has.
    turns = [
        AssistantTurn(tool_calls=[call("explore", {"task": "Map the parser."})]),
        AssistantTurn(tool_calls=[call("resume", {"summary": "parser mapped."})]),
        AssistantTurn(tool_calls=[call("explore", {"task": "Map the parser."})]),
    ]
    ok, orch, ui, script = run_orchestrated(turns, goal="map", mode=Mode.PLAN)
    assert_equal(len(explorer_requests(script)), 1, "no second explorer spawned")
    explore_results = [r for n, r, _ in ui.chat.tool_results if n == "explore"]
    assert_equal(len(explore_results), 2, "both calls produced a result")
    assert_true(
        explore_results[0].startswith("Exploration findings:"), "first call explored"
    )
    assert_true(
        not explore_results[1].startswith("Exploration findings:"),
        f"second call denied, not re-run: {explore_results[1][:60]}",
    )


def test_mode_display_flow_plan_visual_and_executors() -> None:
    # The mode box narrates the turn: PLAN while the planner runs, VISUAL for
    # the explorer sub-agent (then restored), NORMAL while executors run,
    # PLAN again for review pings.
    turns = [
        AssistantTurn(tool_calls=[call("explore", {"task": "Look around."})]),
        AssistantTurn(tool_calls=[call("resume", {"summary": "looked."})]),
        AssistantTurn(tool_calls=[call("plan", {"plan": "p", "todos": ["only step"]})]),
        AssistantTurn(content="Did the step."),
        AssistantTurn(content="Goal reached."),
    ]
    ok, orch, ui, script = run_orchestrated(turns, goal="go", mode=Mode.PLAN)
    assert_true(ok, "pipeline completes")
    assert_equal(
        ui.modes[:3],
        [Mode.PLAN, Mode.VISUAL, Mode.PLAN],
        "planning shows PLAN, the explorer flips to VISUAL and restores",
    )
    assert_true(Mode.NORMAL in ui.modes[3:], "executors show NORMAL")
    assert_true(
        ui.modes.index(Mode.NORMAL, 3) < len(ui.modes) - 1
        and Mode.PLAN in ui.modes[ui.modes.index(Mode.NORMAL, 3):],
        "the review ping flips back to PLAN",
    )


def test_edit_call_spawns_editor_and_returns_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("hello.py").write_text('def greet():\n    return "hi"\n')
            from tools.files.adapter import HashlineTools

            probe = HashlineTools().execute(
                "read_file", {"file_path": "hello.py", "_force": True}
            )
            second_line_id = probe.splitlines()[2].split("|", 1)[0]
            turns = [
                AssistantTurn(
                    tool_calls=[
                        call(
                            "edit",
                            {
                                "filename": "hello.py",
                                "line_ranges": "2",
                                "instructions": 'return "hello" instead of "hi"',
                            },
                        )
                    ]
                ),
                AssistantTurn(
                    tool_calls=[
                        call(
                            "replace",
                            {"file": "hello.py", "id": second_line_id, "content": '    return "hello"'},
                        )
                    ]
                ),
                AssistantTurn(content="Replaced the greeting string."),
                AssistantTurn(content="Greeting updated."),
            ]
            ok, orch, ui, script = run_orchestrated(turns, goal="fix the greeting")
            changed = Path("hello.py").read_text()
        finally:
            os.chdir(cwd)

    assert_true(ok, "turn completes")
    assert_true('return "hello"' in changed, f"editor applied the change: {changed}")
    editor_reqs = [
        r for r in script.requests
        if sorted(tool_names(r)) == ["delete", "grep_search", "insert", "lsp", "read_file", "replace"]
    ]
    assert_true(editor_reqs, "the editor ran with its own toolset")
    assert_equal(
        editor_reqs[0]["overrides"].get("qw35_session"), "scratch",
        "editor rides the scratch session",
    )
    assert_true(
        "qwowl35 editor" in system_of(editor_reqs[0]), "editor has its own strict prompt"
    )
    # The mode box flips to INSERT for the editor's lifetime, then restores.
    assert_equal(
        ui.modes, [Mode.NORMAL, Mode.INSERT, Mode.NORMAL],
        "NORMAL -> INSERT (editor) -> NORMAL",
    )
    edit_results = [r for n, r, _ in ui.chat.tool_results if n == "edit"]
    report = next((r for r in edit_results if "Editor result for hello.py" in r), None)
    assert_true(report is not None, f"executor received the editor report: {edit_results[:1]}")
    # The visualized diff spans the WHOLE editor session (start -> end).
    assert_true("Diff:" in report and "--- a/hello.py" in report, f"session diff present: {report}")
    assert_true(
        '-    return "hi"' in report and '+    return "hello"' in report,
        f"diff shows begin-of-session vs end-of-session content: {report}",
    )
    # Anchors are the editor's working vocabulary — the executor never sees
    # <line><hash>| ids or the "Current file (ids...)" state blocks.
    assert_true("(ids" not in report, f"no anchor-state block leaks: {report}")
    import re as _re
    assert_true(
        not any(_re.match(r"^[0-9a-f]{2,4}\|", line) for line in report.splitlines()),
        f"no <line><hash>| rows leak to the executor: {report}",
    )


def test_plan_callbacks_bind_at_construction_and_are_invoked() -> None:
    # The bash-approval pattern for every interactive tool: callbacks passed
    # to Orchestrator(...) reach the ONE PlanTools instance, and the gate
    # actually invokes them (a real user decision, source="user").
    from tools.plan import PlanDecision

    seen = {"plans": [], "questions": []}

    async def plan_cb(plan: str) -> PlanDecision:
        seen["plans"].append(plan)
        return PlanDecision(kind="approve")

    async def question_cb(questions: list) -> dict:
        seen["questions"].append(questions)
        return {q.get("question", ""): "option A" for q in questions}

    turns = [
        AssistantTurn(
            tool_calls=[call("plan", {"plan": "just do it", "todos": ["do it"]})]
        ),
        AssistantTurn(content="Did it."),          # executor
        AssistantTurn(content="Goal reached."),    # review ping
    ]
    ui = FakeUI()
    orch = Orchestrator(
        client=None, config=Config(), ui=ui,
        question_callback=question_cb, plan_callback=plan_cb,
    )
    assert_true(orch.registry.plan._plan_callback is plan_cb, "plan callback bound")
    assert_true(orch.registry.plan._question_callback is question_cb, "question callback bound")
    script = Script(turns).install()
    agent_mod.WAKEUP_HOLD_SECONDS = 0
    try:
        ok = asyncio.run(orch.run_turn("do it", Mode.PLAN))
    finally:
        script.uninstall()
    assert_true(ok, "turn completes")
    assert_equal(len(seen["plans"]), 1, "the plan call invoked the REAL reviewer")
    assert_true("just do it" in seen["plans"][0], "the modal carries the plan text")
    assert_equal(orch.registry.plan.last_decision.source, "user", "a user decision, not auto")
    assert_true(not ui.warnings, f"no fallback warnings when bound: {ui.warnings}")


def test_unbound_plan_gate_announces_auto_approval() -> None:
    # No reviewer bound: the machine may proceed (unattended fallback) but
    # must SAY so — a status warning and a transcript line, and the decision
    # is marked source="auto". Silent impersonation of the user is the bug
    # this guards against.
    turns = [
        AssistantTurn(
            tool_calls=[call("plan", {"plan": "just do it", "todos": ["do it"]})]
        ),
        AssistantTurn(content="Did it."),          # executor
        AssistantTurn(content="Goal reached."),    # review ping
    ]
    ok, orch, ui, script = run_orchestrated(turns, goal="do it", mode=Mode.PLAN)
    assert_true(ok, "turn completes")
    assert_equal(orch.registry.plan.last_decision.source, "auto", "decision marked auto")
    assert_true(
        any("no reviewer is bound" in w for w in ui.warnings),
        f"auto-approval announced in the status bar: {ui.warnings}",
    )
    assert_true(
        any("auto-approved" in n for n in ui.chat.system_notes),
        f"auto-approval announced in the transcript: {ui.chat.system_notes}",
    )


def test_plan_call_is_the_approval_gate() -> None:
    # Writing the todos and gating the plan are ONE atomic `plan` call: the
    # approve decision ends planning immediately (stop_when — no closing
    # narration round is consumed) and the executor receives the call's
    # `plan` field as the plan description.
    turns = [
        AssistantTurn(
            tool_calls=[
                call(
                    "plan",
                    {
                        "plan": "Patch the header builder, then verify against cal.",
                        "todos": ["do the thing"],
                    },
                )
            ]
        ),
        AssistantTurn(content="Did the thing."),  # executor (NOT more planning)
        AssistantTurn(content="Goal reached."),   # review ping closes the loop
    ]
    ok, orch, ui, script = run_orchestrated(turns, goal="do the thing", mode=Mode.PLAN)
    assert_true(ok, "pipeline completes")
    assert_equal(orch.registry.plan.last_decision.kind, "approve", "gate resolved")
    exec_reqs = exec_requests(script)
    assert_true(exec_reqs, "an executor spawned after the approved plan call")
    exec_context = exec_reqs[0]["messages"][1]["content"]
    assert_true("Approved plan:" in exec_context, "executor received the plan block")
    assert_true(
        "Patch the header builder, then verify against cal." in exec_context,
        f"the plan field reaches the executor as the description: {exec_context}",
    )
    plan_reqs = [
        r for r in planner_requests(script)
        if not any("Executor report:" in str(m.get("content", "")) for m in r["messages"])
    ]
    assert_equal(len(plan_reqs), 1, "planning consumed exactly one round: the plan call")


def test_planning_without_a_plan_call_fails_cleanly() -> None:
    # A planner that only narrates never opens the gate: there is no
    # synthesized-gate rescue (writing todos IS the gate), so the turn ends
    # without execution and says so.
    turns = [
        AssistantTurn(content="The plan is to do the thing carefully."),
    ]
    ok, orch, ui, script = run_orchestrated(turns, goal="do the thing", mode=Mode.PLAN)
    assert_true(not ok, "the turn does not reach execution")
    assert_true(
        "without an approved plan" in orch.turn_log[-1][1],
        f"outcome names the missing approval: {orch.turn_log[-1][1]}",
    )
    assert_equal(len(exec_requests(script)), 0, "no executor spawned")


def test_editor_noop_run_is_reported_as_ineffective() -> None:
    # Regression: an editor made only byte-identical "edits" yet summarized
    # "I've rewritten the logic"; the executor burned a verify-and-rebuild
    # cycle discovering the truth. The mutation results are ground truth: an
    # all-no-op editor run is reported as ineffective, flagged as an error.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("hello.py").write_text('def greet():\n    return "hi"\n')
            from tools.files.adapter import HashlineTools

            probe = HashlineTools().execute(
                "read_file", {"file_path": "hello.py", "_force": True}
            )
            second_line_id = probe.splitlines()[2].split("|", 1)[0]
            turns = [
                AssistantTurn(
                    tool_calls=[
                        call(
                            "edit",
                            {
                                "filename": "hello.py",
                                "line_ranges": "2",
                                "instructions": "improve the return",
                            },
                        )
                    ]
                ),
                # The editor "edits" the line to its identical current content
                # and then claims success.
                AssistantTurn(
                    tool_calls=[
                        call(
                            "replace",
                            {"file": "hello.py", "id": second_line_id, "content": '    return "hi"'},
                        )
                    ]
                ),
                AssistantTurn(content="I've rewritten the greeting logic."),
                AssistantTurn(content="Understood, handling it differently."),
            ]
            ok, orch, ui, script = run_orchestrated(turns, goal="fix the greeting")
        finally:
            os.chdir(cwd)

    assert_true(ok, "turn completes")
    ineffective = [
        (r, err) for n, r, err in ui.chat.tool_results
        if n == "edit" and "no effective changes" in r
    ]
    assert_true(ineffective, f"ineffective editor run reported: {[r for n, r, _ in ui.chat.tool_results if n == 'edit']}")
    assert_true(ineffective[0][1], "flagged as an error, not a success")
    assert_true(
        all("I've rewritten" not in r for r, _ in ineffective),
        "the editor's false summary is not relayed as the outcome",
    )


def test_identical_plan_recall_after_revise_is_denied() -> None:
    # Regression heir: a planner once rewrote the identical todo list 13
    # times, each a full round-trip. After a revise decision, re-sending the
    # IDENTICAL `plan` call is denied by the dedup guard WITHOUT re-firing
    # the modal, and the denied-only turn ends planning (stop_on_stall) —
    # the turn fails cleanly instead of spinning.
    from tools.plan import PlanDecision

    presented: list[str] = []

    async def always_revise(plan: str) -> PlanDecision:
        presented.append(plan)
        return PlanDecision(kind="revise", text="split the step")

    same = {"plan": "do it", "todos": ["do the thing"]}
    turns = [
        AssistantTurn(tool_calls=[call("plan", same)]),   # gate: revise
        AssistantTurn(tool_calls=[call("plan", same)]),   # identical → denied → stall
        AssistantTurn(content="never reached"),
    ]
    ui = FakeUI()
    orch = Orchestrator(client=None, config=Config(), ui=ui, plan_callback=always_revise)
    script = Script(turns).install()
    agent_mod.WAKEUP_HOLD_SECONDS = 0
    try:
        ok = asyncio.run(orch.run_turn("do the thing", Mode.PLAN))
    finally:
        script.uninstall()
    assert_true(not ok, "no approved plan, no execution")
    assert_equal(len(presented), 1, "the identical re-call never re-fired the modal")
    plan_results = [r for n, r, _ in ui.chat.tool_results if n == "plan"]
    assert_equal(len(plan_results), 2, "revise result + denial surfaced")
    from agent import REPEATED_TOOL_NOTES

    denial_pool = [note.format(name="plan") for note in REPEATED_TOOL_NOTES]
    assert_true(
        plan_results[1] in denial_pool,
        f"second call denied from the repeat-note pool: {plan_results[1]!r}",
    )


def test_progress_before_approval_errors_and_every_todo_executes() -> None:
    # A small model may try to mark progress before any plan exists; the
    # error result costs one round but planning recovers, and after approval
    # the cursor starts at zero — BOTH todos get executors.
    turns = [
        AssistantTurn(tool_calls=[call("plan", {"progress": "0"})]),  # too early
        AssistantTurn(
            tool_calls=[
                call("plan", {"plan": "two steps", "todos": ["step one", "step two"]})
            ]
        ),
        AssistantTurn(content="Did step one."),
        AssistantTurn(content="Continue."),      # review 1
        AssistantTurn(content="Did step two."),
        AssistantTurn(content="Goal reached."),  # review 2
    ]
    ok, orch, ui, script = run_orchestrated(turns, goal="two steps", mode=Mode.PLAN)
    assert_true(ok, "pipeline completes")
    early = [r for n, r, _ in ui.chat.tool_results if n == "plan"][0]
    assert_true("no approved plan exists yet" in early, early)
    exec_reqs = exec_requests(script)
    assert_equal(len(exec_reqs), 2, "an executor ran for EVERY todo")
    assert_true("step one" in last_user_of(exec_reqs[0]), "first todo executed")
    assert_true("step two" in last_user_of(exec_reqs[1]), "second todo executed")
    assert_equal(orch.registry.plan.state.progress, 2, "cursor reached the end")


def test_executors_inherit_context() -> None:
    # Consecutive executors CONTINUE one conversation: request N+1 starts
    # with request N's messages as a strict prefix plus exactly one new user
    # directive (after N's own reply) — checkpoint prefix reuse shape.
    turns = [
        AssistantTurn(
            tool_calls=[call("plan", {"plan": "two", "todos": ["first", "second"]})]
        ),
        AssistantTurn(content="Did first."),
        AssistantTurn(content="Continue."),      # review 1
        AssistantTurn(content="Did second."),
        AssistantTurn(content="Goal reached."),  # review 2
    ]
    ok, orch, ui, script = run_orchestrated(turns, goal="two steps", mode=Mode.PLAN)
    assert_true(ok, "pipeline completes")
    exec_reqs = exec_requests(script)
    assert_equal(len(exec_reqs), 2, "one executor stream per todo")
    first, second = exec_reqs
    assert_equal(len(first["messages"]), 2, "todo 1: system + opener only")
    assert_equal(
        second["messages"][:2], first["messages"],
        "todo 2's context extends todo 1's (strict prefix)",
    )
    assert_equal(
        len(second["messages"]), 4,
        "todo 2 adds exactly todo 1's reply + one continuation directive",
    )
    assert_equal(second["messages"][2]["role"], "assistant", "todo 1's reply rides along")
    assert_true("Did first." in second["messages"][2]["content"], "inherited, not re-packaged")
    directive = second["messages"][3]["content"]
    assert_true(
        directive.startswith("Previous task done.") or "Your next task (2/2)" in directive,
        f"slim continuation directive: {directive[:60]}",
    )


def test_planner_can_keep_a_todo_open_for_a_retry() -> None:
    # The dialogue at work: the executor reports failure, the planner keeps
    # the todo open (an explicit progress mark re-asserting the current
    # cursor — a write, so the machine fallback stays out) with a guidance
    # line, and the SAME todo is retried on the same executor conversation.
    turns = [
        AssistantTurn(
            tool_calls=[
                call("plan", {"plan": "1. green tests", "todos": ["make tests pass"]})
            ]
        ),
        # attempt 1 fails; the planner keeps it open with guidance.
        AssistantTurn(content="Tests still fail: missing import."),
        AssistantTurn(tool_calls=[call("plan", {"progress": "0"})]),  # keep open
        AssistantTurn(content="Add the missing math import first."),  # round 2
        # attempt 2 succeeds; planner silent -> machine completes it.
        AssistantTurn(content="Fixed the import; tests pass."),
        AssistantTurn(content="Goal reached."),   # final review
    ]
    ok, orch, ui, script = run_orchestrated(turns, goal="make tests pass", mode=Mode.PLAN)
    assert_true(ok, "pipeline completes")
    exec_reqs = exec_requests(script)
    assert_equal(len(exec_reqs), 2, "the kept-open todo was retried")
    retry_context = last_user_of(exec_reqs[1])
    assert_true("make tests pass" in retry_context, "same todo retried")
    assert_true(
        "Add the missing math import first." in retry_context,
        f"planner guidance reached the retry: {retry_context}",
    )
    assert_equal(
        orch.registry.plan.state.progress, 1,
        "todo completed after the successful retry",
    )


def test_planner_cannot_skip_todos_with_a_partial_jump() -> None:
    # Completion is earned by execution: after task 1 the planner marks
    # progress through todo 2 while todo 3 remains — a partial jump-ahead
    # reads as a stale mark, not a verdict. The machine clamps the cursor
    # to the reviewed todo — both remaining todos still get executors.
    turns = [
        AssistantTurn(
            tool_calls=[
                call("plan", {"plan": "1-3", "todos": ["first", "second", "third"]})
            ]
        ),
        AssistantTurn(content="Did first."),
        AssistantTurn(
            tool_calls=[call("plan", {"progress": todo_ref(2, "second")})]  # skips ahead
        ),
        AssistantTurn(content="Skipping ahead!"),  # review round 2
        AssistantTurn(content="Did second."),      # executor 2 STILL runs
        AssistantTurn(content="Continue."),        # review 2
        AssistantTurn(content="Did third."),       # executor 3 STILL runs
        AssistantTurn(content="Goal reached."),    # review 3
    ]
    ok, orch, ui, script = run_orchestrated(turns, goal="three steps", mode=Mode.PLAN)
    assert_true(ok, "pipeline completes")
    exec_reqs = exec_requests(script)
    assert_equal(len(exec_reqs), 3, "every todo earned its executor")
    assert_true("second" in last_user_of(exec_reqs[1]), "todo 2 executed")
    assert_true("third" in last_user_of(exec_reqs[2]), "todo 3 executed")
    assert_equal(
        orch.registry.plan.state.progress, 3,
        "all todos completed only after their executions",
    )


def test_work_dispatch_sends_the_executor_back_to_a_todo() -> None:
    # The explicit dispatch verb: the executor reports failure and the
    # planner answers plan(work=<ref of the same todo>) — the executor
    # conversation continues with that todo again (carrying the guidance
    # line), then execution continues down the list.
    turns = [
        AssistantTurn(
            tool_calls=[call("plan", {"plan": "1-2", "todos": ["first", "second"]})]
        ),
        AssistantTurn(content="Tests failed halfway."),   # executor 1, attempt 1
        AssistantTurn(
            tool_calls=[call("plan", {"work": todo_ref(1, "first")})]  # review 1
        ),
        AssistantTurn(content="Fix the import, then rerun."),  # review round 2
        AssistantTurn(content="Did first."),      # executor 1, attempt 2
        AssistantTurn(content="Continue."),       # review 2 (silent -> done)
        AssistantTurn(content="Did second."),     # executor 2
        AssistantTurn(content="Goal reached."),   # review 3
    ]
    ok, orch, ui, script = run_orchestrated(turns, goal="two steps", mode=Mode.PLAN)
    assert_true(ok, "pipeline completes")
    exec_reqs = exec_requests(script)
    assert_equal(len(exec_reqs), 3, "the dispatch earned another executor run for the todo")
    retry_context = last_user_of(exec_reqs[1])
    assert_true("first" in retry_context, "the dispatched executor reruns todo 1")
    assert_true(
        "Fix the import, then rerun." in retry_context,
        f"planner guidance reached the dispatched executor: {retry_context}",
    )
    assert_true("second" in last_user_of(exec_reqs[2]), "then todo 2 runs")
    assert_equal(orch.registry.plan.state.progress, 2, "all todos complete in the end")


def test_full_completion_mark_ends_the_loop_without_more_executors() -> None:
    # One task often accomplishes the whole plan. When the planner marks
    # EVERYTHING done (cites the final ref) the loop honors it as the "goal
    # reached" signal: no executor runs for the remaining todos.
    turns = [
        AssistantTurn(
            tool_calls=[
                call("plan", {"plan": "1-3", "todos": ["first", "second", "third"]})
            ]
        ),
        AssistantTurn(content="Did everything in one pass."),
        AssistantTurn(
            tool_calls=[call("plan", {"progress": todo_ref(3, "third")})]  # all done
        ),
        AssistantTurn(content="Goal reached."),    # review round 2
    ]
    ok, orch, ui, script = run_orchestrated(turns, goal="three steps", mode=Mode.PLAN)
    assert_true(ok, "pipeline completes")
    assert_equal(len(exec_requests(script)), 1, "no executor after the all-done mark")
    assert_equal(orch.registry.plan.state.progress, 3, "cursor honored at the end")


def test_double_failure_aborts_the_execution_loop() -> None:
    # Systemic failure guard: when an executor AND its review ping both
    # fail (server down, context overflow), the turn aborts instead of the
    # fallback marching through every todo marking them completed.
    turns = [
        AssistantTurn(
            tool_calls=[call("plan", {"plan": "1-2", "todos": ["first", "second"]})]
        ),
        "ERROR",  # executor 1 stream fails
        "ERROR",  # its review ping fails too -> abort
        AssistantTurn(content="never reached"),
    ]
    ok, orch, ui, script = run_orchestrated(turns, goal="two steps", mode=Mode.PLAN)
    assert_true(not ok, "the turn aborts")
    assert_equal(len(exec_requests(script)), 1, "no further executor stream")
    assert_true(
        "both failed" in orch.turn_log[-1][1] if orch.turn_log else False,
        f"outcome names the double failure: {orch.turn_log}",
    )
    assert_equal(
        orch.registry.plan.state.progress, 0,
        "no todo was falsely completed",
    )


def test_replan_during_review_refires_modal_and_updates_handoff() -> None:
    # A replan mid-execution (new remaining todos + reason) re-fires the
    # approval modal — completed work shown — and, once approved, the next
    # executor directive carries the RECOMPUTED plan block with the spliced
    # todo list.
    from tools.plan import PlanDecision

    presented: list[str] = []

    async def approve(plan: str) -> PlanDecision:
        presented.append(plan)
        return PlanDecision(kind="approve")

    turns = [
        AssistantTurn(
            tool_calls=[call("plan", {"plan": "p", "todos": ["first", "second"]})]
        ),
        AssistantTurn(content="Did first."),
        # review 1: record the finished todo, then replace the remaining
        # work (the splice keeps only todos completed per the cursor).
        AssistantTurn(
            tool_calls=[call("plan", {"progress": todo_ref(1, "first")})]
        ),
        AssistantTurn(
            tool_calls=[
                call(
                    "plan",
                    {"todos": ["revised second", "third"], "reason": "scope grew"},
                )
            ]
        ),
        AssistantTurn(content="Did revised second."),
        AssistantTurn(content="Continue."),        # review 2
        AssistantTurn(content="Did third."),
        AssistantTurn(content="Goal reached."),    # review 3
    ]
    ui = FakeUI()
    orch = Orchestrator(client=None, config=Config(), ui=ui, plan_callback=approve)
    script = Script(turns).install()
    agent_mod.WAKEUP_HOLD_SECONDS = 0
    try:
        ok = asyncio.run(orch.run_turn("shifting goal", Mode.PLAN))
    finally:
        script.uninstall()
    assert_true(ok, "pipeline completes")
    assert_equal(len(presented), 2, "the replan re-fired the modal")
    assert_true("scope grew" in presented[1], "the modal carries the reason")
    assert_true("Already completed:" in presented[1], "the modal shows finished work")
    exec_reqs = exec_requests(script)
    assert_equal(len(exec_reqs), 3, "executors ran the original first + spliced tail")
    exec2_context = last_user_of(exec_reqs[1])
    assert_true("revised second" in exec2_context, "executor 2 runs the spliced todo")
    assert_true(
        "plan was revised and re-approved" in exec2_context,
        f"the recomputed plan block reaches the next directive: {exec2_context[:80]}",
    )
    assert_true(
        "revised second" in exec2_context.split("re-approved:")[-1],
        "the refreshed plan block lists the spliced todos",
    )
    state = orch.registry.plan.state
    assert_equal(state.todos, ["first", "revised second", "third"], "spliced list")
    assert_equal(state.progress, 3, "everything completed")


def test_declined_replan_continues_the_existing_plan() -> None:
    # The user vetoes the replan: nothing is committed, the tool result says
    # to continue, and the ORIGINAL remaining todo still gets its executor.
    from tools.plan import PlanDecision

    decisions = [PlanDecision(kind="approve"), PlanDecision(kind="reject")]

    async def gate(plan: str) -> PlanDecision:
        return decisions.pop(0)

    turns = [
        AssistantTurn(
            tool_calls=[call("plan", {"plan": "p", "todos": ["first", "second"]})]
        ),
        AssistantTurn(content="Did first."),
        AssistantTurn(
            tool_calls=[call("plan", {"todos": ["other work"], "reason": "detour"})]
        ),
        AssistantTurn(content="Staying the course."),  # review 1 guidance
        AssistantTurn(content="Did second."),
        AssistantTurn(content="Goal reached."),    # review 2
    ]
    ui = FakeUI()
    orch = Orchestrator(client=None, config=Config(), ui=ui, plan_callback=gate)
    script = Script(turns).install()
    agent_mod.WAKEUP_HOLD_SECONDS = 0
    try:
        ok = asyncio.run(orch.run_turn("steady goal", Mode.PLAN))
    finally:
        script.uninstall()
    assert_true(ok, "pipeline completes")
    declined = [r for n, r, _ in ui.chat.tool_results if n == "plan" and "declined" in r]
    assert_true(declined, "the declined replan told the planner to continue")
    state = orch.registry.plan.state
    assert_equal(state.todos, ["first", "second"], "original plan kept")
    exec_reqs = exec_requests(script)
    assert_equal(len(exec_reqs), 2, "the original second todo still executed")
    assert_true("second" in last_user_of(exec_reqs[1]), "todo 2 executed")


def test_session_notes_reach_the_next_turn() -> None:
    ui = FakeUI()
    orch = Orchestrator(client=None, config=Config(), ui=ui)
    orch.turn_log.append(("build the parser", "Parser built and tested."))
    script = Script([AssistantTurn(content="Nothing to do.")]).install()
    agent_mod.WAKEUP_HOLD_SECONDS = 0
    try:
        asyncio.run(orch.run_turn("what did we do earlier?", Mode.NORMAL))
    finally:
        script.uninstall()
    first_req = script.requests[0]
    assert_true(
        "Parser built and tested." in json.dumps(first_req["messages"]),
        "the next turn's agent sees the session notes",
    )


def test_editor_spawn_receives_obfuscated_background() -> None:
    # The editor's opening message carries a background block: the delegating
    # agent's recent activity (tool calls obfuscated to markdown — never
    # tool-call syntax) and the hidden reasoning of the turn that issued the
    # edit call. The block lands ONLY in the editor's fresh scratch context;
    # the executor's own history never carries it.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("hello.py").write_text('def greet():\n    return "hi"\n')
            from tools.files.adapter import HashlineTools

            probe = HashlineTools().execute(
                "read_file", {"file_path": "hello.py", "_force": True}
            )
            second_line_id = probe.splitlines()[2].split("|", 1)[0]
            turns = [
                AssistantTurn(
                    tool_calls=[
                        call("run_shell_command", {"command": "grep -n hi hello.py"}, index=1)
                    ]
                ),
                AssistantTurn(
                    reasoning="the greeting must say hello, not hi",
                    tool_calls=[
                        call(
                            "edit",
                            {
                                "filename": "hello.py",
                                "line_ranges": "2",
                                "instructions": 'return "hello" instead of "hi"',
                            },
                        )
                    ],
                ),
                AssistantTurn(
                    tool_calls=[
                        call(
                            "replace",
                            {"file": "hello.py", "id": second_line_id, "content": '    return "hello"'},
                        )
                    ]
                ),
                AssistantTurn(content="Replaced the greeting."),
                AssistantTurn(content="Done."),
            ]
            ok, orch, ui, script = run_orchestrated(turns, goal="fix the greeting")
        finally:
            os.chdir(cwd)

    assert_true(ok, "turn completes")
    editor_reqs = [
        r for r in script.requests
        if sorted(tool_names(r)) == ["delete", "grep_search", "insert", "lsp", "read_file", "replace"]
    ]
    assert_true(editor_reqs, "the editor ran")
    opening = editor_reqs[0]["messages"][1]["content"]
    assert_true(
        opening.startswith(BACKGROUND_HEADER),
        f"background opens the editor message: {opening[:160]}",
    )
    assert_true(
        "```bash" in opening and "grep -n hi hello.py" in opening,
        f"bash call obfuscated as a markdown fence: {opening}",
    )
    assert_true('2:    return "hi"' in opening, f"clipped grep output present: {opening}")
    assert_true(
        "the greeting must say hello, not hi" in opening,
        f"spawn-turn hidden reasoning present: {opening}",
    )
    assert_true(
        "<tool_call" not in opening and '"arguments"' not in opening,
        "no tool-call syntax reaches the editor",
    )
    assert_true("\n---\n" in opening and "File: " in opening, "task section follows the background")
    assert_true("Approved plan" not in opening, "NORMAL mode has no plan section")
    # Editor-only injection: no executor request ever carries the background.
    for request in exec_requests(script):
        for message in request["messages"]:
            assert_true(
                BACKGROUND_HEADER not in str(message.get("content", "")),
                "executor history untouched by the spawn background",
            )


def main() -> None:
    test_rewrite_advice_never_names_absent_tools()
    test_editor_spawn_receives_obfuscated_background()
    test_normal_mode_runs_direct_executor()
    test_display_modes_clamp_to_normal_dispatch()
    test_mode_dispatch_selects_the_right_agent()
    test_stage_toolsets_are_segregated()
    test_begin_stage_sets_write_feedback_dialect()
    test_explorer_shell_is_restricted_under_the_hood()
    test_conversation_routes_to_chat_and_keeps_history()
    test_web_mode_answers_with_web_tools_only()
    test_easy_task_runs_executor_with_only_bash_and_edit()
    test_execute_stage_bash_write_gets_anchors_and_syntax_check()
    test_editor_leaving_file_broken_reports_syntax_to_executor()
    test_editor_spawn_diag_memory_is_fresh_and_executor_dedup_persists()
    test_editor_conversation_spans_files()
    test_editor_break_then_fix_does_not_flag_executor()
    test_plan_pipeline_hands_over_data_not_conversation()
    test_explore_spawns_stateless_explorer_each_time()
    test_resume_summary_returns_to_planner()
    test_explorer_budget_callback_binds_at_construction()
    test_explorer_budget_force_choice_drives_a_resume_call()
    test_explorer_budget_stop_choice_notes_user_stopped_early()
    test_explorer_budget_grow_choice_extends_the_budget_then_resumes()
    test_explorer_without_resume_falls_back_to_its_notes()
    test_repeated_identical_explore_call_is_denied()
    test_mode_display_flow_plan_visual_and_executors()
    test_edit_call_spawns_editor_and_returns_report()
    test_plan_callbacks_bind_at_construction_and_are_invoked()
    test_unbound_plan_gate_announces_auto_approval()
    test_plan_call_is_the_approval_gate()
    test_planning_without_a_plan_call_fails_cleanly()
    test_editor_noop_run_is_reported_as_ineffective()
    test_identical_plan_recall_after_revise_is_denied()
    test_progress_before_approval_errors_and_every_todo_executes()
    test_executors_inherit_context()
    test_planner_can_keep_a_todo_open_for_a_retry()
    test_planner_cannot_skip_todos_with_a_partial_jump()
    test_work_dispatch_sends_the_executor_back_to_a_todo()
    test_full_completion_mark_ends_the_loop_without_more_executors()
    test_double_failure_aborts_the_execution_loop()
    test_replan_during_review_refires_modal_and_updates_handoff()
    test_declined_replan_continues_the_existing_plan()
    test_session_notes_reach_the_next_turn()
    print("orchestrator tests passed")


if __name__ == "__main__":
    main()
