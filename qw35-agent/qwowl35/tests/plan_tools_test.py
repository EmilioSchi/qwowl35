"""Tests for the planner tools, the explore/resume schemas, and the subedit
schema.

Run directly: ``python qwowl35/tests/plan_tools_test.py``. No network, no UI —
callbacks are faked.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.explorer import (  # noqa: E402
    EXPLORE_NAME,
    EXPLORE_SCHEMA,
    RESUME_NAME,
    RESUME_SCHEMA,
)
from tools.plan import PlanDecision, PlanTools, todo_ref  # noqa: E402
from tools.subedit import EDIT_SCHEMA, validate_edit_args  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def approving_tools() -> PlanTools:
    async def approve(plan: str) -> PlanDecision:
        return PlanDecision(kind="approve")

    return PlanTools(plan_callback=approve)


def approved_tools(todos: list[str]) -> PlanTools:
    tools = approving_tools()
    result = asyncio.run(
        tools.execute("plan", {"plan": "the approach", "todos": todos})
    )
    assert_true("approved the plan" in result, result)
    return tools


def test_plan_schemas_use_wire_names() -> None:
    names = [schema["function"]["name"] for schema in PlanTools().schemas()]
    assert_equal(names, ["plan", "ask_user_question"], "two wire tools only")


def test_plan_present_fires_gate_and_commits() -> None:
    presented: list[str] = []

    async def approve(plan: str) -> PlanDecision:
        presented.append(plan)
        return PlanDecision(kind="approve")

    tools = PlanTools(plan_callback=approve)
    result = asyncio.run(
        tools.execute(
            "plan",
            {"plan": "### Approach\ndo it", "todos": [" read config ", "run tests"]},
        )
    )
    assert_equal(len(presented), 1, "gate invoked once")
    assert_true("### Approach" in presented[0], "modal carries the plan text")
    assert_true("read config" in presented[0], "modal carries the checklist")
    assert_true(result.startswith("Todo list updated:\n"), result)
    assert_true("Execution will start now" in result, result)
    state = tools.state
    assert_equal(state.todos, ["read config", "run tests"], "todos stripped+committed")
    assert_equal(state.progress, 0, "cursor starts at zero")
    assert_equal(state.version, 1, "approve bumps the version")
    assert_true(state.approved, "approved flag set")
    ref = todo_ref(1, "read config")
    assert_true(f"[>] {ref}: read config" in result, result)


def test_plan_present_requires_plan_text_first_cycle() -> None:
    tools = approving_tools()
    result = asyncio.run(tools.execute("plan", {"todos": ["step"]}))
    assert_true(result.startswith("Error"), result)
    assert_true("include the `plan` markdown" in result, result)
    assert_equal(tools.state.todos, [], "nothing committed")


def test_plan_revise_and_reject_do_not_commit() -> None:
    decisions = [
        PlanDecision(kind="revise", text="split step 2"),
        PlanDecision(kind="reject"),
    ]

    async def gate(plan: str) -> PlanDecision:
        return decisions.pop(0)

    tools = PlanTools(plan_callback=gate)
    revised = asyncio.run(
        tools.execute("plan", {"plan": "p", "todos": ["a", "b"]})
    )
    assert_true("split step 2" in revised, revised)
    assert_true("call `plan` again" in revised, revised)
    assert_equal(tools.state.todos, [], "revise commits nothing")
    assert_equal(tools.state.version, 0, "revise does not bump the version")

    rejected = asyncio.run(
        tools.execute("plan", {"plan": "p", "todos": ["a", "b"]})
    )
    assert_true("rejected the plan" in rejected, rejected)
    assert_equal(tools.state.todos, [], "reject commits nothing")


def test_plan_call_shape_validation() -> None:
    tools = approving_tools()
    empty = asyncio.run(tools.execute("plan", {}))
    assert_true(empty.startswith("Error"), empty)
    both = asyncio.run(
        tools.execute("plan", {"todos": ["a"], "progress": "0"})
    )
    assert_true("not several" in both, both)
    progress_and_work = asyncio.run(
        tools.execute("plan", {"progress": "0", "work": "1ab"})
    )
    assert_true("not several" in progress_and_work, progress_and_work)
    bad_item = asyncio.run(
        tools.execute("plan", {"plan": "p", "todos": ["a", "  "]})
    )
    assert_true("todos[1]" in bad_item, bad_item)
    early = asyncio.run(tools.execute("plan", {"progress": "0"}))
    assert_true("no approved plan exists yet" in early, early)


def test_ask_user_question_dead_only_after_approval() -> None:
    decisions = [
        PlanDecision(kind="revise", text="tighter"),
        PlanDecision(kind="approve"),
    ]

    async def gate(plan: str) -> PlanDecision:
        return decisions.pop(0)

    async def fake_modal(questions: list[dict]) -> dict:
        return {questions[0]["question"]: "A"}

    tools = PlanTools(question_callback=fake_modal, plan_callback=gate)
    args = {
        "questions": [
            {
                "question": "Which one?",
                "header": "Pick",
                "options": [
                    {"label": "A", "description": "a"},
                    {"label": "B", "description": "b"},
                ],
            }
        ]
    }
    before = asyncio.run(tools.execute("ask_user_question", args))
    assert_true("**Pick**: A" in before, before)
    # A revised (uncommitted) plan keeps the question window open — the
    # planner may need to clarify the revision feedback.
    asyncio.run(tools.execute("plan", {"plan": "p", "todos": ["a"]}))
    revised = asyncio.run(tools.execute("ask_user_question", args))
    assert_true("**Pick**: A" in revised, revised)
    # Approval closes it.
    asyncio.run(tools.execute("plan", {"plan": "p", "todos": ["a"]}))
    after = asyncio.run(tools.execute("ask_user_question", args))
    assert_true("no longer available" in after, after)
    assert_true("already been approved" in after, after)


def test_progress_advances_reopens_and_zero_resets() -> None:
    tools = approved_tools(["add subtract()", "wire the CLI", "run tests"])
    state = tools.state
    version_start = state.version

    two = asyncio.run(
        tools.execute("plan", {"progress": todo_ref(2, "wire the CLI")})
    )
    assert_equal(state.progress, 2, "absolute cursor set")
    assert_true(f"[x] {todo_ref(1, 'add subtract()')}" in two, two)
    assert_true("Next: " + todo_ref(3, "run tests") in two, two)
    assert_equal(state.version, version_start + 1, "progress bumps the version")

    one = asyncio.run(
        tools.execute("plan", {"progress": todo_ref(1, "add subtract()")})
    )
    assert_equal(state.progress, 1, "earlier ref reopens later todos")
    assert_true(f"[>] {todo_ref(2, 'wire the CLI')}" in one, one)

    zero = asyncio.run(tools.execute("plan", {"progress": "0"}))
    assert_equal(state.progress, 0, "'0' reopens all")
    assert_true("Next: " + todo_ref(1, "add subtract()") in zero, zero)

    done = asyncio.run(
        tools.execute("plan", {"progress": todo_ref(3, "run tests")})
    )
    assert_true("All todos are completed." in done, done)


def test_work_dispatches_retry_reopen_and_rejects_skip_ahead() -> None:
    tools = approved_tools(["first", "second", "third"])
    state = tools.state

    # Retry the todo about to run / just run: cursor unchanged, but the
    # explicit write bumps the version so the machine respects it.
    version = state.version
    retry = asyncio.run(tools.execute("plan", {"work": todo_ref(1, "first")}))
    assert_equal(state.progress, 0, "retry keeps the cursor on the todo")
    assert_equal(state.version, version + 1, "work bumps the version")
    assert_true("dispatched to" in retry and "first" in retry, retry)

    # Redo an earlier todo: the cursor reopens it (and everything after).
    asyncio.run(tools.execute("plan", {"progress": todo_ref(2, "second")}))
    back = asyncio.run(tools.execute("plan", {"work": todo_ref(2, "second")}))
    assert_equal(state.progress, 1, "cursor reopened to run todo 2")
    assert_true(f"[>] {todo_ref(2, 'second')}" in back, back)
    assert_true("reopened" in back, back)

    # Skipping ahead past unexecuted todos is rejected (statuses derive
    # from the cursor — the jump would mark the skipped ones complete).
    skip = asyncio.run(tools.execute("plan", {"work": todo_ref(3, "third")}))
    assert_true(skip.startswith("Error"), skip)
    assert_true("skips ahead" in skip, skip)
    assert_equal(state.progress, 1, "rejected dispatch does not move the cursor")

    # Dispatching the NEXT pending todo is allowed (explicit "start here").
    next_ok = asyncio.run(tools.execute("plan", {"work": todo_ref(2, "second")}))
    assert_true("dispatched to" in next_ok, next_ok)

    # '0' has no todo to run; before approval there is nothing to dispatch.
    zero = asyncio.run(tools.execute("plan", {"work": "0"}))
    assert_true(zero.startswith("Error"), zero)
    early = asyncio.run(approving_tools().execute("plan", {"work": "1ab"}))
    assert_true("no approved plan exists yet" in early, early)


def test_progress_ref_mismatch_rerenders_list() -> None:
    tools = approved_tools(["add subtract()", "run tests"])
    good = todo_ref(2, "run tests")
    bad_hash = f"2{'00' if good[1:] != '00' else 'ff'}"
    wrong = asyncio.run(tools.execute("plan", {"progress": bad_hash}))
    assert_true(wrong.startswith("Error"), wrong)
    assert_true(good in wrong, "error names the correct ref")
    assert_true("Current todos:" in wrong, wrong)
    assert_equal(tools.state.progress, 0, "mismatch does not move the cursor")

    out_of_range = asyncio.run(tools.execute("plan", {"progress": "9ab"}))
    assert_true("has 2 todos" in out_of_range, out_of_range)
    garbage = asyncio.run(tools.execute("plan", {"progress": "soon"}))
    assert_true(garbage.startswith("Error"), garbage)


def test_replan_requires_reason_and_splices_completed_prefix() -> None:
    tools = approved_tools(["add subtract()", "wire the CLI", "run tests"])
    asyncio.run(tools.execute("plan", {"progress": todo_ref(1, "add subtract()")}))

    no_reason = asyncio.run(tools.execute("plan", {"todos": ["new step"]}))
    assert_true("include a `reason`" in no_reason, no_reason)

    presented: list[str] = []
    original_callback = tools._plan_callback

    async def approve_and_record(plan: str) -> PlanDecision:
        presented.append(plan)
        return PlanDecision(kind="approve")

    tools.set_plan_callback(approve_and_record)
    version_before = tools.state.version
    result = asyncio.run(
        tools.execute(
            "plan",
            {"todos": ["expose a flag", "run tests"], "reason": "CLI scope grew"},
        )
    )
    assert_true("approved the revised plan" in result, result)
    assert_true("CLI scope grew" in presented[0], "modal carries the reason")
    assert_true("Already completed:" in presented[0], "modal shows finished work")
    assert_equal(
        tools.state.todos,
        ["add subtract()", "expose a flag", "run tests"],
        "completed prefix + new remaining steps",
    )
    assert_equal(tools.state.progress, 1, "cursor untouched by the splice")
    assert_equal(tools.state.version, version_before + 1, "replan bumps version")
    tools.set_plan_callback(original_callback)


def test_replan_reject_keeps_existing_plan_and_version() -> None:
    tools = approved_tools(["a", "b"])

    async def decline(plan: str) -> PlanDecision:
        return PlanDecision(kind="reject")

    tools.set_plan_callback(decline)
    version_before = tools.state.version
    result = asyncio.run(
        tools.execute("plan", {"todos": ["c"], "reason": "changed my mind"})
    )
    assert_true("declined the replan" in result, result)
    assert_true("Continue with the existing" in result, result)
    assert_equal(tools.state.todos, ["a", "b"], "existing plan kept")
    assert_equal(tools.state.version, version_before, "no version bump")


def test_plan_auto_approves_without_reviewer() -> None:
    warnings: list[str] = []
    tools = PlanTools(notify=warnings.append)
    result = asyncio.run(
        tools.execute("plan", {"plan": "p", "todos": ["only step"]})
    )
    assert_true("no interactive reviewer configured" in result, result)
    assert_equal(tools.last_decision.source, "auto", "machine decision flagged")
    assert_true(any("no reviewer" in w for w in warnings), warnings)
    assert_equal(tools.state.todos, ["only step"], "auto-approve commits")


def test_ask_user_question_flows_through_callback() -> None:
    seen: list[list[dict]] = []

    async def fake_modal(questions: list[dict]) -> dict:
        seen.append(questions)
        return {questions[0]["question"]: "Option B"}

    tools = PlanTools(question_callback=fake_modal)
    args = {
        "questions": [
            {
                "question": "Which storage layer?",
                "header": "Storage",
                "options": [
                    {"label": "Option A", "description": "sqlite"},
                    {"label": "Option B", "description": "flat files"},
                ],
            }
        ]
    }
    result = asyncio.run(tools.execute("ask_user_question", args))
    assert_equal(len(seen), 1, "callback invoked once")
    assert_true(
        result.startswith("User has provided the following answers:"), result
    )
    assert_true("**Storage**: Option B" in result, result)

    # Without a callback the model is told to decide by itself.
    solo = asyncio.run(PlanTools().execute("ask_user_question", args))
    assert_true("Choose the most sensible option yourself" in solo, solo)
    # Malformed args are rejected before any callback (qwen-code wording).
    bad = asyncio.run(tools.execute("ask_user_question", {"questions": []}))
    assert_equal(
        bad,
        'Parameter "questions" must contain between 1 and 4 questions.',
        "empty questions rejected",
    )


def test_ask_user_question_qwen_code_parity() -> None:
    from tools.plan.ask_user_question import (
        ASK_USER_QUESTION_SCHEMA,
        validate_questions,
    )

    params = ASK_USER_QUESTION_SCHEMA["parameters"]
    assert_equal(
        params["additionalProperties"], False, "top-level additionalProperties"
    )
    assert_true("metadata" in params["properties"], "metadata accepted for parity")
    question_props = params["properties"]["questions"]["items"]["properties"]
    assert_true("multiSelect" in question_props, "multiSelect in schema")
    assert_equal(question_props["multiSelect"]["default"], False, "default false")
    assert_true(
        "multiSelect: true" in ASK_USER_QUESTION_SCHEMA["description"],
        "description documents multiSelect",
    )

    def q(**overrides) -> dict:
        base = {
            "question": "Which one?",
            "header": "Pick",
            "options": [
                {"label": "A", "description": "a"},
                {"label": "B", "description": "b"},
            ],
        }
        base.update(overrides)
        return {"questions": [base]}

    assert_equal(validate_questions(q()), None, "valid args accepted")
    assert_equal(validate_questions(q(multiSelect=True)), None, "multiSelect ok")
    assert_equal(
        validate_questions({"questions": "nope"}),
        'Parameter "questions" must be an array.',
        "non-array rejected",
    )
    assert_equal(
        validate_questions(q(header="")),
        'Question 1: "header" must be a non-empty string.',
        "blank header rejected",
    )
    assert_equal(
        validate_questions(q(header="Way too long header")),
        None,
        "long header accepted (limit is schema guidance only)",
    )
    assert_equal(
        validate_questions(
            q(options=[{"label": "A", "description": "a"}, {"label": "B"}])
        ),
        'Question 1, Option 2: "description" must be a non-empty string.',
        "blank option description rejected",
    )
    assert_equal(
        validate_questions(q(multiSelect="yes")),
        'Question 1: "multiSelect" must be a boolean.',
        "non-bool multiSelect rejected",
    )
    assert_equal(
        validate_questions(q(options=[{"label": "A", "description": "a"}])),
        'Question 1: "options" must contain between 2 and 4 options.',
        "single option rejected",
    )


def test_ask_user_question_result_format() -> None:
    async def dismissing_modal(questions: list[dict]) -> dict:
        return {}

    args = {
        "questions": [
            {
                "question": "Which features?",
                "header": "Features",
                "options": [
                    {"label": "Alpha", "description": "a"},
                    {"label": "Beta", "description": "b"},
                ],
                "multiSelect": True,
            },
            {
                "question": "Which region?",
                "header": "Region",
                "options": [
                    {"label": "EU", "description": "eu"},
                    {"label": "US", "description": "us"},
                ],
            },
        ]
    }

    tools = PlanTools(question_callback=dismissing_modal)
    declined = asyncio.run(tools.execute("ask_user_question", args))
    assert_equal(
        declined, "User declined to answer the questions.", "decline wording"
    )

    async def partial_modal(questions: list[dict]) -> dict:
        # Multi-select answers arrive comma-joined; the second modal was
        # dismissed, so its question is simply absent from the answers.
        return {"Which features?": "Alpha, Beta"}

    tools = PlanTools(question_callback=partial_modal)
    result = asyncio.run(tools.execute("ask_user_question", args))
    assert_equal(
        result,
        "User has provided the following answers:\n\n**Features**: Alpha, Beta",
        "answers keyed by header, unanswered questions skipped",
    )


def test_explore_and_resume_schemas() -> None:
    assert_equal(EXPLORE_SCHEMA["name"], EXPLORE_NAME, "explore wire name")
    assert_equal(
        EXPLORE_SCHEMA["parameters"]["required"],
        ["task"],
        "explore requires the task description",
    )
    assert_equal(RESUME_SCHEMA["name"], RESUME_NAME, "resume wire name")
    assert_equal(
        RESUME_SCHEMA["parameters"]["required"],
        ["summary"],
        "resume requires the findings summary",
    )
    assert_true(
        "self-contained" in RESUME_SCHEMA["description"],
        "resume description demands a self-contained report",
    )


def test_subedit_schema_and_validation() -> None:
    assert_equal(EDIT_SCHEMA["name"], "edit", "freestyle edit wire name")
    # line_ranges is optional now (the editor sees the whole file when omitted),
    # so only filename + instructions are required.
    assert_equal(
        sorted(EDIT_SCHEMA["parameters"]["required"]),
        ["filename", "instructions"],
        "filename and instructions required; line_ranges optional",
    )
    assert_equal(
        validate_edit_args(
            {"filename": "a.py", "line_ranges": "3-9", "instructions": "rename x to y"}
        ),
        None,
        "valid args with a range accepted",
    )
    assert_equal(
        validate_edit_args({"filename": "a.py", "instructions": "rename x to y"}),
        None,
        "line_ranges may be omitted",
    )
    assert_true(
        validate_edit_args({"filename": "a.py", "line_ranges": "3-9"}).startswith("Error"),
        "missing instructions rejected",
    )


def main() -> None:
    test_plan_schemas_use_wire_names()
    test_plan_present_fires_gate_and_commits()
    test_plan_present_requires_plan_text_first_cycle()
    test_plan_revise_and_reject_do_not_commit()
    test_plan_call_shape_validation()
    test_ask_user_question_dead_only_after_approval()
    test_progress_advances_reopens_and_zero_resets()
    test_work_dispatches_retry_reopen_and_rejects_skip_ahead()
    test_progress_ref_mismatch_rerenders_list()
    test_replan_requires_reason_and_splices_completed_prefix()
    test_replan_reject_keeps_existing_plan_and_version()
    test_plan_auto_approves_without_reviewer()
    test_ask_user_question_flows_through_callback()
    test_ask_user_question_qwen_code_parity()
    test_ask_user_question_result_format()
    test_explore_and_resume_schemas()
    test_subedit_schema_and_validation()
    print("plan/explore/subedit tools tests passed")


if __name__ == "__main__":
    main()
