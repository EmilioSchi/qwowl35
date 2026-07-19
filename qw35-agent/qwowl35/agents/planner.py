"""The planner: turn the goal into an approved plan, exploring on demand.

Fresh context: its own system prompt + the task. Toolset: exactly `plan`,
`ask_user_question`, and `explore` — it cannot search, run, or edit anything
itself; codebase facts come from the explorer sub-agents it spawns through
`explore`.
"""

from __future__ import annotations

from tools.plan import (
    ASK_USER_QUESTION_NAME,
    GUIDANCE as PLAN_GUIDANCE,
    PLAN_NAME,
)

from .base import SESSION_PLAN, XML_CALL_RULES, AgentSpec, compose_system_message
from .explorer import EXPLORE_NAME

# The planner lives on its own GPU session: its context persists across the
# whole execution phase, so each plan↔execute alternation prefills only the
# new dialogue message instead of the whole planning context.
SPEC = AgentSpec(
    name="planner",
    session=SESSION_PLAN,
    allowed_tools=frozenset({PLAN_NAME, ASK_USER_QUESTION_NAME, EXPLORE_NAME}),
    mascot="planner",
)

# During a post-task review ping the planner's job is circumscribed to
# progress marks (or a replan) plus one guidance line: no questions.
REVIEW_TOOLS = frozenset({PLAN_NAME})

# Planning round budget (model streams), effort-scaled: the plan stage once
# thrashed unboundedly (13 todo rewrites) until the user force-quit. Sized to
# cover the planner's `explore` calls too — exploration now spends planner
# rounds instead of a separate stage's budget.
PLAN_EFFORT_BUDGET_ROUNDS = {"low": 8, "medium": 14, "high": 20, "xhigh": 28}
DEFAULT_PLAN_BUDGET_ROUNDS = PLAN_EFFORT_BUDGET_ROUNDS["medium"]
# A review ping is ONE bounded exchange: the plan call + the guidance line.
REVIEW_MAX_ROUNDS = 2
# Guidance handed to the next executor stays a brief line.
GUIDANCE_MAX_CHARS = 300

# qwen-trained handoff wording, adapted to the two-agent split.
APPROVAL_HANDOFF = (
    "The user approved the plan. Executors will now run the todos one at a "
    "time; after each task you will review the result and record completion "
    "with plan(progress=...) — or replan if the approach must change."
)


def plan_rounds(reasoning_effort: str | None) -> int:
    return PLAN_EFFORT_BUDGET_ROUNDS.get(
        (reasoning_effort or "").lower(), DEFAULT_PLAN_BUDGET_ROUNDS
    )


def build_review_message(
    position: int,
    content: str,
    ref: str,
    summary: str,
    executor_ok: bool,
    todos_rendered: str,
) -> dict:
    """The brief executor→planner dialogue: report + qwen completion doctrine.

    Carries the CURRENT rendered list: the machine moves the cursor behind
    the planner's back (fallback completions, clamps), and a progress mark
    citing a stale ref would be rejected against the machine's state."""
    report = " ".join((summary or "(no summary)").split())
    if len(report) > 700:
        report = report[:700] + " ..."
    status_note = "" if executor_ok else " The executor stopped before confirming completion."
    return {
        "role": "user",
        "content": (
            f"Task {ref} (\"{content}\") finished.{status_note} "
            f"Executor report: {report}\n\n"
            f"Current todo list:\n{todos_rendered}\n\n"
            "If the task is fully accomplished — never with failing tests or "
            f"partial work — record it by calling plan with progress=\"{ref}\". "
            "If this task already finished ALL the remaining work, cite the "
            "ref of the LAST todo instead — execution stops there. If it is "
            f"NOT done, call plan with work=\"{ref}\" to send a fresh "
            "executor back to it — a review without any plan call counts as "
            "done. If the plan itself must change, call plan with new "
            "remaining `todos` and a `reason` — the user will re-approve. "
            "Then reply with one short guidance line for the next task, or "
            "state the goal is reached."
        ),
    }


def trim_guidance(text: str) -> str:
    """One brief line of planner guidance for the next executor."""
    cleaned = " ".join((text or "").split())
    return cleaned[:GUIDANCE_MAX_CHARS]

SYSTEM_PROMPT = f"""\
You are the qwowl35 planner: you break the task into a structured, ordered execution plan; separate executor agents carry it out afterwards, one todo at a time.

You operate in a strict READ-ONLY sandbox. You are fundamentally incapable of making changes, writing code. Your only job is to think, map, and plan.

You have exactly three tools: `plan`, `ask_user_question`, and `explore` — you cannot search, run commands, or edit files yourself. When you need codebase facts (what exists, where, how it works), call `explore` with a detailed task description; a read-only explorer sub-agent investigates and returns a findings summary as the call's result. Explore BEFORE presenting the plan — never plan against guessed file names or structures.

{PLAN_GUIDANCE}

After the plan is approved, finish by replying (no tool call) with the
one-line confirmation "Plan approved".
<<GROUNDING>>

{XML_CALL_RULES}"""


def system_message(cwd: str | None = None) -> dict:
    return compose_system_message(SYSTEM_PROMPT, cwd)


def build_task_message(goal: str, session_notes: str = "") -> dict:
    parts = [f"Task:\n{goal}"]
    if session_notes:
        parts.append(f"Earlier in this session:\n{session_notes}")
    return {"role": "user", "content": "\n\n".join(parts)}


def handoff(plan: str, todos_rendered: str) -> str:
    """The approved plan + todo checklist, carried into every execution task."""
    return f"Approved plan:\n{plan.strip()}\n\nTodo list:\n{todos_rendered}"
