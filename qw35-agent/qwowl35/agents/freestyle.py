"""The freestyle executor: bash + edit, the agent that actually does work.

Toolset: exactly `bash` and `edit` — file changes are delegated to the editor
sub-agent through `edit`.

Two shapes: the whole task at once (NORMAL mode, fresh context per turn), and
one todo at a time after plan approval (PLAN mode) — where consecutive todos
CONTINUE one persistent conversation: the first todo opens it with the goal
and the approved plan, each later todo appends a slim continuation directive,
and the full history of the work so far rides along (and rides the
main-session checkpoint stack, so each todo prefills only its directive).
"""

from __future__ import annotations

from tools.bash import GUIDANCE_EXECUTOR as BASH_GUIDANCE_EXECUTOR
from tools.bash import SHELL_NAME
from tools.subedit import EDIT_NAME
from tools.subedit import GUIDANCE as SUBEDIT_GUIDANCE

from .base import SESSION_MAIN, XML_CALL_RULES, AgentSpec, compose_system_message

SPEC = AgentSpec(
    name="execute",
    session=SESSION_MAIN,
    allowed_tools=frozenset({SHELL_NAME, EDIT_NAME}),
    mascot="bash",
    # This agent never sees hashline anchors: its post-bash-write feedback is
    # the plain validation report naming its `edit` delegator tool.
    write_feedback="subedit",
)

SYSTEM_PROMPT = f"""\
You are qwowl35, a coding agent. Solve the task you are given: inspect,
create files, and run/verify with `run_shell_command`; change existing files
with `edit`. Those are your only two tools. Use relative paths. Be concise;
verify before declaring done.

{BASH_GUIDANCE_EXECUTOR}

{SUBEDIT_GUIDANCE}

When the task is complete and verified, finish by replying (no tool call)
with a brief summary of what was done.
<<GROUNDING>>

{XML_CALL_RULES}"""


def system_message(cwd: str | None = None) -> dict:
    return compose_system_message(SYSTEM_PROMPT, cwd)


def build_direct_message(goal: str, session_notes: str = "") -> dict:
    parts = [f"Task:\n{goal}"]
    if session_notes:
        parts.append(f"Earlier in this session:\n{session_notes}")
    return {"role": "user", "content": "\n\n".join(parts)}


_TASK_DIRECTIVE = (
    "Complete ONLY this task now — do NOT start work on any other task in "
    "the plan, even if it looks quick or related. When this task is done and "
    "verified, finish by replying (no tool call) with a brief result summary "
    "for it."
)


def build_task_message(
    goal: str,
    plan_block: str,
    index: int,
    total: int,
    todo: str,
    guidance: str = "",
) -> dict:
    """The FIRST todo's opener: the goal, the approved plan, the planner's
    brief guidance, and exactly the task at hand. Later todos append
    :func:`build_continuation_message` to the same conversation — the
    history itself carries what was already done."""
    parts = [f"Overall goal:\n{goal}", plan_block.strip()]
    if guidance.strip():
        parts.append(f"Planner guidance for this task:\n{guidance.strip()}")
    parts.append(f"Your task ({index}/{total}): {todo}\n{_TASK_DIRECTIVE}")
    return {"role": "user", "content": "\n\n".join(parts)}


def build_continuation_message(
    index: int,
    total: int,
    todo: str,
    guidance: str = "",
    plan_update: str = "",
) -> dict:
    """A later todo's directive on the SAME conversation: slim by design —
    the goal, plan, and completed work already live in the inherited context.
    ``plan_update`` carries the refreshed plan block after a replan."""
    parts = []
    if plan_update.strip():
        parts.append(f"The plan was revised and re-approved:\n{plan_update.strip()}")
    if guidance.strip():
        parts.append(f"Planner guidance for this task:\n{guidance.strip()}")
    parts.append(f"Previous task done. Your next task ({index}/{total}): {todo}\n{_TASK_DIRECTIVE}")
    return {"role": "user", "content": "\n\n".join(parts)}


def task_result(index: int, total: int, todo: str, summary: str) -> str:
    text = summary.strip() or "(no summary provided)"
    return f"Task {index}/{total} ({todo}) result:\n{text}"
