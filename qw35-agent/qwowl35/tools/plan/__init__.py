"""Planner-stage tools: `plan` and `ask_user_question`.

`ask_user_question` replicates qwen-code (pinned commit 7417805,
packages/core/src/tools/askUserQuestion.ts) so the model drives it
on-distribution. `plan` is the unified planning tool — present/replan
(the approval gate) plus hashline-ref progress marks and executor
dispatches over a client-side PlanState; UI interactions raise through
pluggable async callbacks, mirroring the bash approval-callback pattern.
"""

from __future__ import annotations

from .ask_user_question import ASK_USER_QUESTION_SCHEMA
from .guidance import GUIDANCE
from .plan_tool import (
    PLAN_SCHEMA,
    PlanDecision,
    PlanState,
    parse_progress_ref,
    parse_work_ref,
    render_todos,
    todo_ref,
)
from .tools import PlanTools

PLAN_NAME = "plan"
ASK_USER_QUESTION_NAME = "ask_user_question"

PLAN_TOOL_NAMES = frozenset({PLAN_NAME, ASK_USER_QUESTION_NAME})

__all__ = [
    "GUIDANCE",
    "PLAN_TOOL_NAMES",
    "PLAN_NAME",
    "ASK_USER_QUESTION_NAME",
    "PLAN_SCHEMA",
    "ASK_USER_QUESTION_SCHEMA",
    "PlanDecision",
    "PlanState",
    "PlanTools",
    "parse_progress_ref",
    "parse_work_ref",
    "render_todos",
    "todo_ref",
]
