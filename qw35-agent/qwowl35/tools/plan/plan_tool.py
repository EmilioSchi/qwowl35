"""`plan` — the unified planning tool (schema + client-side state).

One tool, four call shapes, mode inferred from the arguments:
present (`plan` + `todos`), progress (`progress` ref only), work-dispatch
(`work` ref only), and replan (`todos` + `reason` after approval).
Presenting or replanning IS the approval gate; progress marks and work
dispatches cite hashline-style refs so the number and the task it claims
are cross-checked, never a bare index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tools.files.hashline.hash import format_line_ref, short_hash_value

PLAN_SCHEMA = {
    "name": "plan",
    "description": (
        "Create, revise, or advance the execution plan. "
        "(1) Present: pass `plan` (markdown clarifying the user's task and "
        "intended approach) and `todos` (ordered list of step strings, "
        "dependency order); the user must approve before work starts. "
        "(2) Progress: pass ONLY `progress` with the ref of the last "
        "completed todo copied exactly from the rendered list (e.g. '3a7' "
        "means todos 1-3 are done); '0' reopens everything; an earlier ref "
        "reopens the todos after it. Never resend `todos` to mark progress. "
        "(3) Replan (after approval): pass new `todos` covering ONLY the "
        "remaining work plus `reason` explaining the change; completed steps "
        "are kept automatically and the user must re-approve. "
        "(4) Dispatch: pass ONLY `work` with the ref of the todo the next "
        "executor should run — the todo that just ran (retry) or an earlier "
        "one (redo; completed todos from there on reopen)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": (
                    "Concise markdown clarifying the task the user asked "
                    "for — the goal, relevant context and constraints, and "
                    "the intended approach/outcome. Do NOT restate the "
                    "todos; they are rendered right below it, and this text "
                    "is handed to every executor as shared context. "
                    "Required when presenting the first plan; optional when "
                    "replanning."
                ),
            },
            "todos": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "description": (
                    "The ordered steps, one short imperative sentence each, "
                    "in dependency order. Each step must be independently "
                    "executable and verifiable."
                ),
            },
            "progress": {
                "type": "string",
                "description": (
                    "Ref of the last completed todo (its number followed by "
                    "its 2-hex hash, e.g. '3a7'), asserting todos 1..N "
                    "complete. '0' reopens all. Mutually exclusive with "
                    "`todos` and `work`."
                ),
            },
            "work": {
                "type": "string",
                "description": (
                    "Ref of the todo to dispatch the next executor to, "
                    "copied exactly from the rendered list. Use it to retry "
                    "the todo that just ran or to redo an earlier one "
                    "(completed todos from there on reopen); it cannot skip "
                    "ahead past todos that have not run. Mutually exclusive "
                    "with `todos` and `progress`."
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    "Why the plan changed. Required when replacing the todos "
                    "of an already-approved plan."
                ),
            },
        },
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class PlanDecision:
    """The user's verdict on a presented plan.

    kind: "approve" | "revise" | "reject". `text` carries the revision
    feedback for "revise" (empty otherwise).
    """

    kind: str
    text: str = ""
    # "user" when a real reviewer decided; "auto" when the machine had to
    # decide because no reviewer was bound. Auto decisions are surfaced
    # loudly — the machine must never silently impersonate the user.
    source: str = "user"


@dataclass
class PlanState:
    """The session's plan: ordered todo strings plus one absolute cursor.

    Statuses are DERIVED, never stored: todos[:progress] are completed, the
    todo at `progress` is in_progress while an executor runs it, the rest
    are pending. `version` counts committed writes (approved present/replan,
    progress mark) — the orchestrator's review-ping fallback needs to know
    whether the planner wrote AT ALL, and content comparison cannot tell an
    untouched cursor from an explicitly re-asserted identical one."""

    todos: list[str] = field(default_factory=list)
    progress: int = 0
    version: int = 0
    last_plan: str | None = None
    last_decision: PlanDecision | None = None
    # An approve decision committed todos — the point after which
    # ask_user_question is no longer available (revise feedback may still
    # need clarifying questions before then).
    approved: bool = False

    def next_index(self) -> int | None:
        return self.progress if self.progress < len(self.todos) else None


def todo_ref(index_1based: int, content: str) -> str:
    """Hashline-style locator: 1-based position + 2-hex content hash."""
    return format_line_ref(index_1based, short_hash_value(content))


_STATUS_MARKS = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}


def render_todos(todos: list[str], progress: int, active: bool = False) -> str:
    """`[x]/[>]/[ ] <ref>: <content>` lines; `active` marks todos[progress]
    in_progress (an executor is on it or its review is running)."""
    lines = []
    for i, content in enumerate(todos):
        if i < progress:
            mark = _STATUS_MARKS["completed"]
        elif active and i == progress:
            mark = _STATUS_MARKS["in_progress"]
        else:
            mark = _STATUS_MARKS["pending"]
        lines.append(f"{mark} {todo_ref(i + 1, content)}: {content}")
    return "\n".join(lines)


_REF_RE = re.compile(r"^(\d+)([0-9a-f]{2})$")


def _parse_ref(field: str, value: object, todos: list[str], progress: int) -> int | str:
    """The 1-based todo index a ref names, or an error string re-rendering
    the list.

    Accepts int or str (the XML call parser may deliver either). The leading
    digits are the 1-based todo index and the trailing 2 hex chars its
    content hash: position disambiguates cross-todo hash collisions, the
    hash cross-checks that the cited index is the todo the model thinks it
    is."""
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        return _ref_error(f"{field} must be a todo ref string", todos, progress)
    text = value.strip().lower()
    match = _REF_RE.match(text)
    if match is None:
        return _ref_error(
            f"{field} ref {value!r} is not a valid ref", todos, progress
        )
    index = int(match.group(1))
    if not 1 <= index <= len(todos):
        return _ref_error(
            f"{field} ref {value!r} points at todo {index}, but the list "
            f"has {len(todos)} todos",
            todos,
            progress,
        )
    expected = todo_ref(index, todos[index - 1])
    if text != expected:
        return _ref_error(
            f"{field} ref {value!r} does not match the current todo list "
            f'(todo {index} is "{todos[index - 1]}", ref {expected})',
            todos,
            progress,
        )
    return index


def parse_progress_ref(value: object, todos: list[str], progress: int) -> int | str:
    """The new progress cursor, or an error string re-rendering the list.

    '0'/0 means nothing completed; otherwise the ref names the last
    completed todo."""
    if isinstance(value, int):
        value = str(value)
    if isinstance(value, str) and value.strip() == "0":
        return 0
    return _parse_ref("progress", value, todos, progress)


def parse_work_ref(value: object, todos: list[str], progress: int) -> int | str:
    """The new progress cursor (0-based index of the todo to execute next),
    or an error string re-rendering the list.

    Statuses derive from the single cursor, so dispatching past the first
    pending todo would silently mark the skipped ones completed — the ref
    may name any already-run todo or the next pending one, nothing beyond."""
    index = _parse_ref("work", value, todos, progress)
    if isinstance(index, str):
        return index
    if index > progress + 1:
        return _ref_error(
            f"work ref {todo_ref(index, todos[index - 1])} skips ahead — "
            f"todos {progress + 1}-{index - 1} have not run yet; dispatch "
            "the next pending todo, or replan",
            todos,
            progress,
        )
    return index - 1


def _ref_error(what: str, todos: list[str], progress: int) -> str:
    rendered = render_todos(todos, progress) if todos else "(no todos)"
    return (
        f"Error: {what}. Current todos:\n{rendered}\n"
        "Cite a ref exactly as rendered above, or '0' to reopen all."
    )
