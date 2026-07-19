"""PlanTools: dispatch + callbacks for the planner stage's two tools."""

from __future__ import annotations

from typing import Awaitable, Callable

from .ask_user_question import ASK_USER_QUESTION_SCHEMA, validate_questions
from .plan_tool import (
    PLAN_SCHEMA,
    PlanDecision,
    PlanState,
    parse_progress_ref,
    parse_work_ref,
    render_todos,
    todo_ref,
)

# Receives the validated `questions` array, returns the user's answers as
# {question_text: answer_text}. Mirrors the bash ApprovalCallback pattern.
QuestionCallback = Callable[[list[dict]], Awaitable[dict]]
# Receives the plan markdown, returns the user's PlanDecision.
PlanApprovalCallback = Callable[[str], Awaitable[PlanDecision]]


class PlanTools:
    """The planner's toolset. `plan` mutates the client-side PlanState and
    raises the approval gate; `ask_user_question` resolves through a UI
    callback wired by the app (unset callbacks fail safe with an explanatory
    tool result)."""

    def __init__(
        self,
        state: PlanState | None = None,
        question_callback: QuestionCallback | None = None,
        plan_callback: PlanApprovalCallback | None = None,
        notify: Callable[[str], None] | None = None,
    ) -> None:
        self.state = state if state is not None else PlanState()
        self._question_callback = question_callback
        self._plan_callback = plan_callback
        # Surface fallback decisions (no callback bound) to the user; the
        # machine must never substitute a user decision silently.
        self.notify = notify

    # The most recent gate outcome, for the orchestrator's transition.
    @property
    def last_plan(self) -> str | None:
        return self.state.last_plan

    @last_plan.setter
    def last_plan(self, value: str | None) -> None:
        self.state.last_plan = value

    @property
    def last_decision(self) -> PlanDecision | None:
        return self.state.last_decision

    @last_decision.setter
    def last_decision(self, value: PlanDecision | None) -> None:
        self.state.last_decision = value

    def set_question_callback(self, callback: QuestionCallback) -> None:
        self._question_callback = callback

    def set_plan_callback(self, callback: PlanApprovalCallback) -> None:
        self._plan_callback = callback

    def schemas(self) -> list[dict]:
        return [
            {"type": "function", "function": PLAN_SCHEMA},
            {"type": "function", "function": ASK_USER_QUESTION_SCHEMA},
        ]

    async def execute(self, name: str, arguments: dict) -> str:
        if name == "plan":
            return await self._plan(arguments)
        if name == "ask_user_question":
            return await self._ask(arguments)
        return f"Error: unknown plan tool {name!r}."

    async def _plan(self, arguments: dict) -> str:
        has_todos = isinstance(arguments.get("todos"), list)
        has_progress = arguments.get("progress") not in (None, "")
        has_work = arguments.get("work") not in (None, "")
        if has_todos + has_progress + has_work > 1:
            return (
                "Error: pass only one of `todos` (present/replace the plan), "
                "`progress` (mark completion), or `work` (dispatch an "
                "executor to a todo), not several."
            )
        if has_progress:
            return self._progress(arguments)
        if has_work:
            return self._work(arguments)
        if has_todos:
            return await self._present(arguments)
        return (
            "Error: the `plan` call needs either `plan`+`todos` (present the "
            "plan), `progress` (mark completion), `work` (dispatch an "
            "executor to a todo), or `todos`+`reason` (replan)."
        )

    async def _ask(self, arguments: dict) -> str:
        if self.state.approved:
            return (
                "Error: ask_user_question is no longer available — the plan "
                "has already been approved. Continue with the `plan` tool."
            )
        error = validate_questions(arguments)
        if error is not None:
            return error
        if self._question_callback is None:
            if self.notify is not None:
                self.notify("question skipped: no interactive user is bound")
            return (
                "No interactive user is available to answer questions right now. "
                "Choose the most sensible option yourself and note the assumption."
            )
        questions = arguments["questions"]
        answers = await self._question_callback(questions)
        if not answers:
            return "User declined to answer the questions."
        lines = []
        for number, question in enumerate(questions, start=1):
            answer = answers.get(str(question.get("question", "")))
            if answer is None:
                continue
            header = question.get("header") or f"Question {number}"
            lines.append(f"**{header}**: {answer}")
        if not lines:
            return "User declined to answer the questions."
        return "User has provided the following answers:\n\n" + "\n".join(lines)

    async def _present(self, arguments: dict) -> str:
        state = self.state
        todos = arguments.get("todos")
        if not todos:
            return "Error: 'todos' must be a non-empty array."
        cleaned: list[str] = []
        for index, item in enumerate(todos):
            if not isinstance(item, str) or not item.strip():
                return f"Error: todos[{index}] must be a non-empty string."
            cleaned.append(item.strip())
        plan_text = arguments.get("plan")
        plan_text = plan_text.strip() if isinstance(plan_text, str) else ""
        reason = arguments.get("reason")
        reason = reason.strip() if isinstance(reason, str) else ""
        replan = state.approved
        if not replan and not plan_text:
            return (
                "Error: include the `plan` markdown clarifying the task and "
                "approach (todos alone are not a reviewable plan)."
            )
        if replan and not reason:
            return (
                "Error: the plan is already approved; to replace the "
                "remaining todos include a `reason` explaining what changed."
            )

        if replan:
            completed = state.todos[: state.progress]
            new_todos = completed + cleaned
            body = plan_text or "(plan text unchanged)"
            modal_parts = [f"Replan — reason: {reason}"]
            if completed:
                modal_parts.append(
                    "Already completed:\n"
                    + render_todos(completed, len(completed))
                )
            modal_parts.append(
                f"Revised remaining steps:\n{body}\n\n"
                + render_todos(new_todos, state.progress)
            )
            modal_text = "\n\n".join(modal_parts)
        else:
            new_todos = cleaned
            modal_text = f"{plan_text}\n\n{render_todos(new_todos, 0)}"

        if self._plan_callback is None:
            if self.notify is not None:
                self.notify("plan auto-approved: no reviewer is bound")
            decision = PlanDecision(kind="approve", source="auto")
        else:
            decision = await self._plan_callback(modal_text)
        state.last_decision = decision

        if decision.kind == "approve":
            state.todos = new_todos
            if not replan:
                state.progress = 0
                state.approved = True
            state.version += 1
            if plan_text:
                state.last_plan = plan_text
            rendered = render_todos(state.todos, state.progress, active=True)
            if decision.source == "auto":
                verdict = "Plan approved (no interactive reviewer configured)."
            elif replan:
                verdict = "The user approved the revised plan. Execution continues."
            else:
                verdict = "The user approved the plan. Execution will start now."
            return f"Todo list updated:\n{rendered}\n\n{verdict}"
        if decision.kind == "revise":
            return (
                "The user asked for changes before approving the plan: "
                f"{decision.text}\nRevise the plan and call `plan` again."
            )
        if replan:
            rendered = render_todos(state.todos, state.progress)
            return (
                "The user declined the replan. Continue with the existing "
                f"plan:\n{rendered}"
            )
        return "The user rejected the plan. Do not start executing it."

    def _work(self, arguments: dict) -> str:
        state = self.state
        if not state.approved:
            return (
                "Error: no approved plan exists yet; present one with "
                "`plan` + `todos` first."
            )
        result = parse_work_ref(arguments.get("work"), state.todos, state.progress)
        if isinstance(result, str):
            return result
        reopened = result < state.progress
        state.progress = result
        state.version += 1
        rendered = render_todos(state.todos, state.progress, active=True)
        content = state.todos[result]
        tail = (
            "The next executor is dispatched to "
            f"{todo_ref(result + 1, content)}: {content}."
        )
        if reopened:
            tail += " Completed todos from there on are reopened."
        return f"Todo list updated:\n{rendered}\n\n{tail}"

    def _progress(self, arguments: dict) -> str:
        state = self.state
        if not state.approved:
            return (
                "Error: no approved plan exists yet; present one with "
                "`plan` + `todos` first."
            )
        result = parse_progress_ref(
            arguments.get("progress"), state.todos, state.progress
        )
        if isinstance(result, str):
            return result
        state.progress = result
        state.version += 1
        rendered = render_todos(state.todos, state.progress, active=True)
        next_index = state.next_index()
        if next_index is None:
            tail = "All todos are completed."
        else:
            content = state.todos[next_index]
            tail = f"Next: {todo_ref(next_index + 1, content)} {content}"
        return f"Todo list updated:\n{rendered}\n\n{tail}"
