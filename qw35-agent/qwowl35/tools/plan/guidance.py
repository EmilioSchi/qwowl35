"""System-prompt guidance for the planner tools."""

from __future__ import annotations

GUIDANCE = """\
Decompose the task into the FEWEST ordered todos that keep each step
independently executable and verifiable (typically 2-5), in dependency
order: each todo one short imperative sentence covering one coherent unit
of work, none overlapping another. Never split one small change into many
micro-todos.
The `plan` text clarifies the user's task — goal, context, constraints —
the approach, AND the key exploration findings (exact file paths, line
references, the snippets that matter): executors receive ONLY this text
and the todo list, never the exploration results. It must not repeat the
todo list (it is shown and handed off together with the todos).

The `plan` tool has four uses:
1. Present the plan: plan="task + approach markdown", todos=["step 1", ...].
   The user approves, rejects, or sends revision feedback; on feedback,
   revise and call `plan` again. Work never starts before approval.
2. Mark progress (after approval): progress="<ref>" where <ref> is copied
   exactly from the rendered todo list (e.g. 3a7 = todos 1-3 done; the
   next todo starts). "0" reopens all; an earlier ref reopens later todos.
   Never resend `todos` just to mark progress.
3. Replan (after approval): todos=[remaining steps], reason="what changed
   and why". Completed steps are kept automatically; the user re-approves.
4. Dispatch (after approval): work="<ref>" sends the next executor to that
   todo — cite the todo that just ran to retry it, or an earlier one to
   redo it (completed todos from there on reopen). Never a todo that has
   not run yet.

If a decision is genuinely the user's to make, ask with `ask_user_question`
(2-4 concrete options) BEFORE the plan is approved — including to clarify
revision feedback — it is unavailable once the plan is approved."""
