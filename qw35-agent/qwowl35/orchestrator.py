"""Mode dispatch: one agent per user-selected TUI mode.

The user picks the mode BEFORE sending the prompt (Shift+Tab or /mode); one
turn then runs exactly one dispatcher: NORMAL → the freestyle executor,
PLAN → planner ⇄ executors, WEB → the web agent, CHAT → the chat agent.

Every agent is fully segregated: it runs a FRESH context made of its own
system prompt plus exactly the material handed over to it, and it advertises
only its own tools (the registry swaps the wire toolset per stage). Two
sub-agents ride inside a stage as tool calls:

- the executor's `edit` spawns the EDITOR (scratch session, INSERT mode box);
- the planner's `explore` spawns a stateless EXPLORER (scratch session,
  VISUAL mode box) that searches freely and reports back through one
  `resume` call — only that summary returns to the planner.

In PLAN mode the executors CONTINUE one persistent conversation on the main
session (each todo appends a slim directive, so the checkpoint stack prefills
only the new message), while the planner's own context persists on the plan
session across the whole execution phase.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import re
from pathlib import Path

from agent import AgentUI, BudgetDecision, TurnRunner
from agents import chat as chat_agent
from agents import editor as editor_agent
from agents import explorer as explorer_agent
from agents import freestyle as freestyle_agent
from agents import planner as planner_agent
from agents import spawn_context
from agents import web as web_agent
from agents.base import SESSION_SCRATCH, AgentSpec
from agents.pipeline import ExplorerRegistry, PipelineRegistry
from client import Qw35Error
from config import Config
from modes import USER_MODES, Mode
from sessions.store import SessionStore, TurnDir
from tools.compress.rerank import set_default_scorer_mode, set_rerank_base_url
from tools.diagnostics import (
    DiagnosticsMemory,
    split_trailing_section,
    validation_report_with_memory,
)
from tools.files.adapter import TOOL_ATTENTION_MARKER, HashlineTools
from tools.files.hashline.grep import GREP_FILE_NAME, GREP_FILE_SCHEMA, run_file_grep
from tools.lsp import LSP_NAME, LSP_SCHEMA, LspQueryTool
from tools.syntax import validate_file
from tools.plan import PlanState, render_todos, todo_ref
from tools_registry import ApprovalCallback

import mascot

# Cap the editor's report inside the executor's `edit` tool result.
EDITOR_REPORT_MAX_CHARS = 6000
# The hashline anchor-state block a mutation result carries ("Current x.py
# (ids, ...):" + <line><hash>| rows). Anchors are the EDITOR's working
# vocabulary and must never reach the executor's context.
_ANCHOR_BLOCK = re.compile(r"\nCurrent [^\n]*\(ids[^\n]*\):\n(?:[^\n]*\|[^\n]*\n?)*")
# How many earlier turns feed the "Earlier in this session" note.
SESSION_NOTE_TURNS = 3

# The mode box shown while a stage runs, by stage name. Sub-agents override
# it for their lifetime (editor → INSERT, explorer → VISUAL) and restore.
STAGE_MODES = {
    "chat": Mode.CHAT,
    "web": Mode.WEB,
    "planner": Mode.PLAN,
    "execute": Mode.NORMAL,
}


class EditorRegistry:
    """The editor's toolset: hashline replace/insert/delete over one shared
    HashlineTools engine, plus three read-only lookups — `read_file` (re-open
    or page a file for fresh line ids), the `lsp` navigation tool, and the
    single-file `grep_search` (the editor's basic variant, id-row output; not
    the explorer's tree-walking grep). Records every mutation
    result so the orchestrator can report the applied diffs back to the
    executor; lookups are never recorded (they are not edits and must not
    count toward "N edits applied" or flip no-op detection)."""

    def __init__(self, files: HashlineTools, lsp: LspQueryTool | None = None) -> None:
        self.files = files
        self.lsp = lsp
        self.results: list[tuple[str, str]] = []
        self.saw_attention = False

    @property
    def diag_memory(self):
        """The active diagnostics memory (the shared files engine's), so
        runner paths that read ``registry.diag_memory`` see the editor's."""
        return getattr(self.files, "diag_memory", None)

    def schemas(self) -> list[dict]:
        schemas = list(self.files.schemas())
        if self.lsp is not None:
            schemas.append({"type": "function", "function": LSP_SCHEMA})
        schemas.append({"type": "function", "function": GREP_FILE_SCHEMA})
        return schemas

    async def execute(self, name: str, arguments: dict) -> str:
        if name == "read_file":
            # Read-only like lsp/grep: dispatched but never recorded — a read
            # is not an edit, so it must not count toward "N edits applied",
            # flip no-op detection, or set the sticky attention flag (the
            # in-band marker stays in the returned string, so the runner still
            # surfaces a broken file's syntax block as an error to the model).
            return await asyncio.to_thread(self.files.execute, name, arguments)
        if name == GREP_FILE_NAME:
            # Read-only like lsp: dispatched directly (never through the
            # compression layer), never recorded, never flips attention.
            return await asyncio.to_thread(run_file_grep, arguments)
        if name == LSP_NAME and self.lsp is not None:
            # The editor speaks hashline ids: render result positions in its
            # own file as <line><hash>|content (copy — never mutate the
            # model's parsed args). Explorer lsp calls stay plain line:col.
            return await asyncio.to_thread(
                self.lsp.execute, {**arguments, "_hashline": True}
            )
        result = await asyncio.to_thread(self.files.execute, name, arguments)
        recorded = result
        if recorded.startswith(TOOL_ATTENTION_MARKER):
            recorded = recorded[len(TOOL_ATTENTION_MARKER):]
            self.saw_attention = True
        self.results.append((name, recorded))
        return result


class Orchestrator:
    """Drop-in for Agent at the app level: run_turn(text, mode) + clear()."""

    def __init__(
        self,
        client,
        config: Config,
        ui: AgentUI,
        approval: ApprovalCallback | None = None,
        restricted_bash: bool = False,
        session_store: SessionStore | None = None,
        question_callback=None,
        plan_callback=None,
        explorer_budget_callback: explorer_agent.ExplorerBudgetCallback | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.ui = ui
        # All interactive callbacks bind at construction (the bash-approval
        # pattern): a built orchestrator is either fully interactive or its
        # fallbacks announce themselves — never silently non-interactive.
        self.registry = PipelineRegistry(
            approval=approval,
            restricted_bash=restricted_bash,
            compress=config.compress,
            rerank=config.rerank,
            question_callback=question_callback,
            plan_callback=plan_callback,
        )
        self.registry.plan.notify = ui.set_warning
        set_default_scorer_mode(config.rerank_scorer)
        # The cross-encoder scorer rides the same server process as chat.
        set_rerank_base_url(config.base_url)
        self.registry.set_editor(self._run_editor)
        self.registry.set_explorer(self._run_explorer)
        # Same fail-open convention as question_callback/plan_callback above:
        # unset means the explorer's round budget still stops silently, same
        # as before this feature existed.
        self.explorer_budget_callback = explorer_budget_callback
        self.files = HashlineTools()
        # ONE shared anchor engine: TurnRunner's post-write advice reads
        # `registry.files` to append fresh line ids + the syntax/LSP check when
        # an execute-stage bash command writes a file (same as NORMAL mode),
        # and it must share read-records/fingerprints with the editor
        # sub-agent, which edits through this same instance.
        self.registry.files = self.files
        self.runner = TurnRunner(client, self.registry, config, ui, [])
        # The chat agent keeps the one persistent conversation.
        self.chat_messages: list[dict] = [chat_agent.system_message()]
        # (goal, outcome) per completed turn — the "earlier in this session"
        # note handed to later turns' stages.
        self.turn_log: list[tuple[str, str]] = []
        # Debug transcript: every stage's final context, concatenated with
        # markers (what debug/headless.py dumps as messages.json).
        self.messages: list[dict] = []
        self.session_store = session_store
        self._turn: TurnDir | None = None
        # The mode box currently displayed; sub-agents save/restore around it.
        self._display_mode: Mode = Mode.NORMAL
        # The user-selected mode of the RUNNING turn (never a display mode).
        # The editor spawn reads it to decide whether the plan state is live:
        # PlanState is only reset at the start of a PLAN turn, so its approved
        # todos are stale context in any other mode.
        self._active_user_mode: Mode = Mode.NORMAL
        # Numbering for explore-N.md run artifacts, across the whole session.
        self._explore_count = 0
        # THE editor conversation: every `edit` call in a turn — whatever file
        # it targets — continues one append-only conversation, mirroring the
        # planner's plan_messages and the executor's exec_messages, so the
        # scratch-session checkpoint stack prefills only the new directive.
        # Reset per user turn like the other agent conversations.
        self._editor_messages: list[dict] = []
        self._editor_diag_memory: DiagnosticsMemory | None = None

    # -- app-facing surface (mirrors Agent) ---------------------------------

    def set_approval(self, approval: ApprovalCallback) -> None:
        self.registry.set_approval(approval)

    def clear(self) -> None:
        self.chat_messages = [chat_agent.system_message()]
        self.turn_log = []
        self.messages = []
        self.runner.reset_conversation_guards()
        self.registry.plan.state = PlanState()
        self._editor_messages = []
        self._editor_diag_memory = None

    def restore(self, turns) -> None:
        """Rehydrate a persisted session (the /sessions command): replay each
        turn's (goal, outcome) into the turn log so later stages see the prior
        context, and re-extend the CHAT conversation verbatim so CHAT mode
        resumes exactly. The server re-primes its KV cache on the first
        request via a normal full prefill — nothing else to hydrate."""
        self.clear()
        for turn in turns:
            self.turn_log.append((turn.goal, turn.outcome))
            self.chat_messages.extend(turn.chat_messages)

    async def run_turn(self, user_text: str, mode: Mode = Mode.NORMAL) -> bool:
        self._editor_messages = []
        self._editor_diag_memory = None
        self.ui.set_state(mascot.State.WAKEUP)
        self.ui.chat.add_user(user_text)
        if self.session_store is not None:
            self._turn = self.session_store.begin_turn(user_text)
        if mode not in USER_MODES:
            # Display modes (VISUAL/INSERT) are never dispatch targets.
            mode = Mode.NORMAL
        self._active_user_mode = mode
        if self._turn is not None:
            self._turn.meta["mode"] = mode.value
            self._turn.flush_meta()
        chat_delta_start = len(self.chat_messages)

        # Data handed across stage boundaries within the turn.
        context: dict = {
            "goal": user_text,
            "plan_block": "",
            "todos": [],
            "outcome": "",
        }
        if mode is Mode.CHAT:
            ok = await self._run_chat(user_text, context)
        elif mode is Mode.WEB:
            ok = await self._run_web(context)
        elif mode is Mode.PLAN:
            ok = await self._run_plan_and_execute(context)
        else:
            ok = await self._run_direct(context)

        self.turn_log.append((user_text, context["outcome"]))
        if self._turn is not None:
            self._turn.meta["ok"] = ok
            self._turn.meta["outcome"] = context["outcome"]
            if mode is Mode.CHAT:
                # The verbatim delta this turn appended to the persistent CHAT
                # conversation — what a restore re-extends to resume exactly.
                self._turn.meta["chat_messages"] = self.chat_messages[chat_delta_start:]
            self._turn.flush_meta()
            self.session_store.end_turn(self._turn, ok)
            self._turn = None
        if ok:
            self.ui.set_state(mascot.State.OK)
        return ok

    # -- mode dispatchers ------------------------------------------------------

    async def _run_chat(self, user_text: str, context: dict) -> bool:
        # The one persistent conversation: chat turns accumulate so
        # follow-ups read naturally. No tools — WEB mode owns the web.
        self.chat_messages.append({"role": "user", "content": user_text})
        self._begin_stage(chat_agent.SPEC, self.chat_messages)
        ok = await self._run_loop_checked()
        context["outcome"] = self._last_assistant_text(self.chat_messages)
        self._log_stage("chat", [])  # persistent list; marker only
        return ok

    async def _run_web(self, context: dict) -> bool:
        messages = [
            web_agent.system_message(),
            web_agent.build_task_message(context["goal"], self._session_notes()),
        ]
        self._begin_stage(web_agent.SPEC, messages)
        ok = await self._run_loop_checked()
        context["outcome"] = self._last_assistant_text(messages)
        self._save_text("web_findings.md", context["outcome"])
        self._log_stage("web", messages)
        return ok

    async def _run_plan_and_execute(self, context: dict) -> bool:
        """The planner ⇄ executor ping-pong.

        The planner's context is PERSISTENT on its own GPU session: after
        every executor task the machine returns to it with one short review
        message (executor report + qwen's todo completion doctrine), the
        planner answers with one `plan` progress mark and one guidance
        line, and the next executor task is dispatched. Each ping rides the
        plan-session checkpoint — prefill is just the new dialogue message.
        Segregation holds: executors keep {run_shell_command, edit}; review
        pings may EXECUTE only `plan` (the wire tools array stays the
        planner stage's, unchanged, so the planner prefix never breaks).

        The executors, in turn, CONTINUE one persistent conversation on the
        main session: the first todo opens it with goal + plan, each later
        todo appends a slim continuation directive — full context
        inheritance between consecutive executors, with checkpoint prefix
        reuse to match.
        """
        plan_tools = self.registry.plan
        # Fresh plan state every turn: stale todos from a previous turn must
        # never feed the executor schedule.
        plan_tools.state = PlanState()

        plan_messages = [
            planner_agent.system_message(),
            planner_agent.build_task_message(context["goal"], self._session_notes()),
        ]
        self._begin_stage(
            planner_agent.SPEC,
            plan_messages,
            max_rounds=planner_agent.plan_rounds(self.config.reasoning_effort),
            # A planner narrating "let me implement" (or spinning on denied
            # rewrites) after real work is DONE planning: end the stage
            # instead of nudging (observed live: the nudge cycle burned the
            # whole budget and starved the executors).
            stop_on_stall=True,
            # The `plan` call IS the gate: an approve decision ends planning
            # the moment it lands and actuates the plan.
            stop_when=lambda: (
                plan_tools.last_decision is not None
                and plan_tools.last_decision.kind == "approve"
            ),
        )
        ok = await self._run_loop_checked()

        decision = plan_tools.last_decision
        if not ok or decision is None or decision.kind != "approve":
            self._log_stage("planner", plan_messages)
            context["outcome"] = "Planning ended without an approved plan; execution was not started."
            return False
        if getattr(decision, "source", "user") == "auto":
            # The machine decided because no reviewer was bound: say so in
            # the transcript, not just the status bar.
            add_system = getattr(self.ui.chat, "add_system", None)
            if callable(add_system):
                add_system("plan auto-approved — no reviewer bound")

        # Nothing has executed yet: reset the cursor so EVERY todo gets an
        # executor task (belt-and-braces — the first approve commits progress 0).
        store = plan_tools.state
        store.progress = 0
        context["todos"] = list(store.todos)
        # active=True points the [>] cursor at the todo the first executor
        # message will carry, mirroring the directive it rides with.
        rendered = (
            render_todos(store.todos, 0, active=True)
            if store.todos
            else "(no todos recorded)"
        )
        context["plan_block"] = planner_agent.handoff(
            plan_tools.last_plan or "(no plan text)", rendered
        )
        self._save_text("plan.md", context["plan_block"])
        self._save_json("todos.json", {"todos": store.todos, "progress": store.progress})

        if not store.todos:
            # An approved plan with no todos degenerates to one direct pass.
            return await self._run_direct(context)

        # The approval handoff sits in the planner's persistent context,
        # framing the review pings that follow.
        plan_messages.append(
            {"role": "user", "content": planner_agent.APPROVAL_HANDOFF}
        )

        # ONE persistent executor conversation for the whole plan: each todo
        # appends its directive to the same list (full context inheritance),
        # so the main-session checkpoint stack prefills only the new message —
        # and ONE diagnostics memory to match (rows shown for todo 2 are still
        # in context during todo 5, so they must not repeat).
        exec_messages: list[dict] = [freestyle_agent.system_message()]
        exec_diag_memory = DiagnosticsMemory()
        spawn_cap = 2 * len(store.todos) + 4
        spawns = 0
        guidance = ""
        plan_update = ""
        completed_results: list[tuple[str, str]] = []
        while spawns < spawn_cap:
            executed_index = store.next_index()
            if executed_index is None:
                break
            content = store.todos[executed_index]
            ref = todo_ref(executed_index + 1, content)

            spawns += 1
            total = len(store.todos)
            position = executed_index + 1
            if len(exec_messages) == 1:
                exec_messages.append(
                    freestyle_agent.build_task_message(
                        context["goal"],
                        context["plan_block"],
                        position,
                        total,
                        content,
                        guidance,
                    )
                )
            else:
                exec_messages.append(
                    freestyle_agent.build_continuation_message(
                        position, total, content, guidance, plan_update
                    )
                )
            plan_update = ""
            self._begin_stage(
                freestyle_agent.SPEC, exec_messages, diag_memory=exec_diag_memory
            )
            exec_ok = await self._run_loop_checked()
            summary = self._last_assistant_text(exec_messages)
            completed_results.append((content, summary))
            self._save_text(
                f"task-{spawns}.md",
                freestyle_agent.task_result(spawns, total, content, summary),
            )

            # The brief dialogue: ONE bounded planner exchange on its own
            # session. A lone review failure never aborts the turn.
            version_before = store.version
            todos_before = list(store.todos)
            plan_messages.append(
                planner_agent.build_review_message(
                    position,
                    content,
                    ref,
                    summary,
                    exec_ok,
                    render_todos(store.todos, store.progress, active=True),
                )
            )
            review_mark = len(plan_messages)
            self._begin_stage(
                planner_agent.SPEC,
                plan_messages,
                max_rounds=planner_agent.REVIEW_MAX_ROUNDS,
                allowed=planner_agent.REVIEW_TOOLS,
                stop_on_stall=True,
            )
            review_ok = await self._run_loop_checked()
            if not exec_ok and not review_ok:
                # Executor AND its review both failed: nothing is deciding
                # anymore (server down, context overflow, ...). Without this
                # abort the version-unchanged fallback would march through
                # every remaining todo, "completing" them all with garbage.
                self._log_stage("execute", exec_messages)
                self._log_stage("planner", plan_messages)
                done_lines = [
                    freestyle_agent.task_result(i + 1, len(completed_results), task, s)
                    for i, (task, s) in enumerate(completed_results[:-1])
                ]
                done_lines.append(
                    f"Stopped at task {ref} ({content}): the "
                    "executor and the planner review both failed."
                )
                context["outcome"] = "\n".join(done_lines)
                self._save_json(
                    "todos.json", {"todos": store.todos, "progress": store.progress}
                )
                return False
            guidance = planner_agent.trim_guidance(
                self._last_assistant_text(plan_messages[review_mark:])
            )
            if guidance:
                add_system = getattr(self.ui.chat, "add_system", None)
                if callable(add_system):
                    add_system(f"planner: {guidance}")

            if store.todos != todos_before:
                # An approved replan spliced the remaining todos: recompute
                # the handoff, carry it to the next executor directive, and
                # extend the spawn budget for the new tail — under a hard
                # ceiling so replan loops cannot spin forever.
                context["todos"] = list(store.todos)
                context["plan_block"] = planner_agent.handoff(
                    plan_tools.last_plan or "(no plan text)",
                    render_todos(store.todos, store.progress, active=True),
                )
                plan_update = context["plan_block"]
                self._save_text("plan.md", context["plan_block"])
                spawn_cap = min(
                    spawns + 2 * len(store.todos[store.progress:]) + 4,
                    spawn_cap + 2 * len(store.todos),
                )
                add_system = getattr(self.ui.chat, "add_system", None)
                if callable(add_system):
                    add_system("plan revised and re-approved")
            elif store.progress >= len(store.todos):
                # The planner explicitly marked EVERYTHING done: that is its
                # "goal reached" signal — honor it, the loop ends here and
                # no executor runs for the remaining todos (a single task
                # often accomplishes the whole plan).
                pass
            elif store.progress > executed_index + 1:
                # A PARTIAL jump-ahead skips todos while later ones remain —
                # that reads as a stale/hallucinated mark, not a verdict:
                # completion is earned by execution, so the planner may
                # complete only up through the todo just executed
                # (decreases — reopens — are always allowed).
                store.progress = executed_index + 1

            # Machine reads the STATE, never the prose: no `plan` call at
            # all means the planner delegated the verdict to the executor's
            # outcome — and the machine always makes progress (no retry
            # spiral on an unreviewed failure; it is recorded instead). An
            # explicit write — even one re-asserting the same cursor — is
            # respected, hence the version counter, not a value diff.
            if store.version == version_before:
                store.progress = min(executed_index + 1, len(store.todos))

        pending_left = store.todos[store.progress:]
        total_done = len(completed_results)
        outcome_lines = [
            freestyle_agent.task_result(i + 1, total_done, task, summary)
            for i, (task, summary) in enumerate(completed_results)
        ]
        if pending_left:
            unfinished = ", ".join(pending_left)
            outcome_lines.append(
                f"Stopped at the task budget with unfinished todos: {unfinished}"
            )
        context["outcome"] = "\n".join(outcome_lines) or "Goal reached with no tasks executed."
        self._save_json("todos.json", {"todos": store.todos, "progress": store.progress})
        # The persistent executor conversation is logged ONCE (it grows per
        # todo; re-logging each round would duplicate it quadratically).
        self._log_stage("execute", exec_messages)
        self._log_stage("planner", plan_messages)
        return True

    async def _run_direct(self, context: dict) -> bool:
        messages = [
            freestyle_agent.system_message(),
            freestyle_agent.build_direct_message(context["goal"], self._session_notes()),
        ]
        self._begin_stage(freestyle_agent.SPEC, messages)
        ok = await self._run_loop_checked()
        context["outcome"] = self._last_assistant_text(messages)
        self._save_text("result.md", context["outcome"])
        self._log_stage("execute", messages)
        return ok

    # -- the explorer sub-agent -------------------------------------------------

    async def _run_explorer(self, arguments: dict) -> str:
        """The planner's `explore` call: spawn a stateless explorer, return
        its `resume` summary as the tool result.

        Zero context each spawn (its own system prompt + the planner's task,
        nothing else) on the scratch GPU session, so the planner's
        plan-session lineage and the executors' main-session checkpoints
        stay untouched. The mode box shows VISUAL for the sub-run.
        """
        task = ""
        if isinstance(arguments, dict):
            task = str(arguments.get("task", "")).strip()
        if not task:
            return "Error: `explore` needs a non-empty `task` describing what to find."

        # A fresh explorer holds no earlier diagnostics in its context, so the
        # shared engines get a fresh seen-store for the spawn's duration; the
        # calling stage's store is restored on the way out.
        prev_diag_memory = self.registry.diag_memory
        self.registry.set_diag_memory(DiagnosticsMemory())
        messages = [
            explorer_agent.system_message(),
            explorer_agent.build_task_message(task),
        ]
        registry = ExplorerRegistry(self.registry)
        runner = TurnRunner(self.client, registry, self.config, self.ui, messages)
        runner.allowed_tools = explorer_agent.SPEC.allowed_tools
        runner.write_feedback = explorer_agent.SPEC.write_feedback
        runner.request_overrides = {"qw35_session": SESSION_SCRATCH}
        runner.event_sink = self._transcript_event
        runner.max_rounds = explorer_agent.effort_rounds(self.config.reasoning_effort)
        # The `resume` call ends the run the moment its summary lands.
        runner.stop_when = lambda: registry.resume_summary is not None

        effort_tier = (self.config.reasoning_effort or "medium").lower()
        if effort_tier not in explorer_agent.EXPLORE_EFFORT_BUDGET_ROUNDS:
            effort_tier = "medium"
        user_stopped = False

        async def on_budget_reached() -> BudgetDecision:
            nonlocal effort_tier, user_stopped
            if self.explorer_budget_callback is None:
                return BudgetDecision(kind="stop")
            context = explorer_agent.ExplorerBudgetContext(
                task=task,
                effort=effort_tier,
                max_rounds=runner.max_rounds,
                next_tier=explorer_agent.next_tier(effort_tier),
                notes_preview=self._last_assistant_text(messages).strip(),
            )
            decision = await self.explorer_budget_callback(context)
            if decision.kind == "grow" and decision.max_rounds and context.next_tier:
                effort_tier = context.next_tier[0]
            elif decision.kind == "stop":
                user_stopped = True
            return decision

        runner.on_round_budget_reached = on_budget_reached
        prev = self._display_mode
        self._notify_mode(Mode.VISUAL)
        try:
            await runner.run_loop()
        except Qw35Error as exc:
            return f"Error: the explorer could not run ({exc.short_code()}: {exc.message})."
        finally:
            self.registry.set_diag_memory(prev_diag_memory)
            self._record_timings(runner)
            self._log_stage("explorer", messages)
            self._notify_mode(prev)

        summary = (registry.resume_summary or "").strip()
        if summary:
            report = f"Exploration findings:\n{summary}"
        else:
            # The explorer never called `resume` (round budget spent, or a
            # stream error ended the run): fall back to its last notes so
            # the planner still gets SOMETHING, clearly labeled.
            notes = self._last_assistant_text(messages).strip()
            if notes:
                report = (
                    "The explorer finished without calling `resume`; "
                    f"its last notes:\n{notes}"
                )
            else:
                report = (
                    "The explorer returned no findings (its budget ran out "
                    "before a `resume` call). Refine the task and call "
                    "`explore` again, or plan with what you already know."
                )
            if user_stopped:
                report += "\n\n(The user chose to stop this exploration early.)"
        self._explore_count += 1
        self._save_text(f"explore-{self._explore_count}.md", f"Task:\n{task}\n\n{report}")
        return report

    # -- the editor sub-agent -------------------------------------------------

    async def _run_editor(self, arguments: dict) -> str:
        filename = str(arguments.get("filename", "")).strip()
        line_ranges = str(arguments.get("line_ranges", "")).strip()
        instructions = str(arguments.get("instructions", "")).strip()

        # The editor's own read_file calls must satisfy the absolute-path
        # schema, so its whole view is anchored on the absolute path (the
        # executor-facing report keeps the delegator's verbatim filename).
        abs_path = os.path.abspath(filename)

        # The spawned editor continues ONE persistent conversation for the
        # whole turn, like the planner and the executor: the first `edit`
        # opens it with the system prompt, every later `edit` — whatever file
        # it targets — appends a slim continuation directive, so the
        # scratch-session checkpoint stack prefills only the new message.
        # Diagnostics memory is shared across spawns for the same reason:
        # rows shown to an earlier editor are still in context, so they must
        # not repeat.
        prev_diag_memory = self.registry.diag_memory
        is_first_spawn = not self._editor_messages
        if is_first_spawn:
            editor_diag_memory = DiagnosticsMemory()
            editor_messages = [
                {"role": "system", "content": editor_agent.EDITOR_SYSTEM_PROMPT},
            ]
        else:
            editor_messages = self._editor_messages
            editor_diag_memory = self._editor_diag_memory
        self.registry.set_diag_memory(editor_diag_memory)
        try:
            opened = await asyncio.to_thread(
                self.files.execute, "read_file", {"file_path": abs_path, "_force": True}
            )
            if opened.startswith(TOOL_ATTENTION_MARKER):
                opened = opened[len(TOOL_ATTENTION_MARKER):]
            if opened.lstrip().startswith("Error"):
                return opened

            # Snapshot for the session-wide diff in the final report.
            start_content = self._read_file_text(filename)

            # read_file output: header line, annotated body, then the
            # trailing diagnostics section (canonical grammar — carved with the
            # shared splitter, not blank-line guesswork). Slice only the body.
            header, _, rest = opened.partition("\n")
            body, tail = split_trailing_section(rest)
            total_lines = len(body.splitlines())
            spans = editor_agent.parse_line_ranges(line_ranges, total_lines)
            annotated = editor_agent.slice_annotated_body(body, spans)
            if tail.strip():
                annotated = f"{annotated}\n\n{tail.strip()}"

            # Background for the editor: the plan excerpt (PLAN turns only —
            # PlanState is stale in any other mode), the delegating agent's
            # recent activity with every tool call obfuscated to markdown, and
            # the hidden reasoning of the turn that issued this `edit` call.
            # Reads the executor's history ONLY — exec_messages/main-session
            # prefixes are never touched; the block lands solely in the
            # editor's fresh scratch context.
            state = self.registry.plan.state
            in_plan = (
                self._active_user_mode is Mode.PLAN
                and state.approved
                and bool(state.todos)
            )
            current_todo = None
            if in_plan and 0 <= state.progress < len(state.todos):
                current_todo = (
                    f"({state.progress + 1}/{len(state.todos)}): "
                    f"{state.todos[state.progress]}"
                )
            background = spawn_context.build_editor_background(
                plan_markdown=state.last_plan if in_plan else None,
                current_todo=current_todo,
                history=list(self.runner.messages),
                reasoning=self.runner.last_reasoning,
            )

            if is_first_spawn:
                editor_messages.append(
                    editor_agent.build_editor_user_message(
                        abs_path, instructions, annotated, background
                    )
                )
            else:
                editor_messages.append(
                    editor_agent.build_editor_continuation_message(
                        abs_path, instructions, annotated, background
                    )
                )
            # Commit the conversation only once a spawn is real (an unreadable
            # file above returns without leaving a task-less opener behind);
            # appends after this point land in the shared list directly.
            self._editor_messages = editor_messages
            self._editor_diag_memory = editor_diag_memory
            editor_registry = EditorRegistry(self.files, self.registry.lsp)
            runner = TurnRunner(self.client, editor_registry, self.config, self.ui, editor_messages)
            runner.allowed_tools = editor_agent.SPEC.allowed_tools
            runner.write_feedback = editor_agent.SPEC.write_feedback
            runner.request_overrides = {"qw35_session": SESSION_SCRATCH}
            runner.event_sink = self._transcript_event
            prev = self._display_mode
            self._notify_mode(Mode.INSERT)
            try:
                ok = await runner.run_loop()
            except Qw35Error as exc:
                return f"Error: the editor could not run ({exc.short_code()}: {exc.message})."
            finally:
                # Log the full conversation only on the first spawn; later
                # spawns would duplicate the growing prefix quadratically.
                if is_first_spawn:
                    self._log_stage("editor", editor_messages)
                self._notify_mode(prev)
        finally:
            self.registry.set_diag_memory(prev_diag_memory)

        summary = self._last_assistant_text(editor_messages)
        if not editor_registry.results:
            report = (
                f"Editor made no changes to {filename}."
                + (f"\n{summary.strip()}" if summary.strip() else "")
            )
            return report
        # The FILE is the ground truth, not the editor's self-assessment: an
        # editor once reported "I've rewritten the logic" while every
        # mutation had left the file byte-identical, costing the executor a
        # verify-and-rebuild cycle to find out.
        end_content = self._read_file_text(filename)
        if start_content is not None and end_content == start_content:
            return TOOL_ATTENTION_MARKER + (
                f"Editor made no effective changes to {filename}: the file is "
                "byte-identical to when the editor session started. The "
                "requested change has NOT been applied — restate the "
                "instructions more concretely (exact lines and replacement "
                "content) or handle it another way."
            )

        # One diff for the WHOLE editor session (start -> end), not the last
        # mutation's: the visualized result must show everything the editor
        # did to the file, however many edit/insert/delete steps it took.
        if start_content is not None and end_content is not None:
            diff_lines = list(
                difflib.unified_diff(
                    start_content.splitlines(),
                    end_content.splitlines(),
                    fromfile=f"a/{filename}",
                    tofile=f"b/{filename}",
                    lineterm="",
                )
            )
            changes = f"Diff:\n" + "\n".join(diff_lines)
        else:  # fallback: the per-mutation reports, anchors scrubbed
            changes = "\n\n".join(
                _ANCHOR_BLOCK.sub("\n", result).strip()
                for _name, result in editor_registry.results
            )
        if len(changes) > EDITOR_REPORT_MAX_CHARS:
            changes = changes[:EDITOR_REPORT_MAX_CHARS] + "\n... (editor report truncated)"
        edits = len(editor_registry.results)
        report = (
            f"Editor result for {filename} ({edits} edit"
            f"{'s' if edits != 1 else ''} applied):\n{summary.strip()}\n\n{changes}"
        )
        if not ok:
            report = f"{report}\n\n(The editor stopped before confirming completion.)"
        if editor_registry.saw_attention:
            # A mutation tripped the syntax flag at some point. Re-validate the
            # FINAL content: if errors remain, tell the executor WHAT is wrong
            # — the bare error flag used to travel without the syntax block,
            # reading as a mystery failure — and keep the flag; if the editor
            # ended up fixing the file, don't cry wolf over the interim break.
            validation = None
            if end_content is not None:
                validation = await asyncio.to_thread(validate_file, filename, end_content)
            if validation is not None and validation.errors:
                # This section lands in the EXECUTOR's context, so it dedups
                # against the executor's memory (restored above) — an executor
                # that already saw these rows gets the honest count, not the
                # full list again.
                sifted = (
                    self.registry.diag_memory.sift(filename, validation, end_content)
                    if self.registry.diag_memory is not None
                    else None
                )
                section = validation_report_with_memory(validation, sifted)
                return TOOL_ATTENTION_MARKER + f"{report}\n\n{section}"
            if validation is not None and validation.checked:
                return report
            # Could not re-verify (unreadable file / unknown language): keep
            # the conservative flag.
            return TOOL_ATTENTION_MARKER + report
        return report

    @staticmethod
    def _read_file_text(filename: str) -> str | None:
        try:
            return Path(filename).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    # -- plumbing --------------------------------------------------------------

    def _begin_stage(
        self,
        spec: AgentSpec,
        messages: list[dict],
        max_rounds: int | None = None,
        allowed: frozenset[str] | None = None,
        stop_on_stall: bool = False,
        stop_when=None,
        diag_memory: DiagnosticsMemory | None = None,
    ) -> None:
        """Point the shared TurnRunner at a stage's context: its message
        list, wire toolset, allowlist, session, and round budget. `allowed`
        overrides the EXECUTION allowlist only (the wire array stays the
        stage's — a planner review ping must not change the plan session's
        prefix). `stop_on_stall` ends the stage on the first no-progress
        turn after real work — the natural handoff for stages whose output
        is read from state (the todo store), not from further prose.
        `diag_memory` is the stage context's diagnostics seen-store: pass the
        SAME store on every re-aim of one persistent context (the executor
        conversation), omit it for a fresh context (fresh store, sees
        everything once)."""
        allowed_tools = allowed if allowed is not None else spec.allowed_tools
        self.registry.set_stage(spec.name, allowed_tools)
        self.registry.set_diag_memory(
            diag_memory if diag_memory is not None else DiagnosticsMemory()
        )
        self.runner.messages = messages
        self.runner.allowed_tools = allowed_tools
        self.runner.write_feedback = spec.write_feedback
        self.runner.max_rounds = max_rounds
        self.runner.stop_on_stall = stop_on_stall
        self.runner.stop_when = stop_when
        self.runner.request_overrides = {"qw35_session": spec.session}
        self.runner.event_sink = self._transcript_event
        self.runner.reset_turn_guards()
        self._notify_mode(STAGE_MODES.get(spec.name, Mode.NORMAL))

    async def _run_loop_checked(self) -> bool:
        ok = await self.runner.run_loop()
        self._record_timings(self.runner)
        return ok

    def _record_timings(self, runner: TurnRunner) -> None:
        if self._turn is not None:
            self._turn.record_timings(runner.last_timings)

    def _transcript_event(self, kind: str, fields: dict) -> None:
        if self._turn is not None:
            self._turn.record(kind, **fields)

    @staticmethod
    def _last_assistant_text(messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant" and not message.get("tool_calls"):
                content = message.get("content") or ""
                if content.strip():
                    return content
        return ""

    def _session_notes(self) -> str:
        """A compact record of recent turns, for follow-up requests."""
        if not self.turn_log:
            return ""
        lines = []
        for goal, outcome in self.turn_log[-SESSION_NOTE_TURNS:]:
            goal_line = " ".join(goal.split())[:200]
            outcome_line = " ".join((outcome or "(no recorded outcome)").split())[:300]
            lines.append(f"- {goal_line} -> {outcome_line}")
        return "\n".join(lines)

    def _log_stage(self, name: str, messages: list[dict]) -> None:
        """Append a stage's final context to the debug transcript."""
        self.messages.append({"role": "system", "content": f"=== stage: {name} ==="})
        self.messages.extend(messages)

    def _notify_mode(self, mode: Mode) -> None:
        self._display_mode = mode
        set_mode = getattr(self.ui, "set_mode", None)
        if callable(set_mode):
            set_mode(mode)

    def _save_text(self, name: str, text: str) -> None:
        if self._turn is not None:
            self._turn.save(name, text)

    def _save_json(self, name: str, payload: dict) -> None:
        if self._turn is not None:
            self._turn.save_json(name, payload)
