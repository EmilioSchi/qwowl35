"""The agent loop: stream → tool calls → execute → loop, driving mascot state.

This is the essential plumbing ported from little-coder, minus all its setup.
It owns the running conversation and reports state through a small UI interface
(``set_state``, ``set_error``, and a ``chat`` log) that the Textual app provides.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal, Protocol

# How long to hold the WAKEUP frame so it is actually seen before prefill starts.
WAKEUP_HOLD_SECONDS = 0.8
COMPACT_ARG_CHARS = 140
COMPACT_RESULT_CHARS = 220
MALFORMED_TOOL_CALL_MAX_RETRIES = 3
MALFORMED_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call\b[\s\S]*?<\s*(?:function(?:\b|=)|bash|run_shell_command|beginTransaction|read_file|replace|edit|insert|delete)\b",
    re.IGNORECASE,
)
MALFORMED_TOOL_CALL_FEEDBACK = (
    "Your previous response contained a malformed <tool_call> block, so no tool ran. "
    "Rewrite the tool call using valid nested XML and send only the corrected tool call. "
    "Use exactly one <function=tool_name> element inside <tool_call>. "
    "Put arguments in child <parameter=name>value</parameter> elements; do not use JSON or XML attributes for arguments. "
    "Escape XML text as &amp;, &lt;, and &gt; when those characters are literal content; raw quotes are okay in parameter text. "
    "For creating or replacing a whole file, prefer a bash heredoc command; for editing an existing non-empty file, use your file-editing tools."
)

# When a turn ends with no tool call but the text reads as if the model was still
# mid-task (it described a next step and then stopped), nudge it to continue
# rather than ending the turn. Bounded by this cap so a model that keeps trailing
# off without acting still terminates (no infinite loop).
CONTINUATION_MAX_NUDGES = 2
# Tool-agnostic on purpose: this text is injected into EVERY stage's loop,
# and naming specific tools once pushed a planner (which has none of them)
# into a repeat-denial spiral.
CONTINUATION_FEEDBACK = (
    "You ended your turn without calling a tool, but your message reads as if you "
    "were still working — you described a next step and then stopped. If you are not "
    "finished, continue now by calling the next tool. If "
    "the task really is complete, reply with a brief explicit confirmation."
)

# Loop guard: when the model emits the same tool call (identical name AND
# arguments) immediately after one we just ran, we refuse to run it again and
# feed this note back as the tool result. A repeated identical call can only
# yield the result already in context, so re-running it wastes a round-trip and
# is the classic shape of a tool loop. A call that changes its arguments (or a
# different tool) clears the guard and runs normally.
#
# Rather than the same line (or the same line with an incrementing counter) over
# and over — which is itself repetitive — we keep a pool of differently-worded
# notes and pick one at random for each repeat, never reusing the one used for
# the immediately preceding repeat. Each just makes the same neutral point a
# different way: the call matches a previous one, the earlier result still
# stands, and there is no need to keep repeating it. Tone is neutral (no blame,
# no frustration) and none of them prescribe a specific next action — what to do
# next is the model's choice.
REPEATED_TOOL_NOTES = (
    "This `{name}` call is identical to the previous one, so it wasn't re-run — the result above still stands.",
    "Same `{name}` call as a moment ago; nothing has changed, so it would just return the result already shown above.",
    "This `{name}` request repeats the last one, so it was skipped — you already have its result above.",
    "That `{name}` call matches one just made; the result above already covers it, so no need to send it again.",
    "`{name}` was already called with these exact arguments — the output above is what it returns, so you can carry on.",
    "This is a repeat of an earlier `{name}` call; re-running it gives the same result shown above.",
    "The same `{name}` call came through again — its result is unchanged from the one above, so feel free to continue.",
    "No need to resend this `{name}` call; it is identical to a prior one and the result above already answers it.",
)


def repeated_tool_message(
    name: str,
    exclude: str | None = None,
    rng: random.Random | None = None,
) -> str:
    """Pick a neutral, direction-free note for an identical repeated call.

    A random note from :data:`REPEATED_TOOL_NOTES` is chosen, avoiding
    ``exclude`` (the note used for the previous repeat) so two consecutive
    denials never read the same. ``rng`` lets callers/tests pin the choice.
    """
    notes = [note.format(name=name) for note in REPEATED_TOOL_NOTES]
    if exclude is not None and len(notes) > 1:
        notes = [note for note in notes if note != exclude]
    chooser = rng or random
    return chooser.choice(notes)


# The shell tool's wire name (qwen-code's trained `run_shell_command`) plus
# the legacy "bash" alias still used by older transcripts and tests.
SHELL_TOOL_NAMES = frozenset({"bash", "run_shell_command"})

# The anchored mutation tools. When the model emits several of these together in
# one turn on the SAME file, the runner coalesces the contiguous run into one
# batch (one write, one diff, one syntax/LSP pass) via ``registry.execute_batch``
# instead of validating each separately. See ``_edit_batch_slice`` / ``_run_edit_batch``.
_EDIT_TOOL_NAMES = frozenset({"replace", "insert", "delete"})

# Registry strings for a bash command that was gated and never executed. No file
# was written, so a wholesale-write target in such a command must not be recorded
# (see :meth:`TurnRunner._rewrite_advice_for`).
_BASH_NOT_RUN_PREFIXES = ("Command denied", "Command not run")


def stage_violation_message(name: str, allowed: frozenset[str]) -> str:
    """Tool-result text for a call outside the active stage's toolset."""
    tools = ", ".join(sorted(allowed))
    return (
        f"The tool `{name}` is not available in this stage. "
        f"Tools available right now: {tools}. Continue with one of those."
    )


def _bash_syntax_warning(arguments: object) -> str:
    """Tree-sitter syntax-check block for a bash call's command, or ``""``."""
    try:
        command = arguments.get("command") if isinstance(arguments, dict) else None
        if not isinstance(command, str) or not command:
            return ""
        return format_warning_block("bash", check_bash(command))
    except Exception:  # noqa: BLE001 - syntax warnings are best-effort
        return ""


# When a bash command wholesale-rewrites a file (a truncating `>` redirect, e.g.
# `cat > f <<EOF`) that this conversation already wrote once, the command still
# runs but we append a nudge: prefer the anchored edit tools over replacing the
# whole file every time. As with REPEATED_TOOL_NOTES, the wording varies (and
# never repeats the immediately preceding one) so the same point made again does
# not read as a stuck loop. Each note names the file and points at read +
# the edit family; none forbids the rewrite outright.
REWRITE_ADVICE_NOTES = (
    "Note: `{file}` was already written earlier in this session. To change it, use the `edit` tool instead of rewriting the whole file.",
    "Heads up: that replaced all of `{file}` again. For a targeted change, prefer the `edit` tool over a full `cat >` rewrite.",
    "`{file}` already exists from a previous write — changing it with the `edit` tool is cheaper than re-emitting the entire file each time.",
    "Reminder: you already created `{file}`. Land later changes with the `edit` tool rather than overwriting it wholesale.",
    "This rewrote `{file}` from scratch again. When a file already exists, describe the fix to the `edit` tool so unrelated lines stay untouched.",
    "Tip: for an existing file like `{file}`, the `edit` tool changes just what you need — no need to re-send the whole file through bash.",
)


def rewrite_advice_message(
    file: str,
    exclude: str | None = None,
    rng: random.Random | None = None,
    pool: tuple = REWRITE_ADVICE_NOTES,
) -> str:
    """Pick a differently-worded nudge for a repeated wholesale file rewrite.

    Mirrors :func:`repeated_tool_message`: a random note from ``pool``,
    avoiding ``exclude`` (the note used for the previous nudge) so two
    consecutive nudges never read the same. ``rng`` lets callers/tests pin
    the choice.
    """
    notes = [note.format(file=file) for note in pool]
    if exclude is not None and len(notes) > 1:
        notes = [note for note in notes if note != exclude]
    chooser = rng or random
    return chooser.choice(notes)


def escalated_rewrite_message(file: str, count: int) -> str:
    """A firm, fixed directive once the soft nudge has been ignored.

    The soft :data:`REWRITE_ADVICE_NOTES` rotate and read as gentle tips; once a
    file has been wholesale-rewritten this many times the model is in a rewrite
    loop, so we stop rotating and stop hedging: one consistent imperative that
    names the exact tools. Returned with ``is_error=True`` so it reads as a
    correction, not another optional suggestion.
    """
    return (
        f"STOP writing `{file}` through bash redirects (`>` or `>>`). You have now "
        f"written the whole file this way {count} times and it is still wrong — "
        f"re-emitting or appending keeps layering on mistakes. Do NOT write it "
        f"through bash again. Use the `edit` tool with precise instructions to "
        f"change ONLY the specific lines that are wrong."
    )


# Redirect targets under these roots are scratch/output (e.g. `prog > /tmp/out`),
# not authored source files, so a repeated write to them must not earn a rewrite
# nudge. Keyed off the normalized (posixpath) target.
_SCRATCH_TARGET_PREFIXES = ("/tmp/", "/private/tmp/", "/private/var/", "/var/")


def _is_scratch_target(target: str) -> bool:
    return any(target.startswith(prefix) for prefix in _SCRATCH_TARGET_PREFIXES)


# Cap on how many freshly-written files we auto-read anchors for in a single bash
# result, so a command writing many files cannot flood the model.
_AUTO_READ_MAX_FILES = 3


def _authored_write_targets(command: str) -> list[str]:
    """De-duplicated, non-scratch files a command writes (truncating or append).

    Shared by the rewrite-recognition nudge and the post-write anchor auto-read so
    both agree on exactly which files count as authored source. Order-preserving.
    """
    targets: list[str] = []
    seen: set[str] = set()
    for target in truncating_write_targets(command) + append_write_targets(command):
        if target in seen or _is_scratch_target(target):
            continue
        seen.add(target)
        targets.append(target)
    return targets


def build_auto_read_block(files_tool: object, command: str) -> str:
    """Anchor read-outs for the files a successful bash command just authored.

    Returns a block (or ``""``) to append to a bash result so the model already
    has the line anchors and can edit the file in place with the anchored edit
    tools — no separate ``read`` round-trip. This complements the
    rewrite-recognition policy (which discourages re-writing the whole file):
    instead of only nudging, we hand the model exactly what it needs to edit.
    Empty/missing files and unreadable targets are silently skipped. When any
    written file fails its syntax/LSP check, the returned block is prefixed with
    TOOL_ATTENTION_MARKER so the caller flags the bash result is_error, exactly
    like the edit tools do.
    """
    if not isinstance(command, str) or not command:
        return ""
    targets = _authored_write_targets(command)
    if not targets:
        return ""
    # Skip files whose current anchors the model already holds — but a file that
    # changed since it was read (this write, an external editor, another process)
    # is NOT current, so it still earns fresh anchors.
    has_current = getattr(files_tool, "has_current_anchors", None)
    if callable(has_current):
        targets = [t for t in targets if not has_current(t)]
    if not targets:
        return ""
    blocks: list[str] = []
    attention = False
    for target in targets[:_AUTO_READ_MAX_FILES]:
        try:
            # A just-created file is often the first of its language this session;
            # give its server a bounded moment to boot so the syntax check below
            # is the real LSP one, not a silent tree-sitter fallback. Runs on a
            # worker thread (see _auto_read_written_files).
            warm_lsp(target)
            # _force bypasses the redundant-read gate: the auto-read's whole job is
            # to surface fresh line ids, so it must always get the file body.
            output = files_tool.execute("read_file", {"file_path": target, "_force": True})  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - best-effort convenience read
            continue
        marked = output.startswith(TOOL_ATTENTION_MARKER)
        if marked:
            output = output[len(TOOL_ATTENTION_MARKER):]
        if not output or not output.strip() or output.lstrip().startswith("Error"):
            continue
        attention = attention or marked
        blocks.append(
            f"You just wrote `{target}`. Its current line ids are below — edit it "
            "in place with replace; no separate read_file needed:\n"
            f"{output}"
        )
    if not blocks:
        return ""
    remaining = len(targets) - _AUTO_READ_MAX_FILES
    if remaining > 0:
        blocks.append(f"(+{remaining} more written file(s); open them with read_file when needed.)")
    joined = "\n\n".join(blocks)
    # A written file with syntax errors must escalate the bash result the same
    # way an edit would: re-mark the joined block so the dispatch loop can
    # strip the marker and flag is_error.
    return TOOL_ATTENTION_MARKER + joined if attention else joined


# Shown-issue cap for the plain post-write report, mirroring the hashline
# anchors cap so a pathological file cannot flood the model.
_MAX_SHOWN_POST_WRITE_ISSUES = 5


def build_post_write_report(command: str, edit_hint: bool = True, memory=None) -> str:
    """Validation report for the files a successful bash command just authored,
    for agents that do NOT speak the hashline dialect.

    The plain counterpart of :func:`build_auto_read_block`: no anchor ids, no
    ``read_file``, no hashline session state — the freestyle executor's
    `edit` sub-agent tool takes plain line numbers. A clean file gets a
    one-line confirmation (its content is already verbatim in the model's
    context from the write itself); a file with errors gets the issue list with
    plain ``line N:`` content rows, and the whole block is prefixed with
    TOOL_ATTENTION_MARKER so the dispatch loop flags the bash result is_error
    exactly like the hashline path. Rows the calling agent instance was already
    shown (``memory``, its per-agent DiagnosticsMemory) are summarised instead
    of repeated; headline counts always reflect the file's CURRENT state, so a
    still-broken file keeps its attention flag. ``edit_hint=False`` drops the
    `edit`-tool clause for agents without that tool. Returns ``""`` when there
    is nothing to report. May block a bounded moment warming a language server —
    call from a worker thread, never the event loop.
    """
    if not isinstance(command, str) or not command:
        return ""
    targets = _authored_write_targets(command)
    if not targets:
        return ""
    blocks: list[str] = []
    attention = False
    for target in targets[:_AUTO_READ_MAX_FILES]:
        try:
            text = Path(target).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue
        try:
            warm_lsp(target)
            v = validate_file(target, text)
        except Exception:  # noqa: BLE001 - validation is best-effort
            continue
        if v is None:
            continue
        report = v.report()
        if not report:
            continue  # unknown language / nothing checkable
        file_lines = text.splitlines()
        sifted = memory.sift(target, v, text) if memory is not None else None
        if not v.errors:
            if sifted is not None and clean_validation_report is not None:
                report = clean_validation_report(v, sifted, _MAX_SHOWN_POST_WRITE_ISSUES)
            blocks.append(f"Wrote `{target}` ({len(file_lines)} lines). {report}")
            continue
        attention = True
        if sifted is not None and sifted.all_prior:
            blocks.append(
                f"You just wrote `{target}` — it still has problems. Syntax check "
                f"({v.label}) — {len(v.errors)} issue(s), {ALL_UNCHANGED}."
            )
            continue
        new_errors = sifted.errors if sifted is not None else list(v.errors)
        new_warnings = sifted.warnings if sifted is not None else list(v.warnings)
        prior_errors = sifted.prior_errors if sifted is not None else 0
        prior_warnings = sifted.prior_warnings if sifted is not None else 0
        if edit_hint:
            header = (
                f"You just wrote `{target}` — it has problems. Syntax check "
                f"({v.label}) — {len(v.errors)} issue(s). Fix ONLY these lines "
                "with the `edit` tool (filename, line ranges, instructions); "
                "do NOT rewrite the file through bash:"
            )
        else:
            header = (
                f"You just wrote `{target}` — it has problems. Syntax check "
                f"({v.label}) — {len(v.errors)} issue(s):"
            )
        lines = [header]
        shown = new_errors[:_MAX_SHOWN_POST_WRITE_ISSUES]
        for line_no, _col, message in shown:
            lines.append(f"- {message}")
            if 1 <= line_no <= len(file_lines):
                lines.append(f"  line {line_no}: {file_lines[line_no - 1]}")
        extra = len(new_errors) - len(shown)
        if extra > 0:
            lines.append(f"- … and {extra} more")
        if prior_errors:
            lines.append(unchanged_note(prior_errors))
        warn_shown = new_warnings[:_MAX_SHOWN_POST_WRITE_ISSUES]
        if warn_shown or prior_warnings:
            lines.append(f"Warnings (not blocking) — {len(v.warnings)}:")
            lines.extend(f"- {message}" for _line, _col, message in warn_shown)
            warn_extra = len(new_warnings) - len(warn_shown)
            if warn_extra > 0:
                lines.append(f"- … and {warn_extra} more")
            if prior_warnings:
                lines.append(unchanged_note(prior_warnings, "warning"))
        if sifted is not None:
            sifted.mark_rendered(shown, warn_shown)
        blocks.append("\n".join(lines))
    if not blocks:
        return ""
    remaining = len(targets) - _AUTO_READ_MAX_FILES
    if remaining > 0:
        blocks.append(f"(+{remaining} more written file(s) not shown.)")
    joined = "\n\n".join(blocks)
    return TOOL_ATTENTION_MARKER + joined if attention else joined


import mascot
from client import (
    AssistantTurn,
    ContentDelta,
    Finish,
    PrefillProgress,
    Qw35Error,
    ReasoningDelta,
    StreamAccumulator,
    ToolCallArgsDelta,
    ToolCallBegin,
    ToolCallDemoted,
    ToolCallFinal,
    ToolCallName,
    Usage,
)
from config import Config
from prompts import build_system_message
from tools.bash import append_write_targets, truncating_write_targets
from tools.files.adapter import TOOL_ATTENTION_MARKER
from tools_registry import ToolRegistry

try:  # syntax warnings are best-effort; never let them break the agent loop.
    from tools.syntax.checker import check_bash, format_warning_block
except Exception:  # pragma: no cover - defensive fallback

    def check_bash(command):  # type: ignore[misc]
        return []

    def format_warning_block(label, msgs):  # type: ignore[misc]
        return ""


try:  # post-write validation is best-effort too.
    from tools.syntax import validate_file, warm_lsp
except Exception:  # pragma: no cover - defensive fallback

    def validate_file(path, source):  # type: ignore[misc]
        return None

    def warm_lsp(path):  # type: ignore[misc]
        return False


try:  # diagnostics presentation (per-agent dedup wording); same guard.
    from tools.diagnostics import ALL_UNCHANGED, clean_validation_report, unchanged_note
except Exception:  # pragma: no cover - defensive fallback
    clean_validation_report = None  # type: ignore[assignment]
    ALL_UNCHANGED = "all unchanged and already reported above"

    def unchanged_note(count: int, noun: str = "issue") -> str:  # type: ignore[misc]
        return f"- {count} unchanged {noun}(s) already reported above (not repeated)"


class AgentUI(Protocol):
    chat: object  # ChatLog, with add_* helpers

    def set_state(self, state: mascot.State) -> None: ...
    def begin_generation(self) -> None: ...
    def set_prefill(
        self,
        percent: float,
        processed: int | None = None,
        total: int | None = None,
        session_ctx: int | None = None,
    ) -> None: ...
    def add_reasoning_delta(self, text: str) -> None: ...
    def set_usage(self, usage: dict, timings: dict | None = None) -> None: ...
    def set_error(self, code: str, message: str) -> None: ...
    def pop_queued_user_batch(self) -> str | None: ...


@dataclass(frozen=True)
class BudgetDecision:
    """What to do when a stage's round budget runs out (see
    ``TurnRunner.on_round_budget_reached``). ``"stop"`` reproduces today's
    silent cutoff; ``"grow"`` raises ``max_rounds`` (to ``max_rounds``, the
    field) and keeps looping; ``"force"`` seeds a named ``tool_choice`` (the
    ``forced_tool`` field) on the next stream so the model is guaranteed to
    call that tool instead of trailing off.
    """

    kind: Literal["stop", "grow", "force"]
    max_rounds: int | None = None
    forced_tool: str | None = None


# How many times a forced tool_choice (BudgetDecision(kind="force")) is
# re-armed after a malformed/failed attempt before giving up and falling
# back to the stop path — bounds the worst case instead of looping forever
# on a model that can't produce valid XML even when forced.
FORCE_TOOL_CHOICE_MAX_ATTEMPTS = 3


class TurnRunner:
    """The reusable streaming → tool-dispatch → guard loop.

    Owns one caller-provided message list plus the client/registry/config/ui
    plumbing, and drives the model until it ends its turn. Every loop guard
    (malformed-XML retry, continuation nudge, repeated-bash denial,
    wholesale-rewrite escalation) lives here so every driver — the freestyle
    :class:`Agent` today, the smart-mode orchestrator's stages next — gets
    the same discipline without re-implementing the loop.
    """

    # Stage discipline (smart mode): when set, a tool call outside this set is
    # denied with an error tool-result instead of executing, keeping a
    # restricted stage honest without changing the wire toolset. ``None``
    # (the default, and the freestyle behavior) allows every registered tool.
    # A class-level default so instances built via ``__new__`` inherit it.
    allowed_tools: frozenset[str] | None = None
    # Post-bash-write feedback dialect (see AgentSpec.write_feedback):
    # "hashline" = anchor ids via the registry's hashline engine (the NORMAL
    # agent, which holds read_file/replace/insert/delete); "subedit" =
    # plain validation report naming the `edit` delegator tool; "report" =
    # validation report only. Class default keeps NORMAL mode unchanged; the
    # orchestrator overrides it per stage from the AgentSpec.
    write_feedback: str = "hashline"
    # Extra request fields merged over config.gen_params() for every stream —
    # the orchestrator uses this for `qw35_session` (per-stage GPU sessions).
    # None (default) sends the plain params.
    request_overrides: dict | None = None
    # Session-transcript observer: called with (kind, fields) whenever a
    # message is appended to history (assistant turns with reasoning, tool
    # results, guard feedback, queued user input). Failures are swallowed —
    # the observer can never break the loop. None (default) records nothing.
    event_sink: Callable[[str, dict], None] | None = None
    # qw35_timings of the most recent stream (session_path, cached tokens,
    # checkpoint depth...); the orchestrator's prefix-discipline tripwire
    # reads it. None until a stream carrying usage completes.
    last_timings: dict | None = None
    # Hidden reasoning of the most recent assistant turn in THIS loop.
    # Reasoning never enters the message list (only content + tool_calls are
    # persisted), so this is the one place the orchestrator can read the
    # thinking that led to the tool call it is currently executing — captured
    # for the editor spawn's background block. Assigned right after the turn
    # is appended and before its tools dispatch, so mid-dispatch readers see
    # the ISSUING turn's reasoning. A class default so ``__new__``-built test
    # instances inherit it.
    last_reasoning: str = ""
    # Round budget: when set, run_loop stops cleanly (returns True) after
    # this many model streams — a machine transition mirroring the explore
    # budget, so no stage can thrash unboundedly. None (default) = unbounded
    # (classic freestyle behavior).
    max_rounds: int | None = None
    # Called when the round budget is spent, before the loop actually stops —
    # gives the driver a chance to intervene (grow the budget, force a
    # specific tool call) instead of a silent cutoff. None (default) = the
    # classic behavior above, unchanged. See BudgetDecision.
    on_round_budget_reached: Callable[[], Awaitable[BudgetDecision]] | None = None
    # One-shot tool_choice for the next stream only (see BudgetDecision.force
    # handling in run_loop / _stream_assistant). None = no forcing.
    _pending_tool_choice: dict | None = None
    # Remaining re-arms of a forced tool_choice after a failed/malformed
    # attempt, without re-asking the driver. 0 = not currently forcing.
    _force_attempts_remaining: int = 0
    # State-predicate stop: when set, checked after each tool-call turn —
    # a True return ends the loop cleanly. The driver declares "this state
    # means the stage is done" (e.g. planning ends the moment a todo goes
    # in_progress: the model marked work as STARTING, so stop planning).
    stop_when = None
    # Stall exit: when True, once ANY tool has executed in this loop, a
    # no-progress turn (content only, or nothing but denials) ends the loop
    # cleanly instead of nudging. This is the natural stage handoff — a
    # planner that wrote its todos and starts narrating "let me implement"
    # is DONE planning; nudging it once produced an endless
    # narrate → repeat-denied → narrate cycle that starved the executors.
    stop_on_stall: bool = False
    # Per-tool-name last-executed signatures (see reset_turn_guards). A
    # class default so instances built via ``__new__`` inherit it; the dict
    # is created by reset_turn_guards()/__init__.
    _last_signatures: dict | None = None
    # Signature of the last call that EXECUTED (any tool) — the consecutive
    # guard for shell/web_fetch: re-running the same command after other
    # work (edit -> re-run tests) is legitimate and must never be denied.
    _last_executed_signature: str | None = None
    # Rewrite advice speaks only of `edit`: both modes run the same
    # executor toolset (run_shell_command + edit) now.
    rewrite_advice_notes: tuple = REWRITE_ADVICE_NOTES

    def __init__(
        self,
        client,
        registry: ToolRegistry,
        config: Config,
        ui: AgentUI,
        messages: list[dict] | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.config = config
        self.ui = ui
        self.messages: list[dict] = messages if messages is not None else []
        # Last-executed signature PER TOOL NAME, used to deny a repeated
        # identical `plan` call. Per-tool (not a single slot): a planner once
        # looped identical todo rewrites interleaved with
        # ask_user_question, and the interleaving cleared a single-slot
        # guard every time. Reset per user turn.
        self._last_signatures: dict[str, str] = {}
        self._last_executed_signature = None
        # The denial note used for the previous consecutive repeat, so the next
        # one picks a different wording. Reset whenever a distinct call runs.
        self._last_repeat_msg: str | None = None
        # Per-file count of wholesale bash writes (truncating `>` redirect) this
        # conversation. 1st write: silent. 2nd: soft, rotating nudge toward the
        # anchored edit tools. 3rd+: a firm, fixed escalation (is_error) because a
        # repeated full rewrite of the same file is a loop. Persists across turns.
        self._bash_rewrite_counts: dict[str, int] = {}
        # The soft nudge used last, so consecutive soft nudges differ in wording.
        self._last_rewrite_advice: str | None = None

    def reset_turn_guards(self) -> None:
        """Start a fresh user turn: it may legitimately repeat the last call."""
        self._last_signatures = {}
        self._last_executed_signature = None
        self._last_repeat_msg = None
        self.last_reasoning = ""

    def reset_conversation_guards(self) -> None:
        """Forget all per-conversation guard state (a cleared conversation)."""
        self.reset_turn_guards()
        self._bash_rewrite_counts.clear()
        self._last_rewrite_advice = None
        # The cleared context no longer contains any previously-shown
        # diagnostics, so the active per-agent memory must forget them too.
        memory = getattr(self.registry, "diag_memory", None)
        if memory is not None:
            memory.clear()

    @staticmethod
    def _tool_signature(name: str, arguments: object) -> str:
        """Stable identity for a tool call: name plus canonical arguments."""
        return name + "\x00" + json.dumps(arguments, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _has_malformed_tool_call_text(content: str) -> bool:
        return bool(content and MALFORMED_TOOL_CALL_PATTERN.search(content))

    @staticmethod
    def _looks_unfinished(content: str) -> bool:
        """Heuristic: the model trailed off mid-task without calling a tool.

        Conservative — fires only on a clear continuation signal AND no completion
        signal, so a genuine wrap-up still ends the turn. The call site caps how
        often this can fire, so a model that keeps trailing off still terminates.
        """
        if not content or not content.strip():
            return False
        text = content.strip()
        low = text.lower()
        done_markers = (
            "done", "complete", "finished", "all set", "works correctly",
            "is correct", "let me know", "would you like", "to summarize",
            "in summary", "passes", "success",
        )
        if any(marker in low[-200:] for marker in done_markers):
            return False
        if text.endswith(":"):
            return True
        # A dangling enumerator like a trailing "3." with nothing after it.
        if re.search(r"(?:^|\n)\s*\d+[.)]\s*$", text):
            return True
        last_line = low.splitlines()[-1].strip()
        intent_markers = (
            "let me ", "let's ", "i'll ", "i will ", "now i", "next, i",
            "next i ", "i need to ", "i should ", "let me fix", "let me test",
            "let me check", "let me try", "let me update",
        )
        return any(marker in last_line for marker in intent_markers)

    async def run_loop(self) -> bool:
        """Stream and dispatch until the model ends its turn. True on success.

        The caller has already appended whatever user/directive message opens
        the turn (and reset the per-turn guards if that message starts a fresh
        user turn).
        """
        malformed_tool_retries = 0
        continuation_nudges = 0
        rounds = 0
        executed_any = False  # any tool actually ran in this loop
        if self._last_signatures is None:  # instances built via __new__
            self._last_signatures = {}

        while True:
            try:
                turn = await self._stream_assistant()
            except Qw35Error as exc:
                self.ui.chat.flush_reasoning()
                self.ui.chat.flush_assistant()
                self.ui.set_error(exc.short_code(), exc.message)
                return False
            rounds += 1

            self.messages.append(self._assistant_message(turn))
            self.last_reasoning = turn.reasoning or ""
            self._emit_event(
                "assistant",
                content=turn.content or "",
                tool_calls=[
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in turn.tool_calls
                ],
            )

            if not turn.tool_calls:
                if self._has_malformed_tool_call_text(turn.content):
                    malformed_tool_retries += 1
                    self.ui.set_warning("retrying tool call")
                    if malformed_tool_retries > MALFORMED_TOOL_CALL_MAX_RETRIES:
                        self.ui.set_error(
                            "TOOL",
                            "The model kept emitting malformed tool_call XML.",
                        )
                        return False
                    self.messages.append({"role": "user", "content": MALFORMED_TOOL_CALL_FEEDBACK})
                    continue
                if self._inject_queued_user_batch():
                    malformed_tool_retries = 0
                    continue
                if self.stop_on_stall and executed_any:
                    # Work happened and the model is done tooling: that IS
                    # the stage handoff — no continuation nudge.
                    self.ui.set_state(mascot.State.OK)
                    return True
                if (
                    self._looks_unfinished(turn.content)
                    and continuation_nudges < CONTINUATION_MAX_NUDGES
                ):
                    continuation_nudges += 1
                    self.ui.set_warning("nudging to continue")
                    self.messages.append(
                        {"role": "user", "content": CONTINUATION_FEEDBACK}
                    )
                    continue
                self.ui.set_state(mascot.State.OK)
                return True

            malformed_tool_retries = 0
            executed_this_turn = False
            calls = turn.tool_calls
            batch_fn = getattr(self.registry, "execute_batch", None)
            i = 0
            while i < len(calls):
                # Coalesce a contiguous run of edit calls (replace/insert/delete)
                # on the SAME file into ONE batch: one write, one diff, one
                # syntax/LSP pass, instead of validating each separately. A lone
                # edit (group of one) or any non-edit call stays on the exact
                # single-call path (`_run_single_call`), so its behavior — and
                # every guard — is unchanged. Registries without execute_batch
                # (e.g. the planner) also fall through to the single-call path.
                group = self._edit_batch_slice(calls, i)
                if len(group) >= 2 and batch_fn is not None:
                    if await self._run_edit_batch(group, batch_fn):
                        executed_this_turn = True
                        executed_any = True
                else:
                    for call in group:
                        if await self._run_single_call(call):
                            executed_this_turn = True
                            executed_any = True
                i += len(group)
            # Tools ran this turn — real progress, so refresh the continuation budget.
            continuation_nudges = 0
            self._inject_queued_user_batch()
            # The driver's state predicate says the stage is done (e.g. the
            # plan was just approved through the gate).
            if self.stop_when is not None and self.stop_when():
                self.ui.set_state(mascot.State.OK)
                return True
            # A turn of nothing but denials after real work is a stall: the
            # stage has what it needs — transition instead of spinning
            # (observed live: narrate → repeat-denied plan rewrite → narrate,
            # forever, while the executors starved).
            if self.stop_on_stall and executed_any and not executed_this_turn:
                self.ui.set_state(mascot.State.OK)
                return True
            # Round budget spent: stop cleanly — a machine transition, not an
            # error (the caller decides what the stage produced) — unless the
            # driver wants a say first (grow the budget, force a specific
            # tool call) via on_round_budget_reached.
            if self.max_rounds is not None and rounds >= self.max_rounds:
                if self._force_attempts_remaining > 0:
                    # Re-arm the same forced tool_choice without re-asking
                    # the driver — the previous forced attempt didn't land
                    # (malformed XML, wrong tool, etc.).
                    self._force_attempts_remaining -= 1
                    continue
                if self.on_round_budget_reached is not None:
                    decision = await self.on_round_budget_reached()
                    if decision.kind == "grow" and decision.max_rounds:
                        self.max_rounds = decision.max_rounds
                        continue
                    if decision.kind == "force" and decision.forced_tool:
                        self._pending_tool_choice = {
                            "type": "function",
                            "function": {"name": decision.forced_tool},
                        }
                        self._force_attempts_remaining = FORCE_TOOL_CHOICE_MAX_ATTEMPTS - 1
                        continue
                self.ui.set_info("stage round budget reached")
                self.ui.set_state(mascot.State.OK)
                return True
            # Loop again so the model sees the tool results.

    async def _run_single_call(self, call) -> bool:
        """Execute one tool call with every guard. Returns True iff a tool
        actually ran (False for a stage-denial or a deduped repeat).

        This is the per-call loop body, extracted verbatim so a lone edit and
        any non-edit call take the exact same path they always did.
        """
        self.ui.set_state(mascot.state_for_tool(call.name))
        # Stage discipline: a call outside the active stage's toolset is
        # denied with an error result — never executed. Inert (None) in
        # freestyle mode.
        if self.allowed_tools is not None and call.name not in self.allowed_tools:
            result = stage_violation_message(call.name, self.allowed_tools)
            self.ui.chat.add_tool_result(call.index, call.name, result, is_error=True)
            self._append_tool_message(call, result, is_error=True)
            return False
        # The call box was already streamed in during _stream_assistant.
        signature = self._tool_signature(call.name, call.arguments)
        # Dedup semantics differ by tool. `plan` and `explore`:
        # per-tool and turn-persistent (a planner once laundered
        # identical lists through interleaved ask_user_question
        # calls; an identical `explore` re-spawn burns a whole
        # sub-agent run for a report it already has). Shell and
        # web_fetch: STRICTLY CONSECUTIVE — an identical command is
        # denied only when nothing else executed since, because a
        # re-run after other work (edit -> re-run the tests) is
        # meaningful. Edit tools are never deduped: a content-free
        # "already did that" note would misreport the file state.
        if call.name in ("plan", "explore"):
            is_repeat = signature == self._last_signatures.get(call.name)
        elif call.name in (*SHELL_TOOL_NAMES, "web_fetch"):
            is_repeat = signature == self._last_executed_signature
        else:
            is_repeat = False
        if is_repeat and not self._tool_arguments_invalid(call.arguments):
            # Repeated identical call: deny without executing and feed
            # back a randomly-worded note (different from the previous
            # one). Leave the guard set so a third identical call is
            # denied too; a changed call clears it below.
            result = repeated_tool_message(call.name, exclude=self._last_repeat_msg)
            self._last_repeat_msg = result
            self.ui.chat.add_tool_result(call.index, call.name, result, is_error=True)
            self._append_tool_message(call, result, is_error=True)
            return False
        is_error = False
        try:
            result = await self.registry.execute(call.name, call.arguments)
        except Exception as exc:  # noqa: BLE001 - feed errors back to the model
            result = f"Tool error: {exc}"
            is_error = True
        # A file tool flags a successful-but-needs-attention result (syntax
        # errors present) with an in-band marker. Strip it and surface the
        # result as an error so the model prioritises the fix.
        if result.startswith(TOOL_ATTENTION_MARKER):
            result = result[len(TOOL_ATTENTION_MARKER):]
            is_error = True
        if (
            call.name in SHELL_TOOL_NAMES
            and not is_error
            and not result.startswith(_BASH_NOT_RUN_PREFIXES)
        ):
            advice, escalate = self._rewrite_advice_for(call.arguments)
            if advice:
                result = f"{result}\n\n{advice}"
                # A repeated full rewrite is a loop; flag it as an error so
                # it reads as a correction, not another optional tip.
                if escalate:
                    is_error = True
            # Deterministic low-level syntax check of the command itself.
            # Informational (not an error): bash also reports its own
            # syntax errors at runtime; this adds a clean, anchored note.
            syntax = _bash_syntax_warning(call.arguments)
            if syntax:
                result = f"{result}\n\n{syntax}"
            # If the command authored a file, hand the model its fresh line
            # anchors so it can edit in place without a separate read. A
            # marked block means a written file failed its syntax/LSP check;
            # surface it as an error exactly like an edit would be.
            anchors = await self._auto_read_written_files(call.arguments)
            if anchors:
                if anchors.startswith(TOOL_ATTENTION_MARKER):
                    anchors = anchors[len(TOOL_ATTENTION_MARKER):]
                    is_error = True
                result = f"{result}\n\n{anchors}"
        self.ui.chat.add_tool_result(call.index, call.name, result, is_error=is_error)
        self._append_tool_message(call, result, is_error=is_error)
        # A distinct call ran: it becomes ITS TOOL'S new guard target
        # (`plan`) and the overall last-executed one (shell/
        # web_fetch), and the repeat streak restarts.
        self._last_signatures[call.name] = signature
        self._last_executed_signature = signature
        self._last_repeat_msg = None
        return True

    def _edit_group_key(self, call) -> str | None:
        """The batch key (absolute target path) of a groupable edit call, else
        None. None keeps the call on the single-call path: non-edit tools, an
        edit denied by stage discipline, invalid-JSON args, or an edit with no
        explicit ``file`` (one relying on the engine's remembered last file)."""
        if call.name not in _EDIT_TOOL_NAMES:
            return None
        if self.allowed_tools is not None and call.name not in self.allowed_tools:
            return None
        args = call.arguments
        if not isinstance(args, dict) or args.get("_invalid_json") is True:
            return None
        target = args.get("file")
        if not isinstance(target, str) or not target:
            return None
        return os.path.abspath(target)

    def _edit_batch_slice(self, calls: list, i: int) -> list:
        """The maximal contiguous run of edit calls from ``i`` sharing one file
        key (length >= 1). A lone or ungroupable call returns just ``[calls[i]]``."""
        key = self._edit_group_key(calls[i])
        if key is None:
            return [calls[i]]
        group = [calls[i]]
        j = i + 1
        while j < len(calls) and self._edit_group_key(calls[j]) == key:
            group.append(calls[j])
            j += 1
        return group

    async def _run_edit_batch(self, group: list, batch_fn) -> bool:
        """Dispatch a group of same-file edit calls as ONE batch and append one
        tool result per call.id. Returns True (a batch always attempts to run).

        On a result-count mismatch or an unexpected raise, emits one error per
        call and does NOT re-dispatch — re-running would double-apply the writes
        the batch already made.
        """
        for call in group:
            self.ui.set_state(mascot.state_for_tool(call.name))
        fallback = None
        try:
            results = await batch_fn([(c.name, c.arguments) for c in group])
        except Exception as exc:  # noqa: BLE001 - execute_batch is designed not to raise
            results = None
            fallback = f"Tool error: {exc}"
        if not isinstance(results, list) or len(results) != len(group):
            msg = fallback or "Tool error: batch result count mismatch."
            for call in group:
                self.ui.chat.add_tool_result(call.index, call.name, msg, is_error=True)
                self._append_tool_message(call, msg, is_error=True)
            return True
        for call, result in zip(group, results):
            is_error = False
            if isinstance(result, str) and result.startswith(TOOL_ATTENTION_MARKER):
                result = result[len(TOOL_ATTENTION_MARKER):]
                is_error = True
            self.ui.chat.add_tool_result(call.index, call.name, result, is_error=is_error)
            self._append_tool_message(call, result, is_error=is_error)
            # Mirror the single-call guard-signature updates (cosmetic for edit
            # tools, which are never deduped, but keeps the state identical).
            signature = self._tool_signature(call.name, call.arguments)
            self._last_signatures[call.name] = signature
            self._last_executed_signature = signature
        self._last_repeat_msg = None
        return True

    def _emit_event(self, kind: str, **fields) -> None:
        if self.event_sink is None:
            return
        try:
            self.event_sink(kind, fields)
        except Exception:
            pass

    def _append_tool_message(
        self, call, result: str, *, is_error: bool
    ) -> None:
        self.messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": result}
        )
        self._emit_event(
            "tool_result",
            id=call.id,
            name=call.name,
            result=result,
            is_error=is_error,
        )

    def _append_user_message(self, user_text: str) -> None:
        self.ui.chat.add_user(user_text)
        self.messages.append({"role": "user", "content": user_text})
        self._emit_event("user", text=user_text)

    def _inject_queued_user_batch(self) -> bool:
        pop_batch = getattr(self.ui, "pop_queued_user_batch", None)
        if pop_batch is None:
            return False
        queued = pop_batch()
        if not queued:
            return False
        self._append_user_message(queued)
        # Treat queued input as a fresh user turn inside the active worker.
        self._last_tool_signature = None
        self._last_repeat_msg = None
        return True

    async def _stream_assistant(self) -> AssistantTurn:
        self.ui.begin_generation()
        self.ui.set_state(mascot.State.PREFILL)
        acc = StreamAccumulator()
        thinking_started = False
        generating = False

        params = dict(self.config.gen_params())
        if self.request_overrides:
            params.update(self.request_overrides)
        if self._pending_tool_choice is not None:
            # One-shot: forces exactly this stream, then falls back to the
            # model's own choice again (re-armed by run_loop if it fails).
            params["tool_choice"] = self._pending_tool_choice
            self._pending_tool_choice = None
        async for event in self.client.stream_chat(
            messages=self.messages,
            tools=self.registry.schemas(),
            **params,
        ):
            acc.add(event)
            if isinstance(event, PrefillProgress):
                self.ui.set_prefill(
                    event.percent, event.processed, event.total, event.session_ctx
                )
            elif isinstance(event, ReasoningDelta):
                if not thinking_started:
                    self.ui.set_state(mascot.State.THINKING)
                    thinking_started = True
                self.ui.add_reasoning_delta(event.text)
                self.ui.chat.add_reasoning_chunk(event.text)
            elif isinstance(event, ContentDelta):
                if not generating:
                    self.ui.chat.flush_reasoning()
                    self.ui.set_state(mascot.State.INFERENCE)
                    generating = True
                self.ui.chat.add_assistant_chunk(event.text)
            elif isinstance(event, ToolCallBegin):
                # Tool call starting — close any open text, open a growing box.
                # With raw streaming the name may still be empty (the box shows
                # the raw XML until ToolCallName arrives).
                self.ui.chat.flush_reasoning()
                self.ui.chat.flush_assistant()
                generating = False
                if event.name:
                    self.ui.set_state(mascot.state_for_tool(event.name))
                self.ui.chat.begin_tool_call(event.index, event.name)
            elif isinstance(event, ToolCallArgsDelta):
                self.ui.chat.update_tool_call(event.index, event.fragment)
            elif isinstance(event, ToolCallName):
                self.ui.set_state(mascot.state_for_tool(event.name))
                self.ui.chat.name_tool_call(event.index, event.name)
            elif isinstance(event, ToolCallFinal):
                self.ui.chat.finalize_tool_call(event.index, event.arguments)
            elif isinstance(event, ToolCallDemoted):
                # Not a tool call after all — drop the box; the raw text follows
                # as content/reasoning deltas (and the malformed-call retry
                # logic sees it in the accumulated turn).
                self.ui.chat.demote_tool_call(event.index)
            elif isinstance(event, Usage):
                self.last_timings = event.timings or {}
                self.ui.set_usage(event.usage, event.timings)
            elif isinstance(event, Finish):
                pass  # recorded by the accumulator

        self.ui.chat.flush_reasoning()
        self.ui.chat.flush_assistant()
        return acc.finalize()

    @staticmethod
    def _assistant_message(turn: AssistantTurn) -> dict:
        message: dict = {"role": "assistant", "content": turn.content or ""}
        if turn.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            TurnRunner._history_tool_arguments(call.arguments),
                            ensure_ascii=False,
                        ),
                    },
                }
                for call in turn.tool_calls
            ]
        return message

    @staticmethod
    def _history_tool_arguments(arguments: dict) -> dict:
        if isinstance(arguments, dict) and arguments.get("_invalid_json") is True:
            return {}
        return arguments

    @staticmethod
    def _tool_arguments_invalid(arguments: dict) -> bool:
        return isinstance(arguments, dict) and arguments.get("_invalid_json") is True

    def _rewrite_advice_for(self, arguments: dict) -> tuple[str | None, bool]:
        """Count wholesale bash writes; advise/escalate on a repeated rewrite.

        The first wholesale write (a truncating ``>`` redirect) of a file is just
        counted. The second returns a soft, rotating nudge toward the anchored edit
        tools. The third and beyond return a firm, fixed escalation, since a file
        rewritten three+ times is a rewrite loop. New targets in the same command
        are counted silently. Returns ``(message, escalate)`` — ``(None, False)``
        when nothing is repeated; ``escalate`` marks the firm directive so the
        caller can flag it as an error.
        """
        command = arguments.get("command") if isinstance(arguments, dict) else None
        if not isinstance(command, str) or not command:
            return None, False
        # Both wholesale rewrites (`>`) and incremental appends (`>>`) of the same
        # authored file count toward the loop — appending is just the slower escape
        # hatch. Scratch/output targets (e.g. `prog > /tmp/out`) are skipped.
        targets = _authored_write_targets(command)
        repeated: str | None = None
        repeated_count = 0
        for target in targets:
            count = self._bash_rewrite_counts.get(target, 0) + 1
            self._bash_rewrite_counts[target] = count
            if count >= 2 and repeated is None:
                repeated, repeated_count = target, count
        if repeated is None:
            return None, False
        if repeated_count >= 3:
            return escalated_rewrite_message(repeated, repeated_count), True
        advice = rewrite_advice_message(
            repeated, exclude=self._last_rewrite_advice, pool=self.rewrite_advice_notes
        )
        self._last_rewrite_advice = advice
        return advice, False

    async def _auto_read_written_files(self, arguments: dict) -> str:
        """Post-write feedback for files a bash command just wrote.

        Dialect per :attr:`write_feedback`: hashline anchors for agents that
        hold the hashline tools, a plain validation report otherwise. Lets the
        model fix a freshly-written file immediately, without a separate read.
        Best-effort: returns ``""`` on any problem.
        """
        command = arguments.get("command") if isinstance(arguments, dict) else None
        if not isinstance(command, str) or not command:
            return ""
        if not _authored_write_targets(command):
            return ""
        try:
            if self.write_feedback == "hashline":
                files_tool = getattr(self.registry, "files", None)
                if files_tool is None:
                    return ""
                return await asyncio.to_thread(build_auto_read_block, files_tool, command)
            # The plain report dedups against the CURRENT agent's diagnostics
            # memory (the registry points at it; hashline reads its own via
            # the files engine inside build_auto_read_block).
            return await asyncio.to_thread(
                build_post_write_report,
                command,
                self.write_feedback == "subedit",
                getattr(self.registry, "diag_memory", None),
            )
        except Exception:  # noqa: BLE001 - convenience only
            return ""


class Agent(TurnRunner):
    """The freestyle agent: one persistent conversation over the full toolset.

    A thin wrapper over :class:`TurnRunner` that owns the conversation
    semantics — the system message, ``/clear``, and the between-turn history
    compaction. Smart-mode stages drive :class:`TurnRunner` directly instead.
    """

    def __init__(self, client, registry: ToolRegistry, config: Config, ui: AgentUI) -> None:
        super().__init__(
            client,
            registry,
            config,
            ui,
            messages=[build_system_message(registry=registry)],
        )

    def clear(self) -> None:
        """Reset the conversation, keeping only the system message.

        Mirrors qw35-client's ``/clear``: preserve the system prompt, drop the
        rest, and reset the per-conversation guards so the next turn starts fresh.
        """
        system = next((m for m in self.messages if m.get("role") == "system"), None)
        self.messages = (
            [system] if system is not None else [build_system_message(registry=self.registry)]
        )
        self.reset_conversation_guards()

    async def run_turn(self, user_text: str) -> bool:
        """Run one user turn to completion. Returns True on success."""
        self._compact_completed_history()
        self.ui.set_state(mascot.State.WAKEUP)
        self._append_user_message(user_text)
        # A fresh user turn may legitimately repeat the previous turn's call.
        self.reset_turn_guards()
        await asyncio.sleep(WAKEUP_HOLD_SECONDS)  # let the wake animation play
        return await self.run_loop()

    def _compact_completed_history(self) -> None:
        """Replace old tool-call payloads with compact summaries.

        Completed tool calls can include whole-file heredocs or large edit bodies.
        Keeping those exact arguments in later turns burns context without adding
        new state; the filesystem already holds the applied change. We preserve
        the user/assistant text and a short record of each completed tool call.

        Freestyle-only: it rewrites history in place, which is exactly what the
        smart-mode pipeline must never do (its prompts must stay byte-identical
        prefixes for the server's checkpoint stack), so the orchestrator prunes
        at stage boundaries instead of compacting.
        """
        if len(self.messages) < 2:
            return

        compacted: list[dict] = []
        i = 0
        while i < len(self.messages):
            msg = self.messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                lines = ["Completed tool calls:"]
                for tc in msg.get("tool_calls") or []:
                    lines.append(f"- {self._compact_tool_call(tc)}")
                i += 1
                results: list[str] = []
                while i < len(self.messages) and self.messages[i].get("role") == "tool":
                    content = self.messages[i].get("content") or ""
                    results.append(self._shorten(str(content), COMPACT_RESULT_CHARS))
                    i += 1
                if results:
                    lines.append("Results: " + " | ".join(results))
                compacted.append({"role": "assistant", "content": "\n".join(lines)})
                continue
            if msg.get("role") == "tool":
                i += 1
                continue
            compacted.append(msg)
            i += 1
        self.messages = compacted

    @classmethod
    def _compact_tool_call(cls, tc: dict) -> str:
        fn = tc.get("function") or {}
        name = str(fn.get("name") or "tool")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}

        if name in SHELL_TOOL_NAMES:
            command = str(args.get("command") or "")
            first_line = command.splitlines()[0] if command else ""
            return f"bash: {cls._shorten(first_line, COMPACT_ARG_CHARS)!r}"

        parts = [name]
        file = args.get("file")
        if isinstance(file, str) and file:
            parts.append(f"on {file}")
        anchor = args.get("id")
        if isinstance(anchor, str) and anchor:
            parts.append(f"at {anchor}")
        content = args.get("content")
        if isinstance(content, str):
            line_count = 0 if content == "" else len(content.splitlines())
            parts.append(f"with {line_count} lines/{len(content)} chars")
        return " ".join(parts)

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        text = " ".join(text.splitlines()).strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."
