"""The agent loop: stream → tool calls → execute → loop, driving mascot state.

This is the essential plumbing ported from little-coder, minus all its setup.
It owns the running conversation and reports state through a small UI interface
(``set_state``, ``set_error``, and a ``chat`` log) that the Textual app provides.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Protocol

# How long to hold the WAKEUP frame so it is actually seen before prefill starts.
WAKEUP_HOLD_SECONDS = 0.8
COMPACT_ARG_CHARS = 140
COMPACT_RESULT_CHARS = 220
MALFORMED_TOOL_CALL_MAX_RETRIES = 3
MALFORMED_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call\b[\s\S]*?<\s*(?:function(?:\b|=)|bash|beginTransaction|edit|insert|delete)\b",
    re.IGNORECASE,
)
MALFORMED_TOOL_CALL_FEEDBACK = (
    "Your previous response contained a malformed <tool_call> block, so no tool ran. "
    "Rewrite the tool call using valid nested XML and send only the corrected tool call. "
    "Use exactly one <function=tool_name> element inside <tool_call>. "
    "Put arguments in child <parameter=name>value</parameter> elements; do not use JSON or XML attributes for arguments. "
    "Escape XML text as &amp;, &lt;, and &gt; when those characters are literal content; raw quotes are okay in parameter text. "
    "For creating or replacing a whole file, prefer a bash heredoc command; for editing an existing non-empty file, beginTransaction for line ids first and use edit/insert/delete."
)

# When a turn ends with no tool call but the text reads as if the model was still
# mid-task (it described a next step and then stopped), nudge it to continue
# rather than ending the turn. Bounded by this cap so a model that keeps trailing
# off without acting still terminates (no infinite loop).
CONTINUATION_MAX_NUDGES = 2
CONTINUATION_FEEDBACK = (
    "You ended your turn without calling a tool, but your message reads as if you "
    "were still working — you described a next step and then stopped. If you are not "
    "finished, continue now by calling the next tool (run the program to check its "
    "output, beginTransaction for line ids, or edit to fix a line). If "
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


# Registry strings for a bash command that was gated and never executed. No file
# was written, so a wholesale-write target in such a command must not be recorded
# (see :meth:`Agent._rewrite_advice_for`).
_BASH_NOT_RUN_PREFIXES = ("Command denied", "Command not run")


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
    "Note: `{file}` was already written earlier in this session. To change it, get its line ids with `beginTransaction` and apply `edit`/`insert`/`delete` instead of rewriting the whole file.",
    "Heads up: that replaced all of `{file}` again. For a targeted change, prefer `beginTransaction` plus the id-based edit tools over a full `cat >` rewrite.",
    "`{file}` already exists from a previous write — editing it in place with `beginTransaction` and `edit` is cheaper than re-emitting the entire file each time.",
    "Reminder: you already created `{file}`. Land later edits via `beginTransaction` for line ids, then `edit`/`insert`/`delete`, rather than overwriting it wholesale.",
    "This rewrote `{file}` from scratch again. When a file already exists, get its line ids with `beginTransaction` and use the line-edit tools so unrelated lines stay untouched.",
    "Tip: for an existing file like `{file}`, the id-based edit tools (`beginTransaction` + `edit`) change just what you need — no need to re-send the whole file through bash.",
)


def rewrite_advice_message(
    file: str,
    exclude: str | None = None,
    rng: random.Random | None = None,
) -> str:
    """Pick a differently-worded nudge for a repeated wholesale file rewrite.

    Mirrors :func:`repeated_tool_message`: a random note from
    :data:`REWRITE_ADVICE_NOTES`, avoiding ``exclude`` (the note used for the
    previous nudge) so two consecutive nudges never read the same. ``rng`` lets
    callers/tests pin the choice.
    """
    notes = [note.format(file=file) for note in REWRITE_ADVICE_NOTES]
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
        f"through bash again. Run `beginTransaction {file}` to get line ids, then "
        f"`edit` to change ONLY the specific lines that are wrong."
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
    Empty/missing files and unreadable targets are silently skipped.
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
    for target in targets[:_AUTO_READ_MAX_FILES]:
        try:
            # _force bypasses the redundant-read gate: the auto-read's whole job is
            # to surface fresh line ids, so it must always get the file body.
            output = files_tool.execute("beginTransaction", {"file": target, "_force": True})  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - best-effort convenience read
            continue
        if output.startswith(TOOL_ATTENTION_MARKER):
            output = output[len(TOOL_ATTENTION_MARKER):]
        if not output or not output.strip() or output.lstrip().startswith("Error"):
            continue
        blocks.append(
            f"You just wrote `{target}`. Its current line ids are below — edit it "
            "in place with edit; no separate beginTransaction needed:\n"
            f"{output}"
        )
    if not blocks:
        return ""
    remaining = len(targets) - _AUTO_READ_MAX_FILES
    if remaining > 0:
        blocks.append(f"(+{remaining} more written file(s); open them with beginTransaction when needed.)")
    return "\n\n".join(blocks)


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


class AgentUI(Protocol):
    chat: object  # ChatLog, with add_* helpers

    def set_state(self, state: mascot.State) -> None: ...
    def begin_generation(self) -> None: ...
    def set_prefill(
        self, percent: float, processed: int | None = None, total: int | None = None
    ) -> None: ...
    def add_reasoning_delta(self, text: str) -> None: ...
    def set_usage(self, usage: dict, timings: dict | None = None) -> None: ...
    def set_error(self, code: str, message: str) -> None: ...
    def pop_queued_user_batch(self) -> str | None: ...


class Agent:
    def __init__(self, client, registry: ToolRegistry, config: Config, ui: AgentUI) -> None:
        self.client = client
        self.registry = registry
        self.config = config
        self.ui = ui
        self.messages: list[dict] = [build_system_message(registry=self.registry)]
        # Signature of the last tool call we actually executed this turn, used
        # to deny an immediately repeated identical call. Reset per user turn.
        self._last_tool_signature: str | None = None
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

    def clear(self) -> None:
        """Reset the conversation, keeping only the system message.

        Mirrors qw35-client's ``/clear``: preserve the system prompt, drop the
        rest, and reset the per-conversation guards so the next turn starts fresh.
        """
        system = next((m for m in self.messages if m.get("role") == "system"), None)
        self.messages = (
            [system] if system is not None else [build_system_message(registry=self.registry)]
        )
        self._last_tool_signature = None
        self._last_repeat_msg = None
        self._bash_rewrite_counts.clear()
        self._last_rewrite_advice = None

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

    async def run_turn(self, user_text: str) -> bool:
        """Run one user turn to completion. Returns True on success."""
        self._compact_completed_history()
        self.ui.set_state(mascot.State.WAKEUP)
        self._append_user_message(user_text)
        # A fresh user turn may legitimately repeat the previous turn's call.
        self._last_tool_signature = None
        self._last_repeat_msg = None
        malformed_tool_retries = 0
        continuation_nudges = 0
        await asyncio.sleep(WAKEUP_HOLD_SECONDS)  # let the wake animation play

        while True:
            try:
                turn = await self._stream_assistant()
            except Qw35Error as exc:
                self.ui.chat.flush_reasoning()
                self.ui.chat.flush_assistant()
                self.ui.set_error(exc.short_code(), exc.message)
                return False

            self.messages.append(self._assistant_message(turn))

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
            for call in turn.tool_calls:
                state = mascot.State.BASH if call.name == "bash" else mascot.State.EDIT
                self.ui.set_state(state)
                # The call box was already streamed in during _stream_assistant.
                signature = self._tool_signature(call.name, call.arguments)
                # Dedup only bash (its original purpose). For edit tools a
                # content-free "already did that" note is misleading — the model
                # may think the edit landed when it did not — so always re-run them
                # and return the true current state.
                if (
                    call.name == "bash"
                    and signature == self._last_tool_signature
                    and not self._tool_arguments_invalid(call.arguments)
                ):
                    # Repeated identical call: deny without executing and feed
                    # back a randomly-worded note (different from the previous
                    # one). Leave the guard set so a third identical call is
                    # denied too; a changed call clears it below.
                    result = repeated_tool_message(call.name, exclude=self._last_repeat_msg)
                    self._last_repeat_msg = result
                    self.ui.chat.add_tool_result(call.index, call.name, result, is_error=True)
                    self.messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": result}
                    )
                    continue
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
                    call.name == "bash"
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
                    # anchors so it can edit in place without a separate read.
                    anchors = await self._auto_read_written_files(call.arguments)
                    if anchors:
                        result = f"{result}\n\n{anchors}"
                self.ui.chat.add_tool_result(call.index, call.name, result, is_error=is_error)
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
                # A distinct call ran: it becomes the new guard target and the
                # repeat streak restarts.
                self._last_tool_signature = signature
                self._last_repeat_msg = None
            # Tools ran this turn — real progress, so refresh the continuation budget.
            continuation_nudges = 0
            self._inject_queued_user_batch()
            # Loop again so the model sees the tool results.

    def _append_user_message(self, user_text: str) -> None:
        self.ui.chat.add_user(user_text)
        self.messages.append({"role": "user", "content": user_text})

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

    def _compact_completed_history(self) -> None:
        """Replace old tool-call payloads with compact summaries.

        Completed tool calls can include whole-file heredocs or large edit bodies.
        Keeping those exact arguments in later turns burns context without adding
        new state; the filesystem already holds the applied change. We preserve
        the user/assistant text and a short record of each completed tool call.
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

        if name == "bash":
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

    async def _stream_assistant(self) -> AssistantTurn:
        self.ui.begin_generation()
        self.ui.set_state(mascot.State.PREFILL)
        acc = StreamAccumulator()
        thinking_started = False
        generating = False

        async for event in self.client.stream_chat(
            messages=self.messages,
            tools=self.registry.schemas(),
            **self.config.gen_params(),
        ):
            acc.add(event)
            if isinstance(event, PrefillProgress):
                self.ui.set_prefill(event.percent, event.processed, event.total)
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
                self.ui.chat.flush_reasoning()
                self.ui.chat.flush_assistant()
                generating = False
                self.ui.set_state(mascot.State.BASH if event.name == "bash" else mascot.State.EDIT)
                self.ui.chat.begin_tool_call(event.index, event.name)
            elif isinstance(event, ToolCallArgsDelta):
                self.ui.chat.update_tool_call(event.index, event.fragment)
            elif isinstance(event, Usage):
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
                            Agent._history_tool_arguments(call.arguments),
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
        advice = rewrite_advice_message(repeated, exclude=self._last_rewrite_advice)
        self._last_rewrite_advice = advice
        return advice, False

    async def _auto_read_written_files(self, arguments: dict) -> str:
        """Anchors for files a bash command just wrote, to append to its result.

        Lets the model edit a freshly-written file immediately, without a separate
        read. Best-effort: returns ``""`` on any problem.
        """
        command = arguments.get("command") if isinstance(arguments, dict) else None
        if not isinstance(command, str) or not command:
            return ""
        if not _authored_write_targets(command):
            return ""
        files_tool = getattr(self.registry, "files", None)
        if files_tool is None:
            return ""
        try:
            return await asyncio.to_thread(build_auto_read_block, files_tool, command)
        except Exception:  # noqa: BLE001 - convenience only
            return ""
