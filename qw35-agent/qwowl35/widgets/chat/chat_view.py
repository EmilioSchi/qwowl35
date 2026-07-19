"""The scrollable chat transcript: one widget per message, streaming repaint
timer, and the per-tool render dispatch."""

from __future__ import annotations

import time
from typing import Any

from rich.console import Group, RenderableType
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

import theme
from config import TOOL_PREVIEW_LINES
from tools.diagnostics import split_trailing_section
from widgets.chat.card import _CardFrame, _agent_edge
from widgets.chat.markdown import _markdown
from widgets.chat.primitives import (
    _CURSOR,
    _BlockquoteFrame,
    _FullWidthLines,
    _line_with_bg,
    highlight_refs,
)
from widgets.chat.renderers.ask import (
    _ask_card,
    _ask_partial_questions,
    _ask_result_tree,
    _parse_ask_result,
)
from widgets.chat.renderers.code import (
    _anchor_header,
    _append_trailing,
    _body_has_diff,
    _capture_spawn_snippet,
    _parse_file_view,
    _render_anchored_code,
    _render_diff,
    _render_inspect_result,
    _render_spawn_snippet,
    _split_diff_section,
    _streaming_code_text,
)
from widgets.chat.renderers.files import (
    _glob_tree_text,
    _grep_tree_text,
    _ls_tree_text,
    _parse_glob_result,
    _parse_grep_result,
    _parse_ls_result,
)
from widgets.chat.renderers.shell import (
    _command_rows,
    _output_rows,
    _render_advisory,
    _split_bash_advisories,
)
from widgets.chat.renderers.todo import _parse_todo_result, _plan_card, _todo_card_text
from widgets.chat.terminal_chrome import (
    _head_capped_rows,
    _prompt_text,
    _term_bg,
    _terminal_window,
)
from widgets.chat.thinking_block import ThinkingBlock
from widgets.chat.tool_args import (
    _FILE_CHANGE_TOOLS,
    _FILE_READ_TOOLS,
    _FILE_VIEW_TOOLS,
    _SHELL_TOOL_NAMES,
    _closed_string_arg,
    _command_from_args,
    _compact_args,
    _parse_args,
    _path_from_args,
    _recover_string_arg,
    _spawn_chip,
)
from widgets.chat.tool_block import ToolBlock, _result_tokens, _window_title


# Repaint streaming text ~20x/s so it visibly evolves while still batching the
# (relatively costly) markdown re-parse.
_REFRESH_INTERVAL = 0.05

# Reveal smoothing for tool-call args. The server streams the tool-call body
# incrementally (raw XML fragments while the call is being generated, see
# stream_tool_call_xml), so the display target only ever contains bytes that
# actually arrived; the reveal cursor trails it a few characters per tick to
# smooth bursty fragment arrival into a steady type-out. Once the call finishes
# (final args or its result arrives) the reveal fast-forwards — anything still
# untyped at that point would be replay, not streaming.
_REVEAL_MIN_STEP = 2
_REVEAL_MAX_STEP = 24
# Cursor blink: toggle every N ticks (N * _REFRESH_INTERVAL seconds).
_BLINK_TICKS = 10


_THINK_FRAME_TICKS = 7   # 7 * _REFRESH_INTERVAL = 0.35s per label frame


# Compact word badge standing in for each tool's name in the call header.
# Deliberately no emoji/unicode (see module docstring) — many terminals
# won't render glyphs here. Grouped by family; anything unlisted falls back to
# "?" via _tool_badge.
_TOOL_BADGES = {
    # shell
    "bash": "Sh",
    "run_shell_command": "Sh",
    # read / open ("beginTransaction" kept for pre-rename transcripts)
    "read_file": "Read",
    "beginTransaction": "Read",
    "inspect_file": "View",
    # search
    "grep_search": "Grep",
    "glob": "Glob",
    "list_directory": "List",
    # edit ("edit" kept for pre-rename transcripts + the freestyle delegator)
    "insert": "Add",
    "delete": "Del",
    "replace": "Edit",
    "edit": "Edit",
    # network
    "web_fetch": "Fetch",
    "search_engine": "Search",
    # planning
    "plan": "Plan",
    # ask the user
    "ask_user_question": "Ask",
    # sub-agent spawn / report (the planner's explorer)
    "explore": "Expl",
    "resume": "Resm",
}


# A fetched web page is mostly noise in the transcript (up to 100K chars); show
# only a few lines collapsed — far fewer than TOOL_PREVIEW_LINES — with the full
# text a Ctrl+o expand away. The model still receives the whole page.
_WEB_PREVIEW_LINES = 5


def _tool_badge(name: str) -> str:
    """A compact word badge standing in for the tool name."""
    return _TOOL_BADGES.get(name, "?")


def _tool_title(name: str, args: dict[str, Any] | None, color: str) -> Text:
    """Tool header: a black-backed badge in `color` (no bold) + grey path."""
    title = Text()
    title.append(_tool_badge(name), style=f"{color} on {theme.BG_BASE}")
    # The header trails with the call's subject: a file path, a URL for a web
    # fetch (so a web_fetch header reads "@>  https://…"), or the query for a
    # web search.
    path = (
        (args or {}).get("file")
        or (args or {}).get("path")
        or (args or {}).get("file_path")
        or (args or {}).get("url")
        or (args or {}).get("query")
    )
    if isinstance(path, str) and path:
        title.append("  ")
        title.append(path, style=theme.FG_MUTED)
    return title


def _preview_lines(body: str, expanded: bool, max_lines: int = TOOL_PREVIEW_LINES) -> tuple[str, int]:
    lines = body.splitlines()
    if not expanded and len(lines) > max_lines:
        return "\n".join(lines[:max_lines]), len(lines) - max_lines
    if len(body) > 20000:
        return body[:20000] + "\n... (truncated)", 0
    return body, 0


class ChatView(VerticalScroll):
    """A vertical scroller holding one widget per message."""

    DEFAULT_CSS = """
    ChatView {
        height: 1fr;
        padding: 0 0 0 1;
        scrollbar-size-vertical: 1;
        scrollbar-background: $bg-base;
        scrollbar-background-hover: $bg-base;
        scrollbar-background-active: $bg-base;
        scrollbar-color: $scroll-bar;
        scrollbar-color-hover: $scroll-bar-hover;
        scrollbar-color-active: $scroll-bar-active;
        scrollbar-corner-color: $bg-base;
    }
    ChatView .msg { margin: 1 0 0 0; width: 1fr; }
    ChatView .user {
        color: $fg-bright;
        padding: 0 1;
    }
    ChatView .assistant { color: $fg-bright; }
    ChatView .thinking { color: $fg-faint; text-style: italic; padding: 0 0 0 1; }
    ChatView .tool-pending { color: $accent; padding: 0 1; }
    ChatView .tool-success { color: $success-soft; padding: 0 1; }
    ChatView .tool-error   { color: $error-soft; padding: 0 1; }
    ChatView .system { color: $fg-faint; text-style: italic; }
    ChatView .warning { color: $warning; text-style: bold; }
    ChatView .error { color: $error; text-style: bold; }
    ChatView .todo { background: $bg-surface; padding: 0 1; }
    ChatView .plan { background: $bg-surface; padding: 0 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._assistant: Static | None = None
        self._assistant_buf = ""
        self._assistant_dirty = False
        self._reasoning: ThinkingBlock | None = None  # the streaming segment
        self._tool_blocks: dict[int, ToolBlock] = {}  # in-flight calls by index
        self._collapsibles: list[ToolBlock] = []      # finished results (Ctrl+O)
        # Indices whose window the user closed (`x`) while the call was still
        # in flight: late args/results for them must not resurrect the widget.
        self._closed_indices: set[int] = set()
        self.tools_expanded = False
        self._frame = 0          # tick counter, drives the reveal cursor blink
        self._blink_on = True    # current cursor-visible phase

    def on_mount(self) -> None:
        self.can_focus = True
        # One persistent timer drives smooth repainting of streaming text.
        self.set_interval(_REFRESH_INTERVAL, self._tick)

    # -- streaming repaint -------------------------------------------------- #
    def _tick(self) -> None:
        self._frame += 1
        blink_on = (self._frame // _BLINK_TICKS) % 2 == 0
        blink_changed = blink_on != self._blink_on
        self._blink_on = blink_on
        if self._assistant_dirty and self._assistant is not None:
            self._assistant.update(_markdown(self._assistant_buf))
            self._assistant_dirty = False
            self._bump()
        thinking = self._reasoning
        if thinking is not None and not thinking.done:
            advance = self._frame % _THINK_FRAME_TICKS == 0  # 0.35s label cadence
            if advance:
                thinking.anim_frame += 1
            # A collapsed dirty body stays unpainted (it is hidden; `dirty`
            # persists until an expand or the flush repaints it).
            if advance or (thinking.dirty and thinking.expanded):
                grew = thinking.dirty and thinking.expanded
                thinking.repaint()
                if grew:
                    self._bump()
        for index, block in list(self._tool_blocks.items()):
            if (
                block.tool_name == "ask_user_question"
                and not block.result_ready
                and self._frame % _THINK_FRAME_TICKS == 0
            ):
                # The ask card shimmers from the first streamed token until
                # its modals resolve — same label cadence as the thinking card.
                block.anim_frame += 1
                block.args_dirty = True
            target = self._call_target(block)
            if block.reveal > len(target):
                # The target can shrink mid-stream (raw XML → recognized
                # command); keep the cursor pinned to its tip.
                block.reveal = len(target)
            if block.stream_done or block.result_ready:
                # Generation of this call is over: anything still untyped
                # would be a fake replay, not streaming — fast-forward.
                if block.reveal < len(target):
                    block.reveal = len(target)
                    block.args_dirty = True
                if block.result_ready:
                    self._promote_result(index)
                    continue
            revealing = block.reveal < len(target)
            if revealing:
                # Smooth real fragment arrival: trail the streamed target a
                # few chars per tick. Adaptive so a backlog catches up, but
                # capped so a burst still types out rather than dumping.
                step = min(
                    _REVEAL_MAX_STEP,
                    max(_REVEAL_MIN_STEP, (len(target) - block.reveal) // 8),
                )
                block.reveal = min(len(target), block.reveal + step)
            # Repaint when content advanced or new args arrived. While typing,
            # the per-tick advance already repaints, so the blink rides along.
            if block.args_dirty or revealing:
                block.update(self._render_tool_call(block))
                block.args_dirty = False
                self._bump()

    # -- low-level append --------------------------------------------------- #
    def _at_bottom(self) -> bool:
        return self.scroll_offset.y >= self.max_scroll_y - 1

    def _append(self, widget: Static) -> Static:
        stick = self._at_bottom()
        self.mount(widget)
        if stick:
            self.scroll_end(animate=False)
        return widget

    def _bump(self) -> None:
        if self._at_bottom():
            self.scroll_end(animate=False)

    # -- user --------------------------------------------------------------- #
    def add_user(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        frame = _BlockquoteFrame(_markdown(text), title="User", timestamp=ts)
        self._append(Static(frame, classes="msg user"))

    def add_system(self, text: str) -> None:
        self._append(Static(Text(text), classes="msg system"))

    def clear(self) -> None:
        """Wipe the transcript: remove every message widget and streaming state.

        Used by the ``/clear`` command. The persistent repaint timer set up in
        ``on_mount`` keeps running; only content and in-flight buffers reset.
        """
        self.remove_children()
        self._assistant = None
        self._assistant_buf = ""
        self._assistant_dirty = False
        self._reasoning = None
        self._tool_blocks.clear()
        self._collapsibles.clear()
        self._closed_indices.clear()
        self.tools_expanded = False

    # -- assistant (streaming markdown) ------------------------------------- #
    def add_assistant_chunk(self, text: str) -> None:
        self._flush_held_tools()
        if self._assistant is None:
            self._assistant_buf = text
            self._assistant = self._append(
                Static(_markdown(self._assistant_buf), classes="msg assistant")
            )
        else:
            self._assistant_buf += text
        self._assistant_dirty = True

    def flush_assistant(self) -> None:
        if self._assistant is not None:
            self._assistant.update(_markdown(self._assistant_buf))
            self._bump()
        self._assistant = None
        self._assistant_buf = ""
        self._assistant_dirty = False

    # -- reasoning (collapsed animated card) --------------------------------- #
    def add_reasoning_chunk(self, text: str) -> None:
        self._flush_held_tools()
        if self._reasoning is None:
            block = ThinkingBlock()
            block.body = text
            self._reasoning = block
            self._append(block)
            block.repaint()  # collapsed animated label from frame 0
        else:
            self._reasoning.body += text
        self._reasoning.dirty = True

    def flush_reasoning(self) -> None:
        if self._reasoning is not None:
            self._reasoning.done = True  # freeze BEFORE the final paint
            self._reasoning.repaint()
            self._bump()
        # Detach: the card stays mounted (frozen, still clickable); the next
        # reasoning segment gets a fresh card of its own.
        self._reasoning = None

    # -- tools (streaming call box + collapsible result) -------------------- #
    def _flush_held_tools(self) -> None:
        """Promote any finished-but-still-typing tool box immediately.

        A held block (result arrived, reveal not yet complete) is waiting on the
        timer to finish its type-out. When new turn output arrives — another tool
        call, reasoning, or assistant text — we must let it go now: the call
        index resets every turn, so a lingering block at index 0 would be
        orphaned (its output never shown) when the next turn reuses that index,
        and its output must not paint after newer content. The reveal already had
        the tool-execution + prefill gap to play.
        """
        for index in [i for i, b in self._tool_blocks.items() if b.result_ready]:
            self._promote_result(index)

    def begin_tool_call(self, index: int, name: str) -> ToolBlock:
        self._flush_held_tools()
        self._closed_indices.discard(index)  # indices reset every turn
        block = ToolBlock(name)
        self._tool_blocks[index] = block
        self._append(block)
        block.update(self._render_tool_call(block))
        return block

    def update_tool_call(self, index: int, fragment: str) -> None:
        if index in self._closed_indices:
            return  # closed via `x` mid-stream; don't resurrect the window
        block = self._tool_blocks.get(index)
        if block is None:  # defensive: args before begin
            block = self.begin_tool_call(index, "?")
        block.args_buf += fragment
        block.args_dirty = True

    def name_tool_call(self, index: int, name: str) -> None:
        """The streamed call's function name became known (qw35 side-channel).

        The display target changes shape here (raw XML → extracted command/
        detail), so clamp the reveal into the new target: the recognized view
        starts typing from wherever its content currently ends."""
        block = self._tool_blocks.get(index)
        if block is None:
            return
        block.tool_name = name
        block.reveal = min(block.reveal, len(self._call_target(block)))
        block.args_dirty = True

    def finalize_tool_call(self, index: int, arguments: str) -> None:
        """Authoritative parsed arguments arrived; the call finished streaming."""
        block = self._tool_blocks.get(index)
        if block is None:
            return
        block.args_buf = arguments
        block.stream_done = True
        block.args_dirty = True

    def demote_tool_call(self, index: int) -> None:
        """The streamed block was not a tool call after all — drop its box.

        The raw text re-arrives as ordinary assistant/reasoning deltas, so
        removing the widget keeps a single source of truth on screen."""
        block = self._tool_blocks.pop(index, None)
        if block is None:
            return
        block.remove()

    def add_tool_result(self, index: int, name: str, text: str, is_error: bool = False) -> None:
        if index in self._closed_indices:
            # The user closed this call's window (`x`); swallow the result so
            # the defensive re-create below can't resurrect it. Display-only:
            # the model's conversation history still carries call + result.
            self._closed_indices.discard(index)
            return
        # A successful `plan` write renders as the pretty checklist card, not
        # a raw result box: the streamed call box is dropped and the card is
        # the one visual.
        if name == "plan" and not is_error:
            rows = _parse_todo_result(text)
            if rows is not None:
                # A presentation/replan carries the plan markdown in its call
                # args; progress-only calls don't, so those stay checklist-only.
                block = self._tool_blocks.get(index)
                plan_arg = _parse_args(block.args_buf).get("plan") if block else None
                self.demote_tool_call(index)
                if isinstance(plan_arg, str) and plan_arg.strip():
                    self._append(Static(_plan_card(plan_arg.strip()), classes="msg plan"))
                self._append(Static(_todo_card_text(rows), classes="msg todo"))
                return
        block = self._tool_blocks.get(index)
        if (
            block is None
            and name == "explore"
            and not is_error
            and text.startswith("Exploration findings:")
        ):
            # The explorer sub-agent streams into this same ChatView with its
            # own per-turn indices; a collision with the parent's `explore`
            # index orphans the streamed Spawn card and its `resume` pops the
            # slot. Fabricating a block here could only draw an EMPTY Spawn
            # card — and the findings already streamed in the Resume card just
            # above — so swallow the visual. Display-only: the model's history
            # still carries call + result.
            return
        if block is None:  # defensive: result with no streamed call
            block = ToolBlock(name)
            self._tool_blocks[index] = block
            self._append(block)
        block.full_result = text
        block.is_error = is_error
        # Tools often finish faster than the command types out. Let the reveal
        # play to the end first; `_tick` promotes the block once it catches up.
        if block.reveal < len(self._call_target(block)):
            block.result_ready = True
            return
        self._promote_result(index)

    def note_question_answer(self, question_index: int, answer: str | None) -> None:
        """A QuestionModal resolved (answer text, or None when dismissed):
        grow the live ask card's tree. Display-only — the tool result the
        model sees is built independently by PlanTools._ask. No-op when no
        ask call is live (headless callbacks, tests, a closed window)."""
        live = [
            (index, b)
            for index, b in self._tool_blocks.items()
            if b.tool_name == "ask_user_question" and b.full_result is None
        ]
        if not live:
            return
        # Tool calls execute sequentially, so the lowest un-resulted index is
        # the call whose modals are up right now.
        _, block = min(live)
        block.ask_answers[question_index] = answer
        block.update(self._render_tool_call(block))
        self._bump()

    def _promote_result(self, index: int) -> None:
        """Swap a finished, fully-revealed call from pending box to result box."""
        block = self._tool_blocks.pop(index, None)
        if block is None:
            return
        block.result_ready = False
        block.reveal = len(self._call_target(block))  # result shows full command
        block.remove_class("tool-pending")
        block.add_class("tool-error" if block.is_error else "tool-success")
        self._collapsibles.append(block)
        block.update(self._render_tool_result(block))
        self._bump()

    def toggle_tools(self) -> None:
        self.tools_expanded = not self.tools_expanded
        for block in self._collapsibles:
            block.update(self._render_tool_result(block))
        for block in self._tool_blocks.values():
            block.args_dirty = True  # the persistent _tick repaints pending cards
        self._bump()

    def repaint_block(self, block: ToolBlock) -> None:
        """Repaint one block after a window-chrome interaction (collapse)."""
        if block in self._collapsibles:
            block.update(self._render_tool_result(block))
        else:
            block.update(self._render_tool_call(block))

    def close_tool_block(self, block: ToolBlock) -> None:
        """Remove a tool window from the transcript (the `x` control).

        Visual only: the tool call and its result stay in the model's
        conversation history — the chat view never touches agent messages.
        """
        for index, pending in list(self._tool_blocks.items()):
            if pending is block:
                self._tool_blocks.pop(index)
                # Late fragments/results for this index must not re-mount it.
                self._closed_indices.add(index)
                break
        if block in self._collapsibles:
            self._collapsibles.remove(block)
        block.remove()

    def _call_detail(self, block: ToolBlock) -> str:
        """The non-bash args preview that types out under the label."""
        detail = _compact_args(_parse_args(block.args_buf))
        if not detail:
            detail = block.args_buf.strip()
            if len(detail) > 240:
                detail = detail[:240] + " ..."
        return detail

    def _call_target(self, block: ToolBlock) -> str:
        """Full text the reveal animation is typing toward."""
        if block.tool_name in _SHELL_TOOL_NAMES:
            return _command_from_args(block.args_buf)
        if block.tool_name in _FILE_CHANGE_TOOLS:
            content = _recover_string_arg(block.args_buf, "content")
            if content:
                return content
        if block.tool_name == "edit" and _spawn_chip(block) == "Editor":
            # The delegator spawning the Editor: the interesting text is the
            # change instructions, typed out inside the Spawn card.
            recovered = _recover_string_arg(block.args_buf, "instructions")
            if recovered:
                return recovered
        if block.tool_name in ("explore", "resume"):
            # The interesting text IS the long string arg: the exploration
            # task being requested, or the findings summary coming back.
            field = "task" if block.tool_name == "explore" else "summary"
            recovered = _recover_string_arg(block.args_buf, field)
            if recovered:
                return recovered
        if not block.tool_name:
            # Raw mode: the call is streaming but not yet recognized; the
            # target is the raw XML itself, untruncated.
            return block.args_buf
        return self._call_detail(block)

    def _delegator_target(self, block: ToolBlock) -> tuple[str, str] | None:
        """(filename, line_ranges) of a delegator edit, once BOTH are complete.

        Wire arg order is filename, line_ranges, instructions — both close
        early while the long instructions still stream, so the capture lands
        well before the editor runs.
        """
        if block.stream_done:
            args = _parse_args(block.args_buf)
            filename = args.get("filename")
            if isinstance(filename, str) and filename.strip():
                # _json_or_string can hand back an int for "12" — coerce.
                return filename.strip(), str(args.get("line_ranges") or "")
            return None
        filename = _closed_string_arg(block.args_buf, "filename")
        ranges = _closed_string_arg(block.args_buf, "line_ranges")
        if filename and filename.strip() and ranges is not None:
            return filename.strip(), ranges
        return None

    def _ensure_spawn_snippet(self, block: ToolBlock):
        """The frozen pre-edit slice for a Spawn card, capturing it lazily.

        The read must happen while the call is still in flight: once the
        result arrives the disk content is post-edit, so a block that never
        captured (restored session, late render) simply shows no slice.
        """
        if block.spawn_snippet is not None or block.spawn_snippet_tried:
            return block.spawn_snippet
        if block.full_result is not None:
            return None
        target = self._delegator_target(block)
        if target is None:
            return None  # args still streaming; retry on a later tick
        block.spawn_snippet_tried = True  # exactly one disk read
        block.spawn_snippet = _capture_spawn_snippet(*target)
        return block.spawn_snippet

    def _render_tool_call(self, block: ToolBlock) -> RenderableType:
        if not block.tool_name:
            # Raw mode: show the tool-call XML growing verbatim until the
            # function is recognized (name_tool_call) or the block demotes.
            target = self._call_target(block)
            revealing = block.reveal < len(target)
            shown = target[: block.reveal] if revealing else target
            label = _tool_title("", None, theme.ACCENT)
            text = Text(shown.lstrip("\r\n"), style=theme.FG_DIM)
            if revealing and self._blink_on:
                text.append(_CURSOR, style=theme.FG_DIM)
            if not text.plain:
                return label
            return Group(label, _FullWidthLines([_line_with_bg(text, theme.BG_BASE)], wrap=True))

        args = _parse_args(block.args_buf)
        label = _tool_title(block.tool_name, args, theme.ACCENT)
        target = self._call_target(block)
        revealing = block.reveal < len(target)
        shown = target[: block.reveal] if revealing else target
        cursor = revealing and self._blink_on

        if block.tool_name in _SHELL_TOOL_NAMES:
            # Pass the FULL command + a reveal budget (not a pre-sliced prefix):
            # the lexer highlights complete syntax once, and the reveal slices the
            # stable, already-coloured Text — so colours no longer flicker as the
            # type-out crosses token boundaries (quotes, heredocs, etc.).
            # No head cap while streaming: the type-out itself is the content.
            reveal = block.reveal if revealing else None
            bg = _term_bg()
            body = _command_rows(
                target,
                cursor=cursor,
                reveal=reveal,
                first_prompt=_prompt_text(block.prompt_host, block.prompt_path),
                bg=bg,
            )
            return _terminal_window(
                title=_window_title(block),
                # Neutral title (the controls carry the color); state shows
                # via the cursor/output, red only on failure.
                color=theme.FG_MUTED,
                body_rows=body,
                collapsed=block.collapsed,
                started_at=block.started_at,
                tokens=None,  # unknowable until the result arrives
                bg=bg,
            )

        chip = _spawn_chip(block)
        if chip:
            # A sub-agent spawn: the task/instructions type out inside the
            # bordered Spawn card, agent chip on the bottom edge.
            inner_parts: list[RenderableType] = []
            if chip == "Editor":
                filename = args.get("filename")
                ranges = args.get("line_ranges")
                if filename:
                    header = str(filename) + (f" (lines {ranges})" if ranges else "")
                    inner_parts.append(Text(header, style=theme.FG_MUTED))
                snippet = self._ensure_spawn_snippet(block)
                if snippet is not None:
                    inner_parts.extend(
                        _render_spawn_snippet(
                            snippet, block.expanded or self.tools_expanded
                        )
                    )
            if shown:
                task = Text(shown, style=theme.FG_BRIGHT)
                if cursor:
                    task.append(_CURSOR, style=theme.FG_BRIGHT)
                inner_parts.append(task)
            elif not inner_parts:
                inner_parts.append(Text("(waiting for task...)", style=theme.FG_FAINT))
            return _CardFrame(
                Group(*inner_parts),
                title="Spawn",
                chip=chip,
                timestamp=block.started_at,
                edge=_agent_edge(chip),
            )

        if block.tool_name == "resume":
            # The Explorer handing its findings back: markdown inside the
            # Resume card. No inline cursor — a cursor char spliced into
            # markdown source would distort the parse mid-stream.
            return _CardFrame(
                _markdown(shown),
                title="Resume",
                chip="Explorer",
                timestamp=block.started_at,
                edge=_agent_edge("Explorer"),
            )

        if block.tool_name in _FILE_CHANGE_TOOLS and _recover_string_arg(block.args_buf, "content"):
            # The target IS the recovered `content`: the new code streams as a
            # highlighted block (same reveal contract as the shell branch).
            reveal = block.reveal if revealing else None
            return Group(
                label,
                _streaming_code_text(
                    _path_from_args(args), target, cursor=cursor, reveal=reveal
                ),
            )

        if block.tool_name == "ask_user_question":
            if block.stream_done:
                # Waiting on the user's modals: the finalized args are the
                # authoritative tree. Unparseable args fall through to the
                # generic box (the tool will error anyway).
                questions = args.get("questions")
                if (
                    isinstance(questions, list)
                    and questions
                    and all(isinstance(q, dict) for q in questions)
                ):
                    return _ask_card(questions, block.ask_answers, block.anim_frame)
            else:
                # Still streaming: the card shows from the first token — the
                # shimmer label alone at first, then the tree grows as each
                # question/header string arrives in the partial buffer.
                return _ask_card(
                    _ask_partial_questions(block.args_buf),
                    block.ask_answers,
                    block.anim_frame,
                    cursor=self._blink_on,
                )

        if target:
            text = Text(shown, style=theme.FG_DIM)
            if cursor:
                text.append(_CURSOR, style=theme.FG_DIM)
            return Group(label, _FullWidthLines([_line_with_bg(text, theme.BG_BASE)], wrap=True))
        return label

    def _render_tool_result(self, block: ToolBlock) -> RenderableType:
        args = _parse_args(block.args_buf)
        body = block.full_result or ""
        expanded = block.expanded or self.tools_expanded
        color = theme.ERROR_SOFT if block.is_error else theme.FG_DIM
        title = _tool_title(block.tool_name, args, color)

        if block.tool_name in _SHELL_TOOL_NAMES:
            output, advisory = _split_bash_advisories(body)
            # Keep only the 20000-char backstop here; the line cap below is the
            # window's own HEAD-view budget, not TOOL_PREVIEW_LINES.
            output, _ = _preview_lines(output, True)
            bg = _term_bg()
            cmd_rows = _command_rows(
                _command_from_args(block.args_buf),
                cursor=False,
                first_prompt=_prompt_text(block.prompt_host, block.prompt_path),
                bg=bg,
            )
            out_rows = _output_rows(output, bg=bg) if output else []
            if expanded:
                rows, hidden = cmd_rows + out_rows, 0
            else:
                rows, hidden = _head_capped_rows(cmd_rows, out_rows)
            note = f"... +{hidden} lines (□ or Ctrl+o)" if hidden else None
            window = _terminal_window(
                title=_window_title(block),
                color=theme.ERROR_SOFT if block.is_error else theme.FG_MUTED,
                body_rows=rows,
                collapsed=block.collapsed,
                started_at=block.started_at,
                tokens=_result_tokens(block),
                note=note,
                bg=bg,
            )
            if advisory.strip() and not block.collapsed:
                # The empty Text separates the window from the advisory block.
                return Group(window, Text(""), *_render_advisory(advisory, expanded))
            return window

        # A delegator `edit` (spawned the Editor): keep the Spawn card on
        # screen, with the editor's summary + diff rendered beneath it. Hard
        # failures (no diff) drop to the plain error fallback like any edit.
        if _spawn_chip(block) == "Editor" and (
            not block.is_error or _body_has_diff(body)
        ):
            rendered = self._render_file_tool_result(block.tool_name, args, body, expanded)
            if rendered is not None:
                instructions = args.get("instructions")
                if not isinstance(instructions, str):
                    instructions = _recover_string_arg(block.args_buf, "instructions") or ""
                inner_parts: list[RenderableType] = []
                filename = args.get("filename")
                if filename:
                    ranges = args.get("line_ranges")
                    header = str(filename) + (f" (lines {ranges})" if ranges else "")
                    inner_parts.append(Text(header, style=theme.FG_MUTED))
                # Cache read only — the disk is post-edit by now, so a block
                # that never captured (restored session) shows no slice.
                if block.spawn_snippet is not None:
                    inner_parts.extend(
                        _render_spawn_snippet(block.spawn_snippet, expanded)
                    )
                if instructions:
                    inner_parts.append(Text(instructions, style=theme.FG_BRIGHT))
                card = _CardFrame(
                    Group(*inner_parts) if inner_parts else Text(""),
                    title="Spawn",
                    chip="Editor",
                    timestamp=block.started_at,
                    edge=_agent_edge("Editor"),
                )
                result_card = _CardFrame(
                    rendered,
                    title="Editor result",
                    chip="Editor",
                    timestamp=block.started_at,
                    edge=_agent_edge("Editor"),
                )
                return Group(card, result_card)

        # Index collision: the editor sub-agent's own tool calls (same
        # ChatView, per-turn indices) reused the delegator `edit` slot, so this
        # result landed on a fabricated block with no args — _spawn_chip can't
        # see the Editor shape. The Spawn card already streamed above; render
        # the report as a standalone "Editor result" card (mirrors the
        # explore/resume split), never the flat badge + text fallback below.
        if (
            block.tool_name == "edit"
            and body.startswith("Editor result for ")
            and (not block.is_error or _body_has_diff(body))
        ):
            rendered = self._render_file_tool_result(block.tool_name, args, body, expanded)
            if rendered is not None:
                return _CardFrame(
                    rendered,
                    title="Editor result",
                    chip="Editor",
                    timestamp=block.started_at,
                    edge=_agent_edge("Editor"),
                )

        # An attention-flagged edit (e.g. it introduced a syntax warning) is
        # still a successful edit with a real diff to show — only a hard
        # failure (no diff at all) should drop to the plain error fallback.
        # A read is the same story with no diff: the editor reads mid-edit files
        # that carry a syntax-check marker (→ is_error), but the read succeeded —
        # let it render as the anchored view. A genuinely failed read has no
        # anchored body, so `_render_file_tool_result` returns None → plain box.
        is_read = block.tool_name in _FILE_READ_TOOLS
        if block.tool_name in _FILE_VIEW_TOOLS and (
            not block.is_error or _body_has_diff(body) or is_read
        ):
            rendered = self._render_file_tool_result(block.tool_name, args, body, expanded)
            if rendered is not None:
                return Group(title, rendered)

        if block.tool_name == "web_fetch" and not block.is_error:
            return Group(title, self._render_web_fetch_result(body, expanded))

        # The explorer's four tools render as self-titled cards (their header
        # row already carries the subject), so no badge title row — it would
        # only repeat the path/pattern. Parse failures fall through to the
        # plain result box below.
        if not block.is_error and not body.startswith("Error:"):
            if block.tool_name == "explore" and body.startswith("Exploration findings:"):
                # The findings already streamed in the explorer's own Resume
                # card just above — showing them again here would print the
                # whole summary twice. Collapsed: the Spawn card plus a
                # one-line pointer; Ctrl+o still reveals the raw findings.
                # Any other body (errors, "finished without calling resume")
                # keeps the generic Result box — it has no Resume-card twin.
                task = _recover_string_arg(block.args_buf, "task") or ""
                card = _CardFrame(
                    Text(task, style=theme.FG_BRIGHT) if task else Text(""),
                    title="Spawn",
                    chip="Explorer",
                    timestamp=block.started_at,
                    edge=_agent_edge("Explorer"),
                )
                if expanded:
                    preview, hidden = _preview_lines(body, expanded)
                    findings = highlight_refs(preview)
                    if hidden:
                        findings.append(
                            Text(f"\n... {hidden} more lines (Ctrl+o to expand)", style="dim")
                        )
                    return Group(card, findings)
                return Group(
                    card,
                    Text("findings reported in the Resume card above", style="dim"),
                )
            if block.tool_name == "resume":
                summary = _recover_string_arg(block.args_buf, "summary") or ""
                if summary.strip():
                    return _CardFrame(
                        _markdown(summary),
                        title="Resume",
                        chip="Explorer",
                        timestamp=block.started_at,
                        edge=_agent_edge("Explorer"),
                    )
            if block.tool_name == "list_directory":
                parsed = _parse_ls_result(body)
                if parsed is not None:
                    path, total, entries, hidden = parsed
                    return _ls_tree_text(path, total, entries, hidden, expanded)
            if block.tool_name == "glob":
                parsed = _parse_glob_result(body)
                if parsed is not None:
                    pattern, base, total, paths, hidden = parsed
                    return _glob_tree_text(pattern, base, total, paths, hidden, expanded)
            if block.tool_name == "grep_search":
                parsed = _parse_grep_result(body)
                if parsed is not None:
                    pattern, scope, total, groups, notes = parsed
                    return _grep_tree_text(pattern, scope, total, groups, notes, expanded)
            if block.tool_name == "inspect_file":
                return _render_inspect_result(args, body, expanded)
            if block.tool_name == "ask_user_question":
                # The frozen ask card: label at rest + only the user's
                # answers. The header row already carries the subject, so no
                # badge title; parse failures keep the plain Result box.
                pairs = _parse_ask_result(body)
                if pairs is not None:
                    return _ask_result_tree(pairs)

        # Success keeps just the gray body under the badge title; only errors
        # earn the loud "Result" label.
        preview, hidden = _preview_lines(body, expanded)
        out = Text("Result\n" if block.is_error else "", style=color)
        out.append(highlight_refs(preview))
        if hidden:
            out.append(Text(f"\n... {hidden} more lines (Ctrl+o to expand)", style="dim"))
        return Group(title, out)

    def _render_web_fetch_result(self, body: str, expanded: bool) -> RenderableType:
        """A fetched page compactly: a size summary + a short preview (only
        ``_WEB_PREVIEW_LINES`` collapsed), the rest a Ctrl+o expand away."""
        total_lines = len(body.splitlines())
        out = Text()
        out.append(
            f"Fetched {total_lines} lines · {len(body):,} chars\n",
            style=theme.FG_GHOST,
        )
        preview, hidden = _preview_lines(body, expanded, max_lines=_WEB_PREVIEW_LINES)
        out.append(highlight_refs(preview))
        if hidden:
            out.append(Text(f"\n... {hidden} more lines (Ctrl+o to expand)", style="dim"))
        return out

    def _render_file_tool_result(
        self,
        name: str,
        args: dict[str, Any],
        body: str,
        expanded: bool,
    ) -> RenderableType | None:
        fallback_path = _path_from_args(args)

        # A trailing diagnostics section (tools/diagnostics grammar) is never
        # diff or file content: carve it off before any diff/anchor parsing —
        # a result without the Current-snippet used to swallow the whole
        # `Syntax check (…)` block into the diff text and draw its bullets as
        # removal rows — and render it through the syntax-status route below.
        body, diagnostics = split_trailing_section(body)

        if name in _FILE_CHANGE_TOOLS:
            intro_lines, diff, after = _split_diff_section(body)
            parts: list[RenderableType] = []
            diag_path = fallback_path
            if intro_lines:
                parts.append(Text("\n".join(intro_lines), style=theme.FG_BRIGHT))
            if diff:
                parts.append(Text("Diff", style=f"bold {theme.FG_DIM}"))
                parts.append(_render_diff(diff))
            if after:
                path, header, anchored, trailing = _parse_file_view(after, fallback_path)
                diag_path = path or fallback_path
                if header and anchored:
                    parts.append(Text(header, style=f"bold {theme.FG_DIM}"))
                    limited = anchored
                    hidden = 0
                    if not expanded and len(anchored) > TOOL_PREVIEW_LINES:
                        limited = anchored[:TOOL_PREVIEW_LINES]
                        hidden = len(anchored) - TOOL_PREVIEW_LINES
                    parts.append(_render_anchored_code(path, limited))
                    if hidden:
                        parts.append(Text(f"... {hidden} more lines (Ctrl+o to expand)", style="dim"))
                    _append_trailing(parts, trailing, expanded, path=diag_path)
                elif after.strip():
                    preview, hidden = _preview_lines(after, expanded)
                    text = highlight_refs(preview)
                    if hidden:
                        text.append(Text(f"\n... {hidden} more lines (Ctrl+o to expand)", style="dim"))
                    parts.append(text)
            if diagnostics:
                _append_trailing(parts, diagnostics.splitlines(), expanded, path=diag_path)
            return Group(*parts) if parts else None

        path, header, anchored, trailing = _parse_file_view(body, fallback_path)
        if not anchored:
            return None
        # Repeat reads of the same file omit the anchors header (token saving on
        # the model side). Synthesize a consistent one so the view looks the same
        # on every read instead of degrading to plain text after the first.
        if not header:
            header = _anchor_header(path or fallback_path)
        limited = anchored
        hidden = 0
        if not expanded and len(anchored) > TOOL_PREVIEW_LINES:
            limited = anchored[:TOOL_PREVIEW_LINES]
            hidden = len(anchored) - TOOL_PREVIEW_LINES
        parts = [
            Text(header, style=f"bold {theme.FG_DIM}"),
            _render_anchored_code(path, limited),
        ]
        if hidden:
            parts.append(Text(f"... {hidden} more lines (Ctrl+o to expand)", style="dim"))
        _append_trailing(parts, trailing, expanded, path=path or fallback_path)
        if diagnostics:
            _append_trailing(parts, diagnostics.splitlines(), expanded, path=path or fallback_path)
        return Group(*parts)
