"""The plan-approval gate: present the plan, get Approve / Revise / Reject.

Mirrors the bash ApprovalModal's borderless keyboard-driven look: the plan
scrolls, three selectable lines below, and choosing Revise (or pressing Tab)
opens a one-line field whose text is fed back to the planner as revision
feedback.

The plan arrives as one string: markdown with `render_todos` checklist lines
(`[ ]/[>]/[x] <ref>: <content>`) spliced in. Markdown would collapse those
lines into a single wrapped paragraph, so the string is split here — prose
segments render as markdown, checklist runs as a styled todo card matching
the chat log's glyph language (✔ / ▶ / ○).
"""

from __future__ import annotations

import re

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static

import theme
from tools.plan import PlanDecision
from widgets.chat.markdown import _markdown

_OPTIONS = ("Approve and start", "Write revision feedback", "Reject")

# One `render_todos` line: status mark, 1-based position (the ref's trailing
# 2-hex content hash is model plumbing — parsed away, never shown), content.
_TODO_LINE = re.compile(r"^\[( |>|x)\] (\d+)[0-9a-f]{2}: (.*)$")


def split_plan_segments(plan: str) -> list[tuple[str, object]]:
    """Split the modal text into ("md", source_str) and ("todos", rows)
    segments, where rows are (mark, number, content) tuples from consecutive
    checklist lines."""
    segments: list[tuple[str, object]] = []
    md: list[str] = []
    todos: list[tuple[str, str, str]] = []

    def flush_md() -> None:
        text = "\n".join(md).strip()
        md.clear()
        if text:
            segments.append(("md", text))

    def flush_todos() -> None:
        if todos:
            segments.append(("todos", list(todos)))
            todos.clear()

    for line in plan.splitlines():
        match = _TODO_LINE.match(line)
        if match is not None:
            flush_md()
            todos.append(match.groups())
        else:
            flush_todos()
            md.append(line)
    flush_md()
    flush_todos()
    return segments


def _todo_card(rows: list[tuple[str, str, str]]) -> Text:
    """The checklist rows as one styled block, one glyph-led line per todo."""
    card = Text()
    for i, (mark, number, content) in enumerate(rows):
        if i:
            card.append("\n")
        if mark == "x":
            card.append(" ✔ ", style=theme.SUCCESS)
            card.append(f"{number}. ", style=theme.FG_GHOST)
            card.append(content, style=f"strike {theme.FG_GHOST}")
        elif mark == ">":
            card.append(" ▶ ", style=f"bold {theme.ACCENT}")
            card.append(f"{number}. ", style=f"bold {theme.ACCENT}")
            card.append(content, style=f"bold {theme.FG_BRIGHT}")
        else:
            card.append(" ○ ", style=theme.ACCENT)
            card.append(f"{number}. ", style=theme.FG_DIM)
            card.append(content, style=theme.FG_BRIGHT)
    return card


class PlanApprovalModal(ModalScreen[PlanDecision]):
    DEFAULT_CSS = """
    PlanApprovalModal { align: center middle; background: transparent; }
    PlanApprovalModal #panel {
        width: 90%;
        max-width: 90%;
        height: auto;
        max-height: 100%;
        background: $bg-base;
        border: none;
        padding: 1 2;
    }
    PlanApprovalModal #plan-scroll {
        width: 1fr;
        height: auto;
        max-height: 85%;
        margin-bottom: 1;
        scrollbar-size-vertical: 1;
        scrollbar-background: $bg-base;
        scrollbar-background-hover: $bg-base;
        scrollbar-background-active: $bg-base;
        scrollbar-color: $scroll-bar;
        scrollbar-color-hover: $scroll-bar-hover;
        scrollbar-color-active: $scroll-bar-active;
        scrollbar-corner-color: $bg-base;
    }
    PlanApprovalModal #plan-header { width: 1fr; margin-bottom: 1; }
    PlanApprovalModal .plan-md { width: 1fr; margin-bottom: 1; }
    PlanApprovalModal .plan-todos {
        width: 1fr;
        background: $bg-surface;
        padding: 1 2;
        margin-bottom: 1;
    }
    PlanApprovalModal .option { width: 1fr; color: $fg-bright; padding: 0 1; }
    PlanApprovalModal .option-active { background: $bg-surface; color: $fg-bright; text-style: bold; }
    PlanApprovalModal #revise-input {
        display: none;
        width: 1fr;
        height: 3;
        padding: 1 2;
        margin-top: 1;
        border: none;
        background: $bg-surface;
        color: $fg-bright;
    }
    """

    can_focus = True

    def __init__(self, plan: str) -> None:
        super().__init__()
        self._plan = plan
        self._selected = 0
        self._mode = "select"  # or "revise"

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            with VerticalScroll(id="plan-scroll"):
                yield Static(Text("Proposed plan", style=theme.ACCENT), id="plan-header")
                segments = split_plan_segments(self._plan) or [("md", self._plan)]
                for kind, payload in segments:
                    if kind == "todos":
                        yield Static(_todo_card(payload), classes="plan-todos")
                    else:
                        yield Static(_markdown(payload), classes="plan-md")
            for i, _label in enumerate(_OPTIONS):
                yield Static("", id=f"opt{i}", classes="option")
            yield Input(placeholder="What should change before approval…", id="revise-input")

    def on_mount(self) -> None:
        self._render_options()
        self.focus()

    def _render_options(self) -> None:
        for i, label in enumerate(_OPTIONS):
            opt = self.query_one(f"#opt{i}", Static)
            marker = "› " if i == self._selected else "  "
            opt.update(f"{marker}{i + 1}. {label}")
            opt.set_class(i == self._selected, "option-active")

    def on_key(self, event) -> None:
        if self._mode == "revise":
            if event.key == "escape":
                self._exit_revise_mode()
                event.stop()
            return

        key = event.key
        if key in ("1", "2", "3"):
            self._selected = int(key) - 1
            self._render_options()
            self._confirm()
            event.stop()
        elif key == "up":
            self._selected = (self._selected - 1) % len(_OPTIONS)
            self._render_options()
            event.stop()
        elif key == "down":
            self._selected = (self._selected + 1) % len(_OPTIONS)
            self._render_options()
            event.stop()
        elif key == "enter":
            self._confirm()
            event.stop()
        elif key == "tab":
            self._enter_revise_mode()
            event.stop()
        elif key == "escape":
            self.dismiss(PlanDecision(kind="reject"))
            event.stop()

    def _confirm(self) -> None:
        if self._selected == 0:
            self.dismiss(PlanDecision(kind="approve"))
        elif self._selected == 2:
            self.dismiss(PlanDecision(kind="reject"))
        else:
            self._enter_revise_mode()

    def _enter_revise_mode(self) -> None:
        self._mode = "revise"
        self._selected = 1
        self._render_options()
        # The 85% scroll cap leaves no room for the input on short terminals;
        # trim it to an exact cell count while the input is visible.
        scroll = self.query_one("#plan-scroll")
        scroll.styles.max_height = max(3, self.size.height - 12)
        field = self.query_one("#revise-input", Input)
        field.styles.display = "block"
        field.focus()

    def _exit_revise_mode(self) -> None:
        self._mode = "select"
        scroll = self.query_one("#plan-scroll")
        scroll.styles.max_height = None
        field = self.query_one("#revise-input", Input)
        field.value = ""
        field.styles.display = "none"
        self.focus()

    @on(Input.Submitted, "#revise-input")
    def _revise_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.stop()
        if not text:
            return
        self.dismiss(PlanDecision(kind="revise", text=text))
