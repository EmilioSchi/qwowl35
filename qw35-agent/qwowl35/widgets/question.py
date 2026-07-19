"""The ask_user_question modal: one question, its 2-4 options, plus Other.

The planner's `ask_user_question` carries up to four questions; the app opens
one modal per question and collects {question: answer}. Layout and keys follow
the bash ApprovalModal: number keys / arrows select, Enter confirms, Tab (or
picking Other) opens a free-text line, Escape dismisses (no answer). When the
question sets multiSelect, Space and digit keys toggle options instead of
confirming, Enter submits the toggled set, and the answer is the selected
labels joined with ", ".
"""

from __future__ import annotations

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static

import theme

OTHER_LABEL = "Other (write your own)"


class QuestionModal(ModalScreen[str | None]):
    """Resolves to the chosen answer text, or None when dismissed."""

    DEFAULT_CSS = """
    QuestionModal { align: center middle; background: transparent; }
    QuestionModal #panel {
        width: 90%;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $bg-base;
        border: none;
        padding: 1 2;
    }
    QuestionModal #question-scroll {
        width: 1fr;
        height: auto;
        max-height: 50%;
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
    QuestionModal #question { width: 1fr; }
    QuestionModal .option { width: 1fr; color: $fg-bright; padding: 0 1; }
    QuestionModal .option-active { background: $bg-surface; color: $fg-bright; text-style: bold; }
    QuestionModal #other-input {
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

    def __init__(self, question: dict) -> None:
        super().__init__()
        self._question = question
        options = question.get("options") or []
        self._labels = [str(option.get("label", "")) for option in options]
        self._descriptions = [str(option.get("description", "")) for option in options]
        self._selected = 0
        self._mode = "select"  # or "other"
        self._multi = bool(question.get("multiSelect"))
        self._toggled: set[int] = set()

    def _option_count(self) -> int:
        return len(self._labels) + 1  # + Other

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            with VerticalScroll(id="question-scroll"):
                header = self._question.get("header")
                text = Text()
                if header:
                    text.append(f"[{header}] ", style=theme.ACCENT)
                text.append(str(self._question.get("question", "")), style=theme.FG_BRIGHT)
                for label, description in zip(self._labels, self._descriptions):
                    text.append(f"\n  • {label}", style=theme.FG_BRIGHT)
                    if description:
                        text.append(f" — {description}", style=theme.FG_DIM)
                yield Static(text, id="question")
            for i in range(self._option_count()):
                yield Static("", id=f"opt{i}", classes="option")
            yield Input(placeholder="Write your own answer…", id="other-input")

    def on_mount(self) -> None:
        self._render_options()
        self.focus()

    def _render_options(self) -> None:
        labels = [*self._labels, OTHER_LABEL]
        for i, label in enumerate(labels):
            opt = self.query_one(f"#opt{i}", Static)
            marker = "› " if i == self._selected else "  "
            check = ""
            if self._multi and i < len(self._labels):
                check = "[x] " if i in self._toggled else "[ ] "
            opt.update(f"{marker}{i + 1}. {check}{label}")
            opt.set_class(i == self._selected, "option-active")

    def on_key(self, event) -> None:
        if self._mode == "other":
            if event.key == "escape":
                self._exit_other_mode()
                event.stop()
            return

        key = event.key
        count = self._option_count()
        if key.isdigit() and 1 <= int(key) <= count:
            self._selected = int(key) - 1
            if self._multi and self._selected < len(self._labels):
                self._toggle(self._selected)
            else:
                self._render_options()
                self._confirm()
            event.stop()
        elif key == "space" and self._multi:
            if self._selected < len(self._labels):
                self._toggle(self._selected)
            else:
                self._enter_other_mode()
            event.stop()
        elif key == "up":
            self._selected = (self._selected - 1) % count
            self._render_options()
            event.stop()
        elif key == "down":
            self._selected = (self._selected + 1) % count
            self._render_options()
            event.stop()
        elif key == "enter":
            self._confirm()
            event.stop()
        elif key == "tab":
            self._enter_other_mode()
            event.stop()
        elif key == "escape":
            self.dismiss(None)
            event.stop()

    def _toggle(self, index: int) -> None:
        self._selected = index
        if index in self._toggled:
            self._toggled.discard(index)
        else:
            self._toggled.add(index)
        self._render_options()

    def _confirm(self) -> None:
        if self._multi:
            if self._selected >= len(self._labels):
                self._enter_other_mode()
                return
            if not self._toggled:
                return
            self.dismiss(self._joined_answer())
        elif self._selected < len(self._labels):
            self.dismiss(self._labels[self._selected])
        else:
            self._enter_other_mode()

    def _joined_answer(self, extra: str | None = None) -> str:
        parts = [self._labels[i] for i in sorted(self._toggled)]
        if extra:
            parts.append(extra)
        return ", ".join(parts)

    def _enter_other_mode(self) -> None:
        self._mode = "other"
        self._selected = len(self._labels)
        self._render_options()
        field = self.query_one("#other-input", Input)
        field.styles.display = "block"
        field.focus()

    def _exit_other_mode(self) -> None:
        self._mode = "select"
        field = self.query_one("#other-input", Input)
        field.value = ""
        field.styles.display = "none"
        self.focus()

    @on(Input.Submitted, "#other-input")
    def _other_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.stop()
        if not text:
            return
        if self._multi:
            self.dismiss(self._joined_answer(extra=text))
        else:
            self.dismiss(text)
