"""The ask_user_question card: streaming question tree, answer growth, and
the frozen result tree."""

from __future__ import annotations

import json
import re

from rich.text import Text

import theme
from widgets.chat.primitives import _CURSOR, _shimmer_text


_ASK_LABEL = "~ Asking questions..."
# The exact result strings PlanTools._ask builds (tools/plan/tools.py); the
# parser below must track them or the card falls back to the plain Result box.
_ASK_DECLINED = "User declined to answer the questions."
_ASK_ANSWERS_PREFIX = "User has provided the following answers:"
_ASK_ANSWER_LINE = re.compile(r"^\*\*(.+?)\*\*: (.*)$")


_ASK_STREAM_FIELD = re.compile(r'"(question|header)"\s*:\s*"((?:[^"\\]|\\.)*)')


def _ask_partial_questions(buffer: str) -> list[dict]:
    """Question/header pairs recovered from a still-streaming call's buffer
    (raw XML or JSON fragments alike — the questions payload is JSON text in
    both dialects), so the ask card's tree grows token by token. Order of
    appearance drives the pairing: a `question` string starts a new row, a
    `header` decorates the latest row missing one. Escapes decode once whole;
    a fragment cut mid-escape shows raw until the next fragment lands."""
    questions: list[dict] = []
    for match in _ASK_STREAM_FIELD.finditer(buffer):
        field, raw = match.group(1), match.group(2)
        try:
            value = json.loads(f'"{raw}"')
        except Exception:
            value = raw
        if field == "question":
            questions.append({"question": value})
        elif questions and "header" not in questions[-1]:
            questions[-1]["header"] = value
    return questions


def _ask_card(
    questions: list[dict],
    answers: dict[int, str | None],
    frame: int,
    cursor: bool = False,
) -> Text:
    """The live ask_user_question card: a shimmering label over a guide-lined
    tree of the questions — growing row by row while the call still streams
    (``cursor`` blinks at the tip), then answer by answer as the user works
    through the modals; a dismissed modal leaves "(skipped)"."""
    tree = Text()
    tree.append_text(_shimmer_text(_ASK_LABEL, frame))
    for index, question in enumerate(questions):
        last = index == len(questions) - 1
        tree.append("\n")
        tree.append(" └─ " if last else " ├─ ", style=theme.FG_GHOST)
        header = str(question.get("header") or "").strip()
        if header:
            tree.append(f"[{header}] ", style=theme.ACCENT)
        tree.append(str(question.get("question") or ""), style=theme.FG_BRIGHT)
        if index in answers:
            answer = answers[index]
            tree.append("\n")
            tree.append("    " if last else " │  ", style=theme.FG_GHOST)
            if answer is None:
                tree.append("(skipped)", style=f"italic {theme.FG_FAINT}")
            else:
                tree.append("→ ", style=theme.FG_GHOST)
                tree.append(answer, style=theme.FG_BRIGHT)
    if cursor:
        tree.append(_CURSOR, style=theme.FG_BRIGHT)
    return tree


def _parse_ask_result(body: str) -> list[tuple[str, str]] | None:
    """Parse an ask_user_question success into (header, answer) pairs — []
    when the user declined every question, None when the text isn't the
    known result shape (the card then falls back to the plain Result box)."""
    stripped = body.strip()
    if stripped == _ASK_DECLINED:
        return []
    if not stripped.startswith(_ASK_ANSWERS_PREFIX):
        return None
    pairs: list[tuple[str, str]] = []
    for line in stripped[len(_ASK_ANSWERS_PREFIX):].splitlines():
        if not line.strip():
            continue
        match = _ASK_ANSWER_LINE.match(line)
        if match is None:
            return None
        pairs.append((match.group(1), match.group(2)))
    return pairs or None


def _ask_result_tree(pairs: list[tuple[str, str]]) -> Text:
    """The frozen ask_user_question card: the label at rest plus one
    guide-lined `header: answer` row per answered question — the full
    question texts already showed while the card was live."""
    tree = Text()
    tree.append(_ASK_LABEL, style=f"italic {theme.FG_FAINT}")
    if not pairs:
        tree.append("\n")
        tree.append(" └─ ", style=theme.FG_GHOST)
        tree.append("(declined — no answers)", style=f"italic {theme.FG_FAINT}")
        return tree
    for index, (header, answer) in enumerate(pairs):
        last = index == len(pairs) - 1
        tree.append("\n")
        tree.append(" └─ " if last else " ├─ ", style=theme.FG_GHOST)
        tree.append(header, style=theme.ACCENT)
        tree.append(": ", style=theme.FG_GHOST)
        tree.append(answer, style=theme.FG_BRIGHT)
    return tree
