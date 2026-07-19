"""Todo checklist and plan cards rendered from `plan` tool results."""

from __future__ import annotations

import re

from rich.console import Group
from rich.text import Text

import theme
from widgets.chat.markdown import _markdown


# `plan` result lines: "[ ] ref: content" / "[>] ..." / "[x] ..." (refs are
# hashline-style: 1-based position + 2-hex content hash, e.g. "3a7").
_TODO_LINE = re.compile(r"^\[( |>|x)\] ([^:]+): (.*)$")


def _parse_todo_result(result: str) -> list[tuple[str, str, str]] | None:
    """Parse a `plan` success into (status, ref, content) rows, or None
    when the text is not a rendered todo list (fall back to the plain box).
    The list ends at the first blank line: the gate-decision / next-task
    sentence follows it in the tool result."""
    lines = result.splitlines()
    if not lines or not lines[0].startswith("Todo list updated:"):
        return None
    rows: list[tuple[str, str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            break
        match = _TODO_LINE.match(line)
        if match is None:
            return None
        mark, todo_ref, content = match.groups()
        status = {" ": "pending", ">": "in_progress", "x": "completed"}[mark]
        rows.append((status, todo_ref.strip(), content.strip()))
    return rows or None


def _todo_card_text(rows: list[tuple[str, str, str]]) -> Text:
    """The pretty todo card: one styled row per item, checklist glyphs."""
    done = sum(1 for status, _, _ in rows if status == "completed")
    card = Text()
    card.append("Todo list", style=f"bold {theme.ACCENT}")
    card.append(f"  {done}/{len(rows)} done", style=theme.FG_GHOST)
    for status, todo_id, content in rows:
        card.append("\n")
        if status == "in_progress":
            card.append(" ▶ ", style=f"bold {theme.ACCENT}")
            card.append(f"{todo_id}  ", style=f"bold {theme.ACCENT}")
            card.append(content, style=f"bold {theme.FG_BRIGHT}")
        elif status == "completed":
            card.append(" ✔ ", style=theme.SUCCESS)
            card.append(f"{todo_id}  ", style=theme.FG_GHOST)
            card.append(content, style=f"strike {theme.FG_GHOST}")
        else:
            card.append(" ○ ", style=theme.FG_DIM)
            card.append(f"{todo_id}  ", style=theme.FG_DIM)
            card.append(content, style=theme.FG_DIM)
    return card


def _plan_card(plan_md: str) -> Group:
    """The accepted plan's markdown under its own header, shown once above
    the first todo card so the approved plan survives the transient modal."""
    return Group(Text("Plan", style=f"bold {theme.ACCENT}"), _markdown(plan_md))
