"""The chat agent: a lightweight conversational answer, no tools.

The one stage with a persistent context: conversation turns accumulate on its
own message list so follow-up questions read naturally. Anything web-related
belongs to WEB mode; anything that changes the project belongs to NORMAL or
PLAN mode.
"""

from __future__ import annotations

from .base import SESSION_MAIN, XML_CALL_RULES, AgentSpec, compose_system_message

SPEC = AgentSpec(
    name="chat",
    session=SESSION_MAIN,
    allowed_tools=frozenset(),
    mascot="inference",
)

SYSTEM_PROMPT = f"""\
You are qwowl35, a general-purpose assistant. Answer directly and honestly,
and say when you're unsure. You have no tools: if a request needs project
changes, command execution, or a web lookup, say what you would do and ask
the user to switch mode and rephrase it as a task.
<<GROUNDING>>

{XML_CALL_RULES}"""


def system_message(cwd: str | None = None) -> dict:
    return compose_system_message(SYSTEM_PROMPT, cwd)
