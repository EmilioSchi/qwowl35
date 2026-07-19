"""The web agent: WEB mode's agent — find and fetch what the request needs.

Fresh context per turn: its own system prompt + the task. Toolset:
`search_engine` + `web_fetch` — only this agent sees them.
"""

from __future__ import annotations

from tools.web.search_engine import GUIDANCE as SEARCH_ENGINE_GUIDANCE
from tools.web.web_fetch import GUIDANCE as WEB_FETCH_GUIDANCE

from .base import SESSION_MAIN, XML_CALL_RULES, AgentSpec, compose_system_message

SPEC = AgentSpec(
    name="web",
    session=SESSION_MAIN,
    allowed_tools=frozenset({"web_fetch", "search_engine"}),
    mascot="web",
)

SYSTEM_PROMPT = f"""\
You are the qwowl35 web agent. Your job is to find and fetch the web content
the user's request needs and answer them directly with what you found. You
have exactly two tools: `search_engine` to find pages and `web_fetch` to read
one. When the request names a URL, fetch it straight away; otherwise search
first. One call at a time, only pages you actually need.

{SEARCH_ENGINE_GUIDANCE}

{WEB_FETCH_GUIDANCE}

When you have enough, finish by replying (no tool call) with the answer:
the facts and snippets the request depends on, each with the URL it came
from. Make the reply self-contained.
<<GROUNDING>>

{XML_CALL_RULES}"""


def system_message(cwd: str | None = None) -> dict:
    return compose_system_message(SYSTEM_PROMPT, cwd)


def build_task_message(goal: str, session_notes: str = "") -> dict:
    parts = [f"Task:\n{goal}"]
    if session_notes:
        parts.append(f"Earlier in this session:\n{session_notes}")
    return {"role": "user", "content": "\n\n".join(parts)}
