"""AgentSpec and shared prompt scaffolding for the smart-mode agents.

One module per agent lives in this package; each exposes a ``SPEC``, its own
``system_message()``, and its stage's context builders. Every agent is fully
segregated: it advertises and may call ONLY its own toolset, and it starts
from a fresh context containing its own system prompt plus exactly the
material handed over from the previous stage — never another agent's tools or
working turns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from prompts import _platform_summary

# GPU sessions (the server's `qw35_session` request field). `plan` hosts the
# planner's persistent context so the plan↔execute alternation rides its own
# checkpoint lineage instead of re-prefilling.
SESSION_MAIN = "main"
SESSION_SCRATCH = "scratch"
SESSION_PLAN = "plan"

# The Qwen3.5 tool-call formatting rules every tool-bearing agent repeats.
XML_CALL_RULES = """\
Do not emit JSON inside <tool_call>. Use nested XML; do not put arguments as XML attributes.
Each call has one <function=tool_name> element and child <parameter=name>value</parameter> elements."""


def grounding(cwd: str | None = None) -> str:
    """The working-dir/platform line shared by system prompts.

    No path-style advice here: it depends on the agent's tools (bash prefers
    relative paths; the search tools REQUIRE absolute ones), so each agent
    says its own — a shared "use relative paths" once contradicted the
    explorer's own tool schemas.
    """
    if cwd is None:
        cwd = os.getcwd()
    return f"Working dir: {cwd}. Platform: {_platform_summary()}."


def compose_system_message(body: str, cwd: str | None = None) -> dict:
    """An agent's system message: its role text with grounding substituted."""
    return {"role": "system", "content": body.replace("<<GROUNDING>>", grounding(cwd))}


@dataclass(frozen=True)
class AgentSpec:
    """Static description of one agent/stage.

    ``session`` picks the GPU session: the editor and explorer run on scratch
    so their short-lived contexts never disturb a stage in progress on main.
    ``allowed_tools`` is both the wire toolset (what the stage advertises) and
    the execution allowlist — segregation, not just discipline.
    """

    name: str
    session: str
    allowed_tools: frozenset[str]
    # Mascot/status label; the TUI maps it onto a mascot state when one exists.
    mascot: str = "inference"
    # Post-bash-write feedback dialect: "hashline" (anchor ids +
    # read_file — only for agents holding the hashline tools, i.e. the
    # NORMAL agent), "subedit" (plain validation report naming the `edit`
    # delegator tool), "report" (validation report only, no edit-tool
    # reference). See TurnRunner.write_feedback.
    write_feedback: str = "report"
