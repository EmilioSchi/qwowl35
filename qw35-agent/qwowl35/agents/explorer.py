"""The explorer: a stateless read-only sub-agent the planner spawns on demand.

The planner calls the `explore` tool with a detailed task description; the
orchestrator spawns a FRESH explorer (zero conversation history, scratch GPU
session) that searches with the four qwen-code search tools plus a restricted
shell, then finishes with exactly one `resume` call whose `summary` is the
complete findings report. That summary — nothing else from the explorer's
context — is returned to the planner as the `explore` call's result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from agent import BudgetDecision
from tools.bash import SHELL_NAME
from tools.explore import GUIDANCE as EXPLORE_GUIDANCE
from tools.explore import GLOB_NAME, GREP_NAME, INSPECT_FILE_NAME, LS_NAME
from tools.lsp import LSP_GUIDANCE, LSP_NAME

from .base import SESSION_SCRATCH, XML_CALL_RULES, AgentSpec, compose_system_message

# The explorer's closing tool: one comprehensive findings report ends the run.
RESUME_NAME = "resume"
RESUME_SCHEMA = {
    "name": RESUME_NAME,
    "description": (
        "Finish the exploration: report everything you found in one "
        "comprehensive summary. Call this exactly once, when you are done "
        "searching — only this summary reaches the agent that requested the "
        "exploration, so make it self-contained."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "The complete findings report: file paths, key line "
                    "references, verbatim snippets that matter, how the "
                    "pieces connect, and open questions."
                ),
            },
        },
        "required": ["summary"],
    },
}

# The planner-side tool that spawns an explorer.
EXPLORE_NAME = "explore"
EXPLORE_SCHEMA = {
    "name": EXPLORE_NAME,
    "description": (
        "Spawn a read-only explorer sub-agent on the codebase. It searches, "
        "reads files, and runs observation commands, then returns a findings "
        "summary as this call's result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Detailed exploration task: what to find, where to start, "
                    "and what the findings report must cover."
                ),
            },
        },
        "required": ["task"],
    },
}

# The explorer opens files as `inspect_file`, not qwen-code's trained
# `read_file` name: that name drags its write-tool siblings into a stage
# that has none.
# `run_shell_command` is available for observation (running a program to see
# its output) — under the hood it is a RESTRICTED shell (bash -r: no cd, no
# redirection), which the prompt deliberately does not mention.
# Stateless and short-lived, so it rides the scratch GPU session like the
# editor: its context must never disturb the planner's plan-session lineage
# or the executors' main-session checkpoints.
SPEC = AgentSpec(
    name="explore",
    session=SESSION_SCRATCH,
    allowed_tools=frozenset(
        {LS_NAME, GLOB_NAME, GREP_NAME, INSPECT_FILE_NAME, LSP_NAME, SHELL_NAME, RESUME_NAME}
    ),
    mascot="explore",
)

SYSTEM_PROMPT = f"""\
You are the qwowl35 explorer. You map the project code and behavior a task
touches and report what you find; acting on it is the requesting agent's job.

Your role is EXCLUSIVELY to search and analyze existing code. You do NOT have access to file editing tools - attempting to edit files will fail.

{EXPLORE_GUIDANCE}
{LSP_GUIDANCE}
- `run_shell_command` runs a shell command when observing behavior requires it
  (e.g. running a program to see its output). `is_background` is required;
  use false.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Avoid using run_shell_command with the `find`, `grep`, `cat`, `head`, `tail`,
`sed`, `awk`, or `echo` commands, unless explicitly instructed or when these
commands are truly necessary for the task. Instead, always prefer using the
dedicated tools for these commands:
- File search: Use glob (NOT find or ls)
- Content search: Use grep_search (NOT grep or rg)
- Read files: Use inspect_file (NOT cat/head/tail)

Work freely with the search tools until the task is answered. Then finish
with exactly ONE `resume` call: its `summary` is the complete findings report
(file paths, key line references, the verbatim snippets that matter, how the
pieces connect, and open questions). ONLY that summary reaches the agent that
requested this exploration — nothing else you saw or wrote survives — so make
it self-contained.
<<GROUNDING>>

{XML_CALL_RULES}

NOTE: You are meant to be a fast agent that returns output as quickly as possible.
Complete the search task efficiently and report your findings clearly."""

# Exploration budget in model round-trips (streams), scaled by the session's
# reasoning effort; `TurnRunner.max_rounds` ends the run cleanly when spent
# and the orchestrator falls back to the explorer's last notes.
EXPLORE_EFFORT_BUDGET_ROUNDS = {
    "low": 6,
    "medium": 10,
    "high": 16,
    "xhigh": 24,
}
DEFAULT_EXPLORE_BUDGET_ROUNDS = EXPLORE_EFFORT_BUDGET_ROUNDS["medium"]


def effort_rounds(reasoning_effort: str | None) -> int:
    return EXPLORE_EFFORT_BUDGET_ROUNDS.get(
        (reasoning_effort or "").lower(), DEFAULT_EXPLORE_BUDGET_ROUNDS
    )


def next_tier(reasoning_effort: str | None) -> tuple[str, int] | None:
    """The next effort tier above `reasoning_effort`, as (name, rounds).

    None once already at the ceiling (xhigh) — the "grow" choice in the
    round-budget-reached modal has nowhere left to go.
    """
    tiers = list(EXPLORE_EFFORT_BUDGET_ROUNDS.items())
    current = (reasoning_effort or "").lower()
    names = [name for name, _ in tiers]
    index = names.index(current) if current in names else names.index("medium")
    if index + 1 >= len(tiers):
        return None
    return tiers[index + 1]


@dataclass(frozen=True)
class ExplorerBudgetContext:
    """Presented to the user when a running explorer's round budget runs
    out, so they can choose what happens next instead of a silent cutoff."""

    task: str
    effort: str
    max_rounds: int
    next_tier: tuple[str, int] | None
    notes_preview: str


# Bound at Orchestrator construction (same convention as tools_registry's
# ApprovalCallback / tools/plan/tools's QuestionCallback): receives the
# context above and returns what to do. Unset (None) reproduces today's
# silent stop-on-budget behavior.
ExplorerBudgetCallback = Callable[[ExplorerBudgetContext], Awaitable[BudgetDecision]]


def system_message(cwd: str | None = None) -> dict:
    return compose_system_message(SYSTEM_PROMPT, cwd)


def build_task_message(task: str) -> dict:
    """The planner's task description, verbatim — the explorer is stateless
    by design and receives no session history."""
    return {"role": "user", "content": task}
