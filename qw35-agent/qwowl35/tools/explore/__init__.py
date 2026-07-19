"""Explorer-stage tools: qwen-code-native filesystem search, ported to Python.

`list_directory`, `glob`, `grep_search`, and `inspect_file` replicate the tools
from QwenLM/qwen-code (pinned commit 7417805, packages/core/src/tools/
{ls,glob,grep,read-file}.ts): the wire names, parameter schemas, and result
formats match what the model was trained to drive, so the explorer stays
on-distribution. Only capabilities that need a cloud model or binary decoding
(PDF pages, images, notebooks) are answered with a plain error instead.
"""

from __future__ import annotations

from .glob_tool import GLOB_SCHEMA, run_glob
from .grep_tool import GREP_SCHEMA, run_grep
from .guidance import GUIDANCE
from .inspect_file import INSPECT_FILE_SCHEMA, run_inspect_file
from .ls import LS_SCHEMA, run_ls

try:  # per-agent dedup store; optional like the other diagnostics imports.
    from ..diagnostics import DiagnosticsMemory
except Exception:  # pragma: no cover - defensive fallback
    DiagnosticsMemory = None  # type: ignore[assignment]

# Wire names, verbatim from qwen-code's tool-names.ts — except `inspect_file`,
# which stands in for qwen-code's trained `read_file` name: that name summons
# its trained write-tool siblings in a read-only stage, see inspect_file.py.
LS_NAME = "list_directory"
GLOB_NAME = "glob"
GREP_NAME = "grep_search"
INSPECT_FILE_NAME = "inspect_file"

EXPLORE_TOOL_NAMES = frozenset({LS_NAME, GLOB_NAME, GREP_NAME, INSPECT_FILE_NAME})


class ExploreTools:
    """Dispatches the four explorer search tools. Synchronous (the registry
    runs execute() in a thread, like the other tools); the only state is
    ``diag_memory`` — which diagnostics the CURRENT agent instance has already
    been shown by inspect_file. The orchestrator repoints it at each agent's
    own store (a fresh explorer spawn must see everything once); standalone
    use gets a private store with the same behavior.
    """

    _RUNNERS = {
        LS_NAME: run_ls,
        GLOB_NAME: run_glob,
        GREP_NAME: run_grep,
    }

    def __init__(self) -> None:
        self.diag_memory = DiagnosticsMemory() if DiagnosticsMemory is not None else None

    def schemas(self) -> list[dict]:
        return [
            {"type": "function", "function": LS_SCHEMA},
            {"type": "function", "function": GLOB_SCHEMA},
            {"type": "function", "function": GREP_SCHEMA},
            {"type": "function", "function": INSPECT_FILE_SCHEMA},
        ]

    def execute(self, name: str, arguments: dict) -> str:
        try:
            if name == INSPECT_FILE_NAME:
                return run_inspect_file(arguments, self.diag_memory)
            runner = self._RUNNERS.get(name)
            if runner is None:
                return f"Error: unknown explore tool {name!r}."
            return runner(arguments)
        except Exception as exc:  # noqa: BLE001 - feed errors back to the model
            return f"Error: {exc}"


__all__ = [
    "ExploreTools",
    "EXPLORE_TOOL_NAMES",
    "GUIDANCE",
    "LS_NAME",
    "GLOB_NAME",
    "GREP_NAME",
    "INSPECT_FILE_NAME",
    "LS_SCHEMA",
    "GLOB_SCHEMA",
    "GREP_SCHEMA",
    "INSPECT_FILE_SCHEMA",
    "run_ls",
    "run_glob",
    "run_grep",
    "run_inspect_file",
]
