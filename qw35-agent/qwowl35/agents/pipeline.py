"""PipelineRegistry: one executor for every pipeline tool, segregated per stage.

Each stage advertises and may call ONLY its own toolset: `schemas()` returns
the ACTIVE stage's tools (the wire `tools` array the server renders), and
`execute()` re-checks the same allowlist. Stages run as fresh contexts, so
changing the advertised tools between stages costs nothing — segregation is
the constraint, not prefix reuse.

The two sub-agent tools are not executed here: the orchestrator plugs in
callbacks that run the editor (the executor's `edit`) and the explorer (the
planner's `explore`) and return their reports. The explorer's own toolset
lives in :class:`ExplorerRegistry`, built per spawn.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from tools.bash import SHELL_NAME, BashTool
from tools.compress import compress_requested, compress_tool_result, strip_compress_arg
from tools.explore import (
    EXPLORE_TOOL_NAMES,
    GLOB_SCHEMA,
    GREP_SCHEMA,
    INSPECT_FILE_SCHEMA,
    LS_SCHEMA,
    ExploreTools,
)
from tools.lsp import LSP_NAME, LSP_SCHEMA, LspQueryTool
from tools.plan import PLAN_TOOL_NAMES, PlanTools
from tools.subedit import EDIT_NAME, EDIT_SCHEMA, validate_edit_args
from tools.web import (
    SEARCH_ENGINE_NAME,
    SEARCH_ENGINE_SCHEMA,
    SearchEngineTool,
    WEB_FETCH_NAME,
    WEB_FETCH_SCHEMA,
    WebFetchTool,
)
from tools_registry import ApprovalCallback, _tool_args_error, run_bash_with_approval

from . import explorer as explorer_agent

# Sub-agent callbacks: the orchestrator runs the editor/explorer and returns
# their reports as the tool result.
EditorCallback = Callable[[dict], Awaitable[str]]
ExplorerCallback = Callable[[dict], Awaitable[str]]


class PipelineRegistry:
    """Executor + per-stage wire toolsets for web/planner/execute/chat."""

    def __init__(
        self,
        approval: ApprovalCallback | None = None,
        restricted_bash: bool = False,
        compress: bool = True,
        rerank: bool = True,
        question_callback=None,
        plan_callback=None,
    ) -> None:
        self._compress = compress
        self._rerank = rerank
        # Cumulative observability counters over every compressible result.
        self.compress_original_chars = 0
        self.compress_saved_chars = 0
        self.bash = BashTool(restricted=restricted_bash)
        # The explorer's shell: same wire tool, but under the hood it runs
        # `bash -r` (no cd, no redirection, no /-qualified commands) — the
        # explorer can observe (run a program, check a version) but not
        # write. Its prompt never mentions the restriction.
        self.bash_restricted = BashTool(restricted=True)
        self.explore = ExploreTools()
        # Shared LSP navigation engine (explorer + editor sub-agents); the
        # underlying language servers are module-global in tools.lsp anyway.
        self.lsp = LspQueryTool()
        self.web = WebFetchTool()
        self.search = SearchEngineTool()
        self.plan = PlanTools(
            question_callback=question_callback, plan_callback=plan_callback
        )
        self._approval = approval
        self._editor: EditorCallback | None = None
        self._explorer: ExplorerCallback | None = None
        # Shared hashline engine, assigned by the orchestrator (it owns the
        # instance the editor sub-agent edits through). Never dispatched from
        # here — the execute stage has no hashline wire tools, and its
        # post-bash-write feedback is the plain validation report
        # (TurnRunner.write_feedback = "subedit"), not hashline anchors.
        self.files = None
        # The ACTIVE agent instance's diagnostics memory (which rows it has
        # already been shown). The orchestrator repoints it at every agent
        # seam via set_diag_memory; until then, adopt the explore engine's
        # own store so standalone use still dedups coherently.
        self.diag_memory = self.explore.diag_memory
        self.allowed: frozenset[str] | None = None
        # Wire toolsets per stage: exactly the stage's own tools, nothing else.
        self._stage_schemas: dict[str, list[dict]] = {
            "web": [
                {"type": "function", "function": SEARCH_ENGINE_SCHEMA},
                {"type": "function", "function": WEB_FETCH_SCHEMA},
            ],
            "planner": self.plan.schemas()
            + [{"type": "function", "function": explorer_agent.EXPLORE_SCHEMA}],
            "execute": [
                {"type": "function", "function": self.bash.schema()},
                {"type": "function", "function": EDIT_SCHEMA},
            ],
            "chat": [],
        }
        self._active_stage: str | None = None

    def set_approval(self, approval: ApprovalCallback) -> None:
        self._approval = approval

    def set_editor(self, editor: EditorCallback) -> None:
        self._editor = editor

    def set_explorer(self, explorer: ExplorerCallback) -> None:
        self._explorer = explorer

    def set_stage(self, name: str, allowed: frozenset[str] | None) -> None:
        """Activate a stage: its wire toolset and execution allowlist together."""
        self._active_stage = name
        self.allowed = allowed

    def set_diag_memory(self, memory) -> None:
        """Point every diagnostics-emitting engine at the RUNNING agent
        instance's memory. Engines are session-scoped singletons while the
        dedup contract is per agent instance, so the orchestrator calls this
        at each seam: stage begin (executor context, direct turn) and
        sub-agent spawn/return (fresh editor/explorer must see everything
        once, then the parent's store is restored)."""
        self.diag_memory = memory
        self.explore.diag_memory = memory
        if self.files is not None:
            self.files.diag_memory = memory

    def schemas(self) -> list[dict]:
        if self._active_stage is None:
            return []
        return self._stage_schemas.get(self._active_stage, [])

    async def execute(self, name: str, arguments: dict) -> str:
        if self.allowed is not None and name not in self.allowed:
            tools = ", ".join(sorted(self.allowed)) or "(none)"
            return (
                f"The tool `{name}` is not available in this stage. "
                f"Tools available right now: {tools}. Continue with one of those."
            )
        args_error = _tool_args_error(name, arguments)
        if args_error is not None:
            return args_error
        # `compress` is a registry-level arg: strip it so executors never see
        # it. Non-mutating — the caller reuses `arguments` (dedup signature)
        # after this returns.
        call_args = strip_compress_arg(arguments)
        result = await self._dispatch(name, call_args)
        if self._compress and compress_requested(arguments):
            compressed = compress_tool_result(name, call_args, result, rerank=self._rerank)
            self.compress_original_chars += compressed.original_chars
            self.compress_saved_chars += compressed.saved_chars
            result = compressed.text
        return result

    async def _dispatch(self, name: str, arguments: dict) -> str:
        if name in ("bash", SHELL_NAME):
            return await run_bash_with_approval(self.bash, self._approval, arguments)
        if name == EDIT_NAME:
            error = validate_edit_args(arguments)
            if error is not None:
                return error
            if self._editor is None:
                return "Error: no editor is wired for this session."
            return await self._editor(arguments)
        if name == explorer_agent.EXPLORE_NAME:
            if self._explorer is None:
                return "Error: no explorer is wired for this session."
            return await self._explorer(arguments)
        if name in EXPLORE_TOOL_NAMES:
            return await asyncio.to_thread(self.explore.execute, name, arguments)
        if name == WEB_FETCH_NAME:
            return await asyncio.to_thread(self.web.execute, arguments)
        if name == SEARCH_ENGINE_NAME:
            return await asyncio.to_thread(self.search.execute, arguments)
        if name in PLAN_TOOL_NAMES:
            return await self.plan.execute(name, arguments)
        return f"Error: unknown tool {name!r}."


class ExplorerRegistry:
    """The explorer sub-agent's toolset: the four search tools, the restricted
    shell, and `resume`. Built fresh per spawn; captures the `resume` summary
    so the orchestrator can hand it back as the planner's `explore` result.

    Shares the parent :class:`PipelineRegistry`'s tool engines (search cache,
    restricted shell, approval callback, compression counters) — only the
    dispatch surface is the explorer's own.
    """

    def __init__(self, pipeline: PipelineRegistry) -> None:
        self.pipeline = pipeline
        self.resume_summary: str | None = None

    @property
    def diag_memory(self):
        """The active diagnostics memory (the parent engines'), so runner
        paths that read ``registry.diag_memory`` see the explorer's store."""
        return getattr(self.pipeline.explore, "diag_memory", None)

    def schemas(self) -> list[dict]:
        return [
            {"type": "function", "function": LS_SCHEMA},
            {"type": "function", "function": GLOB_SCHEMA},
            {"type": "function", "function": GREP_SCHEMA},
            {"type": "function", "function": INSPECT_FILE_SCHEMA},
            {"type": "function", "function": LSP_SCHEMA},
            {"type": "function", "function": self.pipeline.bash_restricted.schema()},
            {"type": "function", "function": explorer_agent.RESUME_SCHEMA},
        ]

    async def execute(self, name: str, arguments: dict) -> str:
        if name == explorer_agent.RESUME_NAME:
            summary = ""
            if isinstance(arguments, dict):
                summary = str(arguments.get("summary", "")).strip()
            if not summary:
                return (
                    "Error: `resume` needs a non-empty `summary` carrying the "
                    "complete findings report."
                )
            self.resume_summary = summary
            return "Findings recorded; exploration complete."
        args_error = _tool_args_error(name, arguments)
        if args_error is not None:
            return args_error
        call_args = strip_compress_arg(arguments)
        pipeline = self.pipeline
        if name in ("bash", SHELL_NAME):
            result = await run_bash_with_approval(
                pipeline.bash_restricted, pipeline._approval, call_args
            )
        elif name in EXPLORE_TOOL_NAMES:
            result = await asyncio.to_thread(pipeline.explore.execute, name, call_args)
        elif name == LSP_NAME:
            result = await asyncio.to_thread(pipeline.lsp.execute, call_args)
        else:
            return f"Error: unknown tool {name!r}."
        if pipeline._compress and compress_requested(arguments):
            compressed = compress_tool_result(
                name, call_args, result, rerank=pipeline._rerank
            )
            pipeline.compress_original_chars += compressed.original_chars
            pipeline.compress_saved_chars += compressed.saved_chars
            result = compressed.text
        return result
