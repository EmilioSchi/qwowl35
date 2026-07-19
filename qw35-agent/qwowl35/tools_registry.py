"""Wires the model-facing shell and hashline file tools.

Holds the session-scoped singletons, advertises their OpenAI schemas, and
dispatches calls off the event loop via ``asyncio.to_thread`` so a 60s bash
command or disk write never freezes the UI.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from tools.bash import GUIDANCE as BASH_GUIDANCE
from tools.bash import SHELL_NAME, BashTool, build_bash_approval_options
from tools.base import ToolSpec
from tools.compress import compress_requested, compress_tool_result, strip_compress_arg
from tools.files import GUIDANCE as FILES_GUIDANCE
from tools.files import HashlineTools

from approval import ApprovalDecision

# Called when a bash command is flagged. Receives (command, warnings,
# allowlist_info) and returns an ApprovalDecision. If unset, flagged commands
# are denied by default (fail safe).
ApprovalCallback = Callable[[str, list[str], str], Awaitable[ApprovalDecision]]


class ToolRegistry:
    def __init__(
        self,
        approval: ApprovalCallback | None = None,
        restricted_bash: bool = False,
        compress: bool = True,
    ) -> None:
        self.bash = BashTool(restricted=restricted_bash)
        self.files = HashlineTools()
        # The freestyle agent is ONE persistent context, so one diagnostics
        # memory serves its whole conversation: adopt the hashline engine's
        # store so the post-write report and the file tools dedup against the
        # same seen-set. Cleared with the conversation guards on /clear.
        self.diag_memory = self.files.diag_memory
        self._approval = approval
        self._compress = compress
        # Cumulative observability counters over every compressible result.
        self.compress_original_chars = 0
        self.compress_saved_chars = 0
        # Stage discipline (smart mode): when set, execute() denies any tool
        # outside this set. Belt-and-braces behind the TurnRunner dispatch
        # check, so paths that call the registry directly stay honest too.
        # None (default, freestyle) allows every registered tool.
        self.allowed: frozenset[str] | None = None

    def set_approval(self, approval: ApprovalCallback) -> None:
        self._approval = approval

    def set_allowed(self, allowed: frozenset[str] | None) -> None:
        """Restrict (or with ``None`` unrestrict) the executable tool set."""
        self.allowed = allowed

    def _specs(self) -> list[ToolSpec]:
        """Registered tools in dispatch order, each bundling schema(s) + guidance.

        One ordered source feeds both ``schemas()`` (the wire ``tools`` array)
        and ``guidance_sections()`` (the per-tool system-prompt text), so adding
        a tool here updates both without touching the prompt.
        """
        return [
            ToolSpec(schemas=self.files.schemas(), guidance=FILES_GUIDANCE),
            ToolSpec(
                schemas=[{"type": "function", "function": self.bash.schema()}],
                guidance=BASH_GUIDANCE,
            ),
        ]

    def schemas(self) -> list[dict]:
        """Tools in OpenAI function format (bare schemas wrapped)."""
        return [schema for spec in self._specs() for schema in spec.schemas]

    def guidance_sections(self) -> list[str]:
        """Per-tool system-prompt guidance, in the same order as ``schemas()``."""
        return [spec.guidance for spec in self._specs()]

    async def execute(self, name: str, arguments: dict) -> str:
        if self.allowed is not None and name not in self.allowed:
            tools = ", ".join(sorted(self.allowed))
            return (
                f"The tool `{name}` is not available in this stage. "
                f"Tools available right now: {tools}. Continue with one of those."
            )
        args_error = _tool_args_error(name, arguments)
        if args_error is not None:
            return args_error
        # `compress` is a registry-level arg: strip it so executors never see
        # it. Non-mutating — the caller reuses `arguments` (dedup signature,
        # call descriptions) after this returns.
        call_args = strip_compress_arg(arguments)
        # "bash" stays accepted as a legacy alias of the trained wire name.
        if name in ("bash", SHELL_NAME):
            result = await self._run_bash(call_args)
        else:
            result = await asyncio.to_thread(self.files.execute, name, call_args)
        if self._compress and compress_requested(arguments):
            compressed = compress_tool_result(name, call_args, result)
            self.compress_original_chars += compressed.original_chars
            self.compress_saved_chars += compressed.saved_chars
            result = compressed.text
        return result

    async def _run_bash(self, arguments: dict) -> str:
        return await run_bash_with_approval(self.bash, self._approval, arguments)


async def run_bash_with_approval(
    bash: BashTool,
    approval: ApprovalCallback | None,
    arguments: dict,
) -> str:
    """Approval-gated bash execution, shared by every registry that offers
    bash (the freestyle ToolRegistry and smart mode's pipeline registry)."""
    command = arguments.get("command")
    if not isinstance(command, str) or not command:
        return "Error: 'command' is required."

    options = build_bash_approval_options(command)
    if options.warnings:
        if approval is None:
            return (
                "Command denied (no approver configured). Flagged: "
                + "; ".join(options.warnings)
            )
        decision = await approval(command, list(options.warnings), options.allowlist_info)
        if decision.kind == "deny":
            return "Command denied by user."
        if decision.kind == "alternative":
            return (
                "Command not run. The user asked you to do this differently: "
                + decision.text
            )
        # decision.kind == "accept" → fall through and run it.

    return await asyncio.to_thread(
        bash.execute,
        {"command": command, "is_background": arguments.get("is_background")},
    )


def _tool_args_error(name: str, arguments: dict) -> str | None:
    if not isinstance(arguments, dict):
        return (
            f"Error: Your {name} call's arguments were not a valid JSON object. "
            "Resend exactly one JSON object."
        )
    if arguments.get("_invalid_json") is not True:
        return None
    message = (
        f"Error: Your {name} call's arguments were not a valid JSON object. "
        "Resend exactly one JSON object."
    )
    detail = arguments.get("_json_error")
    if isinstance(detail, str) and detail:
        message += f" Details: {detail}."
    return message
