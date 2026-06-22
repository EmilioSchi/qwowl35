"""Wires the model-facing shell and hashline file tools.

Holds the session-scoped singletons, advertises their OpenAI schemas, and
dispatches calls off the event loop via ``asyncio.to_thread`` so a 60s bash
command or disk write never freezes the UI.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from tools.bash import GUIDANCE as BASH_GUIDANCE
from tools.bash import BashTool, build_bash_approval_options
from tools.base import ToolSpec
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
    ) -> None:
        self.bash = BashTool(restricted=restricted_bash)
        self.files = HashlineTools()
        self._approval = approval

    def set_approval(self, approval: ApprovalCallback) -> None:
        self._approval = approval

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
        args_error = _tool_args_error(name, arguments)
        if args_error is not None:
            return args_error
        if name == "bash":
            return await self._run_bash(arguments)
        return await asyncio.to_thread(self.files.execute, name, arguments)

    async def _run_bash(self, arguments: dict) -> str:
        command = arguments.get("command")
        if not isinstance(command, str) or not command:
            return "Error: 'command' is required."

        options = build_bash_approval_options(command)
        if options.warnings:
            if self._approval is None:
                return (
                    "Command denied (no approver configured). Flagged: "
                    + "; ".join(options.warnings)
                )
            decision = await self._approval(
                command, list(options.warnings), options.allowlist_info
            )
            if decision.kind == "deny":
                return "Command denied by user."
            if decision.kind == "alternative":
                return (
                    "Command not run. The user asked you to do this differently: "
                    + decision.text
                )
            # decision.kind == "accept" → fall through and run it.

        return await asyncio.to_thread(self.bash.execute, {"command": command})


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
