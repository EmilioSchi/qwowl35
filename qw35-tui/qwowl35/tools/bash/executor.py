from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

from .analyzer import build_bash_approval_options

# Maximum execution time for a single command, in seconds.
BASH_TIMEOUT_SECONDS = 60
# Maximum captured output per stream, in bytes.
MAX_OUTPUT_SIZE = 50_000


@dataclass
class CappedBuffer:
    """Accumulates output up to a byte limit, recording when truncation occurs."""

    limit: int
    _data: bytearray = None  # type: ignore[assignment]
    truncated: bool = False

    def __post_init__(self) -> None:
        self._data = bytearray()

    def write(self, chunk: bytes) -> None:
        if self.limit <= 0:
            if chunk:
                self.truncated = True
            return
        remaining = self.limit - len(self._data)
        if remaining <= 0:
            if chunk:
                self.truncated = True
            return
        if len(chunk) > remaining:
            self._data.extend(chunk[:remaining])
            self.truncated = True
            return
        self._data.extend(chunk)

    def __len__(self) -> int:
        return len(self._data)

    def render(self, marker: str) -> str:
        output = self._data.decode("utf8", errors="replace")
        if self.truncated:
            if output and not output.endswith("\n"):
                output += "\n"
            output += marker
        return output


def _format_output(stdout: CappedBuffer, stderr: CappedBuffer) -> str:
    parts: list[str] = []
    if len(stdout) > 0:
        parts.append(stdout.render("... (output truncated)"))
    if len(stderr) > 0:
        prefix = "\n" if parts else ""
        parts.append(prefix + "stderr:\n" + stderr.render("... (stderr truncated)"))
    return "".join(parts)


class BashTool:
    """Executes a bash command, capping output and surfacing exit status.

    Suspicious-command analysis is available via :func:`approval_options` and the
    :mod:`tools.bash.analyzer` helpers; it is intended to gate execution behind an
    approval prompt, not to block it outright.
    """

    name = "bash"
    description = (
        "Execute a bash command on the system. Use this to run shell commands, "
        "inspect files, create new files, run programs, and verify changes. "
        "For existing non-empty files, read anchors and use the file edit tools "
        "instead of rewriting the whole file from bash."
    )

    def __init__(self, restricted: bool = False) -> None:
        # When restricted, commands run under ``bash -r`` (restricted shell):
        # no ``cd``, no output redirection, no ``/``-qualified command paths, and
        # no ``PATH``/``SHELL`` reassignment. Used by unattended debug runs.
        self._restricted = restricted

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    },
                },
                "required": ["command"],
            },
        }

    def approval_options(self, command: str):
        """Warnings and allowlist context for prompting the user before running."""
        return build_bash_approval_options(command)

    def execute(self, args: dict[str, Any]) -> str:
        command = args.get("command")
        if not isinstance(command, str) or command == "":
            raise ValueError("command parameter is required")

        stdout = CappedBuffer(limit=MAX_OUTPUT_SIZE)
        stderr = CappedBuffer(limit=MAX_OUTPUT_SIZE)

        argv = ["bash", "-r", "-c", command] if self._restricted else ["bash", "-c", command]
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                timeout=BASH_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            stdout.write(exc.stdout or b"")
            stderr.write(exc.stderr or b"")
            rendered = _format_output(stdout, stderr)
            return rendered + "\n\nError: command timed out after 60 seconds"

        stdout.write(completed.stdout or b"")
        stderr.write(completed.stderr or b"")
        rendered = _format_output(stdout, stderr)

        if completed.returncode != 0:
            return rendered + f"\n\nExit code: {completed.returncode}"
        if not rendered:
            return "(no output)"
        return rendered
