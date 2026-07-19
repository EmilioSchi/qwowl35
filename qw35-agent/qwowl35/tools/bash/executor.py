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


# Wire name, verbatim from qwen-code's tool-names.ts (SHELL). The tool was
# once advertised as "bash"; the model — trained on qwen-code — kept reaching
# for `run_shell_command` anyway, so the trained name is the real one.
SHELL_NAME = "run_shell_command"


class BashTool:
    """Executes a shell command, capping output and surfacing exit status.

    Suspicious-command analysis is available via :func:`approval_options` and the
    :mod:`tools.bash.analyzer` helpers; it is intended to gate execution behind an
    approval prompt, not to block it outright.
    """

    name = SHELL_NAME
    description = (
        "Executes a given shell command (as `bash -c <command>`) in a "
        "subprocess, ensuring proper handling and security measures."
    )

    def __init__(self, restricted: bool = False) -> None:
        # When restricted, commands run under ``bash -r`` (restricted shell):
        # no ``cd``, no output redirection, no ``/``-qualified command paths, and
        # no ``PATH``/``SHELL`` reassignment. Used by unattended debug runs and
        # the explorer's under-the-hood shell.
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
                        "description": "The exact shell command to execute.",
                    },
                    "is_background": {
                        "type": "boolean",
                        "description": (
                            "Whether to run the command in background. This "
                            "parameter is required to ensure explicit "
                            "decision-making about command execution mode. Set "
                            "to true for long-running processes like "
                            "development servers, watchers, or daemons that "
                            "should continue running without blocking further "
                            "commands. Set to false for one-time commands that "
                            "should complete before proceeding."
                        ),
                    },
                    "compress": {
                        "type": "boolean",
                        "description": (
                            "Optional: false returns the full uncompressed output."
                        ),
                    },
                },
                "required": ["command", "is_background"],
            },
        }

    def approval_options(self, command: str):
        """Warnings and allowlist context for prompting the user before running."""
        return build_bash_approval_options(command)

    def execute(self, args: dict[str, Any]) -> str:
        command = args.get("command")
        if not isinstance(command, str) or command == "":
            raise ValueError("command parameter is required")
        # The schema requires an explicit choice; a compliant model always
        # sends it. Runtime tolerance: a missing flag runs foreground, and the
        # XML tool-call path delivers booleans as strings ("true"/"false").
        raw_background = args.get("is_background")
        is_background = raw_background is True or (
            isinstance(raw_background, str) and raw_background.strip().lower() == "true"
        )

        argv = ["bash", "-r", "-c", command] if self._restricted else ["bash", "-c", command]

        if is_background:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            return (
                f"Command running in background (PID {process.pid}). Its output "
                "is not captured; check its effects (files, ports, logs) with "
                "follow-up commands."
            )

        stdout = CappedBuffer(limit=MAX_OUTPUT_SIZE)
        stderr = CappedBuffer(limit=MAX_OUTPUT_SIZE)
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
