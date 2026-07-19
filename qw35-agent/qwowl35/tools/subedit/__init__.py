"""Freestyle's `edit` tool: delegate a file change to the Editor sub-agent.

The freestyle agent has exactly two tools — bash and this one. An `edit` call
names the file, the line ranges involved, and what to change; the orchestrator
intercepts it, spawns the Editor agent (hashline replace/insert/delete on the
scratch session) against that file slice, and returns the Editor's summary +
diff as this call's tool result. This module owns the schema and guidance; the
routing lives in the orchestrator.
"""

from __future__ import annotations

EDIT_NAME = "edit"

EDIT_SCHEMA = {
    "name": "edit",
    "description": (
        "Delegate a change to an existing file. Name the file, the line range(s) "
        "involved (as printed by `grep -n`), and describe precisely what must "
        "change and why. A dedicated editor applies the change and returns a "
        "summary with the diff. Use bash heredocs only to CREATE new files; "
        "every change to an existing file goes through this tool."
    ),
    "parameters": {
        "properties": {
            "filename": {
                "type": "string",
                "description": "Path of the file to modify.",
            },
            "line_ranges": {
                "type": "string",
                "description": (
                    'The line range(s) to modify, e.g. "12-18" or "40-45, 102" '
                    "(from `grep -n` / `grep -nr` output). Use \"all\" only when "
                    "the whole file is genuinely in scope."
                ),
            },
            "instructions": {
                "type": "string",
                "description": (
                    "What to change and why: the exact edits to make, expected "
                    "behavior, and any constraints (naming, style, invariants)."
                ),
            },
        },
        "required": ["filename", "line_ranges", "instructions"],
        "type": "object",
    },
}

GUIDANCE = """\
To change an existing file, call `edit` with the filename, the line range(s)
(find them with `grep -n pattern file`), and precise instructions for the
change. Do not rewrite existing files through bash redirects — `edit` applies
targeted changes and shows you the diff. Create brand-new files with a bash
heredoc, then use `edit` for every later adjustment."""


def validate_edit_args(arguments: dict) -> str | None:
    for key in ("filename", "line_ranges", "instructions"):
        value = arguments.get(key) if isinstance(arguments, dict) else None
        if not isinstance(value, str) or not value.strip():
            return f"Error: '{key}' is required."
    return None


__all__ = ["EDIT_NAME", "EDIT_SCHEMA", "GUIDANCE", "validate_edit_args"]
