"""`glob` — qwen-code glob.ts replica (schema and output format)."""

from __future__ import annotations

import os
from pathlib import Path

MAX_FILES = 100

GLOB_SCHEMA = {
    "name": "glob",
    "description": (
        "Fast file pattern matching tool that works with any codebase size\n"
        '- Supports glob patterns like "**/*.js" or "src/**/*.ts"\n'
        "- Returns matching file paths sorted by modification time\n"
        "- Use this tool when you need to find files by name patterns\n"
        "- You have the capability to call multiple tools in a single response. "
        "It is always better to speculatively perform multiple searches as a batch "
        "that are potentially useful."
    ),
    "parameters": {
        "properties": {
            "pattern": {
                "description": "The glob pattern to match files against",
                "type": "string",
            },
            "path": {
                "description": (
                    "The directory to search in. If not specified, the current "
                    "working directory will be used. IMPORTANT: Omit this field to "
                    'use the default directory. DO NOT enter "undefined" or "null" '
                    "- simply omit it for the default behavior. Must be a valid "
                    "directory path if provided."
                ),
                "type": "string",
            },
        },
        "required": ["pattern"],
        "type": "object",
    },
}


def run_glob(arguments: dict) -> str:
    pattern = arguments.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        return "Error: The 'pattern' parameter cannot be empty."
    pattern = pattern.strip()

    raw_path = arguments.get("path")
    if isinstance(raw_path, str) and raw_path.strip() and raw_path.strip().lower() not in (
        "undefined",
        "null",
    ):
        base = Path(raw_path.strip())
        if not base.is_absolute():
            base = Path.cwd() / base
    else:
        base = Path.cwd()
    if not base.is_dir():
        return f"Error: search path is not a directory: {base}"

    location = f"within {base}"
    try:
        matches = [p for p in base.glob(pattern) if p.is_file() and ".git" not in p.parts]
    except (ValueError, NotImplementedError) as exc:
        return f"Error: invalid glob pattern {pattern!r}: {exc}"

    if not matches:
        return f'No files found matching pattern "{pattern}" {location}'

    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    matches.sort(key=mtime, reverse=True)
    total = len(matches)
    shown = matches[:MAX_FILES]
    listing = "\n".join(str(p if p.is_absolute() else (base / p)) for p in shown)
    message = (
        f'Found {total} file(s) matching "{pattern}" {location}, '
        f"sorted by modification time (newest first):\n---\n{listing}"
    )
    if total > len(shown):
        omitted = total - len(shown)
        file_term = "file" if omitted == 1 else "files"
        message += f"\n\n(Results truncated: {omitted} more {file_term} matched)"
    return message


__all__ = ["GLOB_SCHEMA", "run_glob", "MAX_FILES"]
