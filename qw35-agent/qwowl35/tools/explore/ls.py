"""`list_directory` — qwen-code ls.ts replica (schema and output format)."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

# Keep a directory with thousands of entries from flooding the context.
MAX_ENTRIES = 200

LS_SCHEMA = {
    "name": "list_directory",
    "description": (
        "Lists the names of files and subdirectories directly within a specified "
        "directory path. Can optionally ignore entries matching provided glob patterns."
    ),
    "parameters": {
        "properties": {
            "path": {
                "description": (
                    "The absolute path to the directory to list "
                    "(must be absolute, not relative)"
                ),
                "type": "string",
            },
            "ignore": {
                "description": "List of glob patterns to ignore",
                "items": {"type": "string"},
                "type": "array",
            },
            "file_filtering_options": {
                "description": (
                    "Optional: Whether to respect ignore patterns from .gitignore "
                    "when listing files"
                ),
                "type": "object",
                "properties": {
                    "respect_git_ignore": {
                        "description": (
                            "Optional: Whether to respect .gitignore patterns when "
                            "listing files. Only available in git repositories. "
                            "Defaults to true."
                        ),
                        "type": "boolean",
                    },
                },
            },
        },
        "required": ["path"],
        "type": "object",
    },
}


def run_ls(arguments: dict) -> str:
    raw_path = arguments.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return "Error: 'path' is required."
    raw_path = raw_path.strip()
    if not os.path.isabs(raw_path):
        return f"Error: Path must be absolute: {raw_path}"
    path = Path(raw_path)
    if not path.exists():
        return f"Error: Directory does not exist: {raw_path}"
    if not path.is_dir():
        return f"Error: Path is not a directory: {raw_path}"

    ignore = arguments.get("ignore")
    patterns = [p for p in ignore if isinstance(p, str)] if isinstance(ignore, list) else []

    entries = []
    try:
        for entry in path.iterdir():
            if any(fnmatch.fnmatch(entry.name, pattern) for pattern in patterns):
                continue
            entries.append((entry.name, entry.is_dir()))
    except PermissionError as exc:
        return f"Error: {exc}"

    if not entries:
        return f"Directory {raw_path} is empty."

    # qwen-code order: directories first, then alphabetical within each group.
    entries.sort(key=lambda item: (not item[1], item[0].lower()))
    total = len(entries)
    shown = entries[:MAX_ENTRIES]
    listing = "\n".join(f"{'[DIR] ' if is_dir else ''}{name}" for name, is_dir in shown)
    message = f"Listed {total} item(s) in {raw_path}:\n---\n{listing}"
    if total > len(shown):
        message += f"\n\n({total - len(shown)} more entries not shown)"
    return message
