"""`grep_search` — qwen-code grep.ts replica (schema and output format)."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

# qwen-code's GrepTool caps llm output at 20k chars.
MAX_OUTPUT_CHARS = 20_000
_SKIP_DIRS = {".git", "node_modules", "__pycache__", "target", ".venv"}

GREP_SCHEMA = {
    "name": "grep_search",
    "description": (
        "A powerful search tool for finding patterns in files\n\n"
        "  Usage:\n"
        "  - ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a "
        "Bash command. The Grep tool has been optimized for correct permissions "
        "and access.\n"
        '  - Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")\n'
        '  - Filter files with glob parameter (e.g., "*.js", "**/*.tsx")\n'
        "  - Case-insensitive by default\n"
    ),
    "parameters": {
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "The regular expression pattern to search for in file contents"
                ),
            },
            "glob": {
                "type": "string",
                "description": (
                    'Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}")'
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "File or directory to search in. Defaults to current working "
                    "directory."
                ),
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Limit output to first N matching lines. Must be a positive "
                    "integer. Optional - shows all matches if not specified."
                ),
            },
            "compress": {
                "type": "boolean",
                "description": "Optional: false returns the full uncompressed output.",
            },
        },
        "required": ["pattern"],
        "type": "object",
    },
}


def _glob_to_regexes(pattern: str) -> list[re.Pattern]:
    """Expand qwen-style brace globs ("*.{ts,tsx}") into fnmatch regexes."""
    brace = re.fullmatch(r"(.*)\{([^}]*)\}(.*)", pattern)
    variants = (
        [f"{brace.group(1)}{alt}{brace.group(3)}" for alt in brace.group(2).split(",")]
        if brace
        else [pattern]
    )
    return [re.compile(fnmatch.translate(v)) for v in variants]


def _file_matches_filter(path: Path, root: Path, regexes: list[re.Pattern]) -> bool:
    if not regexes:
        return True
    rel = str(path.relative_to(root))
    name = path.name
    return any(rx.match(rel) or rx.match(name) for rx in regexes)


def run_grep(arguments: dict) -> str:
    pattern = arguments.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return "Error: 'pattern' is required."
    try:
        # qwen-code grep is case-insensitive by default.
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"Error: invalid regular expression {pattern!r}: {exc}"

    limit = arguments.get("limit")
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        return "Error: limit must be a positive integer"

    raw_path = arguments.get("path")
    if isinstance(raw_path, str) and raw_path.strip():
        root = Path(raw_path.strip())
        if not root.is_absolute():
            root = Path.cwd() / root
        location = f'in path "{raw_path.strip()}"'
    else:
        root = Path.cwd()
        location = "in the workspace directory"
    if not root.exists():
        return f"Error: search path does not exist: {root}"

    glob_filter = arguments.get("glob")
    filter_desc = ""
    filter_regexes: list[re.Pattern] = []
    if isinstance(glob_filter, str) and glob_filter.strip():
        glob_filter = glob_filter.strip()
        filter_desc = f' (filter: "{glob_filter}")'
        filter_regexes = _glob_to_regexes(glob_filter)

    if root.is_file():
        files = [root]
        search_root = root.parent
    else:
        search_root = root
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for filename in sorted(filenames):
                files.append(Path(dirpath) / filename)

    matches: dict[str, list[tuple[int, str]]] = {}
    total = 0
    capped = False
    for file in files:
        if not _file_matches_filter(file, search_root, filter_regexes):
            continue
        try:
            with open(file, "rb") as fh:
                head = fh.read(8192)
                if b"\x00" in head:
                    continue  # binary
                data = head + fh.read()
        except OSError:
            continue
        text = data.decode("utf-8", errors="replace")
        rel = str(file.relative_to(search_root)) if file != root else str(file)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.setdefault(rel, []).append((line_number, line.strip()))
                total += 1
                if limit is not None and total >= limit:
                    capped = True
                    break
        if capped:
            break

    if not matches:
        return f'No matches found for pattern "{pattern}" {location}{filter_desc}.'

    match_term = "match" if total == 1 else "matches"
    header = f'Found {total} {match_term} for pattern "{pattern}" {location}{filter_desc}:\n---\n'
    body = ""
    truncated = False
    for rel, hits in matches.items():
        chunk = f"File: {rel}\n"
        for line_number, line in hits:
            chunk += f"L{line_number}: {line}\n"
        chunk += "---\n"
        if len(header) + len(body) + len(chunk) > MAX_OUTPUT_CHARS:
            truncated = True
            break
        body += chunk

    result = (header + body).rstrip()
    if truncated:
        result += "\n\n(Output truncated at 20000 characters; narrow the search with path/glob/limit.)"
    elif capped:
        result += f"\n\n(Stopped at the first {limit} matching lines per the limit parameter.)"
    return result
