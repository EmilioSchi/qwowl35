"""`grep_search`, the editor's dialect: regex search over ONE file, id rows.

The explorer's grep_search (tools/explore/grep_tool.py) walks trees, filters
globs, and takes limit/compress knobs. The editor needs none of that — it
holds one file and speaks hashline — so its registry advertises this basic
variant under the SAME wire name: pattern + path (a single file), every
matching line returned as a ``<line><hash>|content`` row that
edit/insert/delete/lsp accept directly. Output never passes the compression
layer (the editor registry dispatches it directly); only a raw size guard
protects the context from pathological patterns.

Design rules match tools/lsp/query.py: never raise, workspace-rooted paths,
disk is truth (the hashline edit tools flush every mutation atomically).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .document import Document
from .output import line_view

GREP_FILE_NAME = "grep_search"

# Raw-size safety net (deliberately no `limit` arg): rows beyond this are
# dropped with an honest note. Generous — a sane single-file search never
# gets close.
MAX_OUTPUT_CHARS = 20_000

# Captured at import, same idiom as tools/lsp/query.py: the agent's
# workspace root never changes after launch.
_WORKSPACE_ROOT = os.getcwd()

GREP_FILE_SCHEMA = {
    "name": GREP_FILE_NAME,
    "description": (
        "Search one file for a regular expression (case-insensitive by "
        "default). Every matching line comes back as a line id row "
        "(12af|content) you can address directly with edit/insert/delete/"
        "lsp. Read-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "The regular expression pattern to search for in the file"
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "The file to search (absolute or workspace-relative)."
                ),
            },
        },
        "required": ["pattern", "path"],
    },
}


def run_file_grep(arguments: dict) -> str:
    try:
        return _run(arguments)
    except Exception as exc:  # noqa: BLE001 - never raise into a registry
        return f"Error: grep_search failed ({exc})."


def _run(args: dict) -> str:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return "Error: 'pattern' is required."
    try:
        # Case-insensitive by default, same as the explorer's grep_search.
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"Error: invalid regular expression {pattern!r}: {exc}"

    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return "Error: 'path' is required (one file)."
    raw_path = raw_path.strip()
    root = os.path.realpath(_WORKSPACE_ROOT)
    abs_path = os.path.realpath(
        raw_path if os.path.isabs(raw_path) else os.path.join(root, raw_path)
    )
    if os.path.isdir(abs_path):
        return f"Error: {raw_path} is a directory; grep_search takes one file."
    if not os.path.isfile(abs_path):
        return f"Error: file not found: {raw_path}"
    try:
        inside = Path(abs_path).is_relative_to(root)
    except Exception:  # noqa: BLE001 - defensive (odd paths)
        inside = False
    if not inside:
        return f"Error: {raw_path} is outside the workspace root {root}."

    relpath = os.path.relpath(abs_path, root)
    try:
        doc = Document.load(abs_path)
    except Exception as exc:  # noqa: BLE001 - binary / invalid UTF-8
        return f"Error: cannot read {relpath} ({exc})."

    rows = [
        line_view(line_no, record)
        for line_no, record in enumerate(doc.lines, 1)
        if regex.search(record.content)
    ]
    if not rows:
        return f'No matches found for pattern "{pattern}" in {relpath}.'

    match_term = "match" if len(rows) == 1 else "matches"
    header = (
        f'Found {len(rows)} {match_term} for pattern "{pattern}" in {relpath} (ids):'
    )
    kept: list[str] = []
    size = len(header)
    dropped = 0
    for row in rows:
        if size + len(row) + 1 > MAX_OUTPUT_CHARS:
            dropped = len(rows) - len(kept)
            break
        kept.append(row)
        size += len(row) + 1
    result = "\n".join([header, *kept])
    if dropped:
        result += f"\n- … and {dropped} more (output capped; narrow the pattern)"
    return result
