"""System-prompt guidance for the explorer search tools."""

from __future__ import annotations

GUIDANCE = """\
Use the search tools to understand the project before acting:
- `list_directory` walks one directory level (absolute path required).
- `glob` finds files by name pattern, newest first.
- `grep_search` finds file content by regex (case-insensitive), grouped per file
  with `L<line>:` markers.
- `inspect_file` returns a file's text (absolute path); page big files with
  `offset`/`limit`. Inspection is read-only.

Search efficiently: prefer one well-aimed `grep_search` over walking directories
level by level, and inspect only the files whose content you actually need.
Long results are auto-compressed; a `[compressed: ...]` tail reports what was
elided — re-call the same tool with `compress:false` only if you truly need
everything."""
