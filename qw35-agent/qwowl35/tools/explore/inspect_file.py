"""`inspect_file` — qwen-code read-file.ts replica (schema and output format).

Text files only: PDF/image/notebook support in the original needs decoders and
a cloud model we do not have; those inputs get a plain error instead.

Advertised as `inspect_file`, not qwen-code's trained `read_file` name: that
name was trained alongside write_file/edit/run_shell_command, and a model
that sees it in a READ-ONLY stage summons the whole family (observed live as
an endless "let me write the file" -> read_file loop). This exploration-native
name keeps the trained parameter conventions without triggering the
write-sibling prior. (The hashline layer's read tool deliberately makes the
opposite call: it IS the editing family's opener, so it uses the trained
`read_file` name and schema — the write-sibling prior is exactly right there.)
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # diagnostics are best-effort; never let them break a read tool.
    from ..syntax import validate_file, warm_lsp
except Exception:  # pragma: no cover - defensive fallback
    validate_file = None  # type: ignore[assignment]
    warm_lsp = None  # type: ignore[assignment]

try:  # presentation layer (section grammar + per-agent dedup); same guard.
    from ..diagnostics import ALL_UNCHANGED, DiagnosticsMemory, join_section, unchanged_note
except Exception:  # pragma: no cover - defensive fallback
    DiagnosticsMemory = None  # type: ignore[assignment]
    ALL_UNCHANGED = "all unchanged and already reported above"

    def join_section(body: str, section: str) -> str:  # type: ignore[misc]
        return f"{body}\n\n{section}" if body and section else body or section

    def unchanged_note(count: int, noun: str = "issue") -> str:  # type: ignore[misc]
        return f"- {count} unchanged {noun}(s) already reported above (not repeated)"

# qwen-code pages through text with a 2000-line default window.
DEFAULT_LIMIT = 2000
# Hard byte ceiling on one result: a line cap alone does not bound bytes
# (2000 JSONL lines once blew the server context at 50K+ tokens in a single
# result). Mirrors the bash tool's 50 KB output cap.
MAX_CONTENT_CHARS = 50_000

INSPECT_FILE_SCHEMA = {
    "name": "inspect_file",
    "description": (
        "Opens a file and returns its text for inspection. The file_path "
        "argument MUST be an absolute path. If the file is large, the content "
        "will be truncated and the response will indicate how to see more of "
        "it using the 'offset' and 'limit' parameters. Inspection is "
        "read-only: it never modifies anything."
    ),
    "parameters": {
        "properties": {
            "file_path": {
                "description": (
                    "The absolute path to the file to read (e.g., "
                    "'/home/user/project/file.txt'). Relative paths are not "
                    "supported. You must provide an absolute path."
                ),
                "type": "string",
            },
            "offset": {
                "description": (
                    "Optional: For text files, the 0-based line number to start "
                    "reading from. Requires 'limit' to be set. Use for paginating "
                    "through large files."
                ),
                "type": "number",
            },
            "limit": {
                "description": (
                    "Optional: For text files, maximum number of lines to read. "
                    "Use with 'offset' to paginate through large files. If "
                    "omitted, reads the entire file (if feasible, up to a default "
                    "limit)."
                ),
                "type": "number",
            },
            "compress": {
                "type": "boolean",
                "description": "Optional: false returns the full uncompressed output.",
            },
        },
        "required": ["file_path"],
        "type": "object",
    },
}


def _lsp_section(path: Path, text: str, memory=None) -> str:
    """The ``LSP diagnostics (…)`` block for ``path``, or ``""``.

    Emitted only when a real language server answered (label ``<lang>, lsp``)
    AND it reported at least one error or warning: clean files, unsupported
    languages, --no-lsp, and the tree-sitter fallback all stay silent, so a
    plain read never grows a status block. New diagnostics are listed in full —
    inspection is where the model decides what to fix, so nothing is elided —
    but rows this agent was already shown (``memory``) are summarised instead
    of repeated; the headline counts always describe the file's CURRENT state.
    ``text`` must be the complete file content (the LSP reads disk; a paged
    window would never match and would silently demote the check).
    """
    if validate_file is None or warm_lsp is None:
        return ""
    try:
        warm_lsp(path)
        v = validate_file(path, text)
        if not v.label.endswith(", lsp") or not (v.errors or v.warnings):
            return ""
        header = (
            f"LSP diagnostics ({v.label}) — "
            f"{len(v.errors)} error(s), {len(v.warnings)} warning(s):"
        )
        if memory is None and DiagnosticsMemory is not None:
            memory = DiagnosticsMemory()  # no caller store → per-call, no dedup
        if memory is None:
            sifted = None
        else:
            sifted = memory.sift(str(path), v, text)
        if sifted is None:
            errors, warnings = list(v.errors), list(v.warnings)
            prior_errors = prior_warnings = 0
        else:
            if sifted.all_prior:
                return f"{header[:-1]}: {ALL_UNCHANGED}."
            errors, warnings = sifted.errors, sifted.warnings
            prior_errors, prior_warnings = sifted.prior_errors, sifted.prior_warnings
        lines = [header]
        lines.extend(f"- {message}" for _line, _col, message in errors)
        if prior_errors:
            lines.append(unchanged_note(prior_errors, "error"))
        if warnings or prior_warnings:
            if v.errors:
                lines.append("Warnings (not blocking):")
            lines.extend(f"- {message}" for _line, _col, message in warnings)
            if prior_warnings:
                lines.append(unchanged_note(prior_warnings, "warning"))
        if sifted is not None:
            sifted.mark_rendered(errors, warnings)
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 - diagnostics are best-effort on a read
        return ""


def run_inspect_file(arguments: dict, memory=None) -> str:
    raw_path = arguments.get("file_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return "Error: 'file_path' is required."
    raw_path = raw_path.strip()
    if not os.path.isabs(raw_path):
        return f"Error: File path must be absolute: {raw_path}"
    path = Path(raw_path)
    if not path.exists():
        return f"Error: File not found: {raw_path}"
    if path.is_dir():
        return f"Error: Path is a directory, not a file: {raw_path}"

    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"Error: {exc}"
    if b"\x00" in data[:8192]:
        return (
            f"Error: {raw_path} appears to be a binary file; inspect_file handles "
            "text files only."
        )
    text = data.decode("utf-8", errors="replace")

    offset = arguments.get("offset")
    limit = arguments.get("limit")
    offset = int(offset) if isinstance(offset, (int, float)) and offset > 0 else 0
    limit = int(limit) if isinstance(limit, (int, float)) and limit > 0 else DEFAULT_LIMIT

    lines = text.splitlines()
    total = len(lines)
    window = lines[offset : offset + limit]
    if not window and total > 0:
        return f"Error: offset {offset} is past the end of the file ({total} lines)."
    content = "\n".join(window)

    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + (
            f"\n... (result truncated at {MAX_CONTENT_CHARS} characters; page "
            "through the file with offset/limit)"
        )

    # Diagnostics run on the full text, never the paged window: the language
    # server reads the file from disk, so anything less would never match.
    # Attached as a canonical trailing section (tools/diagnostics grammar), so
    # the compressor and the TUI can carve it back off without guessing.
    section = _lsp_section(path, text, memory)

    truncated = offset > 0 or total > offset + len(window)
    if truncated:
        # 1-based inclusive line numbers, matching read-file.ts.
        start = offset + 1
        end = offset + len(window)
        return join_section(
            f"Showing lines {start}-{end} of {total} total lines."
            f"\n\n---\n\n{content}",
            section,
        )
    return join_section(content, section)
