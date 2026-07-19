"""The `lsp` query tool: code navigation over the multilspy language servers.

Interface-compatible with qwen-code's `lsp` tool (same wire name, parameter
names, 1-based positions, and result formats), restricted to the four
operations multilspy provides natively: goToDefinition, findReferences, hover,
and documentSymbol. Backend: the per-language ``SyncLanguageServer`` handles
owned by :mod:`tools.lsp.diagnostics` — one server per language per session,
shared with the diagnostics path.

Design rules (the package-wide ones):
- **Never raise.** Every failure returns an ``Error: ...`` string.
- **Bounded.** First use waits up to :data:`WAIT_READY_TIMEOUT` for the server
  to boot; each request is capped by ``diagnostics.SYNC_REQUEST_TIMEOUT``.
  Callers run ``execute`` via ``asyncio.to_thread``, so the TUI never blocks.
- **Disk is truth.** multilspy reads files from disk at didOpen; the hashline
  edit tools flush every mutation atomically, so mid-session queries from the
  editor sub-agent always see current content.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from . import diagnostics

LSP_NAME = "lsp"

_OPERATIONS = ("goToDefinition", "findReferences", "hover", "documentSymbol")
_POSITION_OPERATIONS = ("goToDefinition", "findReferences", "hover")

# How long the first call per language may wait for the server to boot. A
# bounded wait beats returning "try again" — that would waste a whole model
# round-trip (jedi is ready in ~1-2s; rust-analyzer/tsserver need headroom).
WAIT_READY_TIMEOUT = 15.0

# Fixed per-operation result caps (no `limit` arg — one less knob for the
# model). Sized so honest results never truncate: definitions are single
# digits, and a ~100-symbol file or a widely-used helper stays whole; the cap
# only guards the context against pathological thousand-row responses.
RESULT_LIMITS = {"goToDefinition": 50, "findReferences": 200, "documentSymbol": 200}

# Captured at import, same idiom as tools/syntax/validate.py: the agent's
# workspace root never changes after launch, and the language servers are
# rooted here.
_WORKSPACE_ROOT = os.getcwd()

LSP_SCHEMA = {
    "name": LSP_NAME,
    "description": (
        "Ask the language server about a symbol: where it is defined "
        "(goToDefinition), everywhere it is used (findReferences), its "
        "type/signature/docs (hover), or all symbols in a file "
        "(documentSymbol). Positions are 1-based. Read-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": list(_OPERATIONS),
                "description": "The LSP operation to perform.",
            },
            "filePath": {
                "type": "string",
                "description": "File path (absolute or workspace-relative).",
            },
            "line": {
                "type": "number",
                "description": (
                    "1-based line number of the symbol (not needed for "
                    "documentSymbol)."
                ),
            },
            "character": {
                "type": "number",
                "description": "1-based column number of the symbol. Defaults to 1.",
            },
        },
        "required": ["operation"],
    },
}

GUIDANCE = """\
- lsp asks the language server about a symbol — more precise than grep_search
  when you need where a symbol is DEFINED (goToDefinition), everywhere it is
  USED (findReferences), its type/signature (hover), or a file's outline
  (documentSymbol). Give filePath plus the 1-based line/character of the
  symbol (documentSymbol needs only filePath). Read-only."""

# LSP SymbolKind numeric values → readable names, kept local so rendering
# never depends on importing multilspy (optional dependency rule).
_SYMBOL_KINDS = {
    1: "File", 2: "Module", 3: "Namespace", 4: "Package", 5: "Class",
    6: "Method", 7: "Property", 8: "Field", 9: "Constructor", 10: "Enum",
    11: "Interface", 12: "Function", 13: "Variable", 14: "Constant",
    15: "String", 16: "Number", 17: "Boolean", 18: "Array", 19: "Object",
    20: "Key", 21: "Null", 22: "EnumMember", 23: "Struct", 24: "Event",
    25: "Operator", 26: "TypeParameter",
}


class LspQueryTool:
    """Engine behind the `lsp` tool. Stateless; safe to share across agents."""

    def execute(self, arguments: dict) -> str:
        try:
            return self._execute(arguments)
        except Exception as exc:  # noqa: BLE001 - never raise into a registry
            return f"Error: LSP request failed ({exc})."

    def _execute(self, args: dict) -> str:
        operation = args.get("operation")
        if operation not in _OPERATIONS:
            return (
                "Error: 'operation' must be one of "
                f"{', '.join(_OPERATIONS)}."
            )
        file_path = args.get("filePath")
        if not isinstance(file_path, str) or not file_path.strip():
            return "Error: 'filePath' is required."

        line0 = col0 = 0
        line_id = None
        if operation in _POSITION_OPERATIONS:
            line = args.get("line")
            if isinstance(line, str):
                # Plain 1-based numbers are the primary form; a string falls
                # back to the hashline dialect — the line number immediately
                # followed by the 2-hex content hash ("12af" = line 12, hash
                # af) — accepted from EVERY agent and hash-verified against
                # the live file once the Document is loaded below.
                line_id = self._parse_line_id(line)
            if line_id is None and (not isinstance(line, (int, float)) or int(line) < 1):
                return (
                    "Error: 'line' must be a 1-based line number or a "
                    "current line id like '12af'."
                )
            character = args.get("character", 1)
            if not isinstance(character, (int, float)) or int(character) < 1:
                return "Error: 'character' must be a positive number (1-based)."
            # Wire is 1-based (qwen-code convention); multilspy speaks LSP 0-based.
            if line_id is None:
                line0 = int(line) - 1
            col0 = max(0, int(character) - 1)

        cap = RESULT_LIMITS.get(operation, 200)

        if not diagnostics.is_enabled():
            return "Error: LSP is disabled (--no-lsp)."
        root = os.path.realpath(_WORKSPACE_ROOT)
        abs_path = os.path.realpath(
            file_path if os.path.isabs(file_path) else os.path.join(root, file_path)
        )
        if not os.path.isfile(abs_path):
            return f"Error: file not found: {file_path}"
        try:
            inside = Path(abs_path).is_relative_to(root)
        except Exception:  # noqa: BLE001 - defensive (odd paths)
            inside = False
        if not inside:
            return f"Error: {file_path} is outside the workspace root {root}."
        language = diagnostics.supported_language(abs_path)
        if language is None:
            supported = ", ".join(sorted(set(diagnostics._EXT_TO_LSP_LANG)))
            return (
                f"Error: no language server for {Path(abs_path).suffix or 'this'} "
                f"files (supported: {supported})."
            )

        handle = diagnostics.get_ready_server(language, root, wait=WAIT_READY_TIMEOUT)
        if handle.status == "failed":
            return (
                f"Error: the {language} language server failed to start; "
                f"lsp is unavailable for {language} files this session."
            )
        if handle.status != "ready":
            return (
                f"Error: the {language} language server is still starting; "
                "try again in a moment."
            )

        relpath = os.path.relpath(abs_path, root)
        # The Document serves two independent jobs: hash-verifying a string
        # `line` id (accepted from any agent), and rendering result rows in
        # the queried file as id|content — the latter ONLY under the private
        # _hashline flag injected by the editor registry (never in
        # LSP_SCHEMA), so non-editor agents always read plain path:line:col.
        # Foreign-file rows and anything the hashline loader rejects keep
        # plain positions too.
        hashline_out = bool(args.get("_hashline"))
        doc = self._load_doc(abs_path) if hashline_out or line_id is not None else None
        if line_id is not None:
            resolved = self._resolve_line_id(doc, line_id, relpath)
            if isinstance(resolved, str):
                return resolved
            line0 = resolved - 1
        render_doc = doc if hashline_out else None
        desc = f"{relpath}:{line0 + 1}:{col0 + 1}"
        try:
            # The lock serialises with diagnostics cycles and other queries on
            # the same server. The sync wrappers open the file themselves and
            # block with diagnostics.SYNC_REQUEST_TIMEOUT.
            with handle.lock:
                if operation == "goToDefinition":
                    results = handle.sync_ls.request_definition(relpath, line0, col0)
                    return self._format_locations(
                        f"Goto definition for {desc}:",
                        f"No definition found for {desc}.",
                        results, root, cap, relpath, render_doc,
                    )
                if operation == "findReferences":
                    # multilspy hard-codes includeDeclaration=False, so the
                    # qwen-code arg is deliberately absent from LSP_SCHEMA
                    # (silently ignored if sent); the declaration is reachable
                    # via goToDefinition.
                    results = handle.sync_ls.request_references(relpath, line0, col0)
                    return self._format_locations(
                        f"References for {desc}:",
                        f"No references found for {desc}.",
                        results, root, cap, relpath, render_doc,
                    )
                if operation == "hover":
                    hover = handle.sync_ls.request_hover(relpath, line0, col0)
                    return self._format_hover(desc, hover)
                symbols, _tree = handle.sync_ls.request_document_symbols(relpath)
                return self._format_symbols(relpath, symbols, root, cap, render_doc)
        except TimeoutError:
            return (
                "Error: the language server did not answer within "
                f"{diagnostics.SYNC_REQUEST_TIMEOUT}s."
            )
        except Exception as exc:  # noqa: BLE001 - degrade, never raise
            return f"Error: LSP request failed ({exc})."

    # --- formatting ------------------------------------------------------

    def _load_doc(self, abs_path: str):
        """Hashline ``Document`` for the requested file, or ``None``.

        Lazy import: keeps the optional editor-dialect path from coupling this
        module's import to the hashline package. ``None`` on any failure
        (binary file, invalid UTF-8) → plain ``path:line:col`` rendering.
        """
        try:
            from tools.files.hashline.document import Document

            return Document.load(Path(abs_path))
        except Exception:  # noqa: BLE001 - dialect rendering is best-effort
            return None

    def _id_row(self, doc: Any, line_no: int) -> str | None:
        """``<line><hash>|<content>`` for a line of the loaded doc, or ``None``."""
        try:
            from tools.files.hashline.output import line_view

            if doc is not None and 1 <= line_no <= len(doc.lines):
                return line_view(line_no, doc.lines[line_no - 1])
        except Exception:  # noqa: BLE001 - fall back to plain positions
            pass
        return None

    def _parse_line_id(self, value: str) -> tuple[int, str] | None:
        """``(line_no, hash)`` for a well-formed hashline line id, else ``None``.

        Grammar (hashline's ``format_line_ref``): the 1-based line number
        immediately followed by the 2-lowercase-hex content hash, no
        separator. The hash is always the fixed trailing 2 chars, so
        "7240" parses as line 72, hash 40 — never line 724.
        """
        match = re.fullmatch(r"([0-9]+)([0-9a-f]{2})", value)
        if match is None or int(match.group(1)) < 1:
            return None
        return int(match.group(1)), match.group(2)

    def _resolve_line_id(
        self, doc: Any, line_id: tuple[int, str], relpath: str
    ) -> int | str:
        """1-based line for a parsed id, or an ``Error: ...`` string.

        The hash byte is checked against the CURRENT file: a mismatch means
        the editor's ids predate an edit, and silently querying the numeric
        line would navigate from whatever now lives there — better to make
        the drift explicit and have the model refresh its view.
        """
        line_no, want = line_id
        shown = f"{line_no}{want}"
        if doc is None:
            return (
                f"Error: cannot verify line id '{shown}' ({relpath} is not "
                "hashline-readable); pass a plain 1-based line number."
            )
        if line_no > len(doc.lines):
            return (
                f"Error: stale line id '{shown}' for {relpath}; the file has "
                f"{len(doc.lines)} lines. Re-read the file to refresh the "
                "ids you hold before retrying."
            )
        from tools.files.hashline.hash import format_short_hash

        have = format_short_hash(doc.lines[line_no - 1].short_hash)
        if have != want:
            return (
                f"Error: stale line id '{shown}' for {relpath}; that line's "
                f"current id is '{line_no}{have}'. Re-read the file to "
                "refresh the ids you hold before retrying."
            )
        return line_no

    def _display_path(self, item: dict, root: str, fallback: str | None = None) -> str:
        """Workspace-relative path for a Location; absolute outside the root.

        ``fallback`` names the queried file: per LSP a result with no URI of
        its own (hierarchical DocumentSymbol rows carry none) is implicitly
        IN the queried document, so pathless items render there instead of
        as a dead-end ``<unknown>``. Resolvable paths are never overridden.
        """
        rel = item.get("relativePath")
        if isinstance(rel, str) and rel:
            return rel
        abs_path = item.get("absolutePath") or ""
        try:
            if abs_path and Path(abs_path).is_relative_to(root):
                return os.path.relpath(abs_path, root)
        except Exception:  # noqa: BLE001 - defensive (odd paths)
            pass
        return abs_path or fallback or "<unknown>"

    def _format_locations(
        self,
        heading: str,
        empty: str,
        results: Any,
        root: str,
        cap: int,
        editable_rel: str | None = None,
        doc: Any = None,
    ) -> str:
        rows = []
        for item in results or []:
            start = (item.get("range") or {}).get("start") or {}
            path = self._display_path(item, root, editable_rel)
            line = int(start.get("line", 0)) + 1
            col = int(start.get("character", 0)) + 1
            row = f"{path}:{line}:{col}"
            if doc is not None and path == editable_rel:
                row = self._id_row(doc, line) or row
            rows.append(row)
        if not rows:
            return empty
        lines = [heading]
        lines.extend(f"{i}. {row}" for i, row in enumerate(rows[:cap], 1))
        extra = len(rows) - cap
        if extra > 0:
            lines.append(f"- … and {extra} more")
        return "\n".join(lines)

    def _format_hover(self, desc: str, hover: Any) -> str:
        text = self._render_hover_contents(
            (hover or {}).get("contents") if isinstance(hover, dict) else None
        )
        if not text:
            return f"No hover information for {desc}."
        return f"Hover for {desc}:\n{text}"

    def _render_hover_contents(self, contents: Any) -> str:
        """MarkupContent | MarkedString | list[MarkedString] → plain text."""
        if isinstance(contents, dict):
            return str(contents.get("value", "")).strip()
        if isinstance(contents, str):
            return contents.strip()
        if isinstance(contents, list):
            parts = [self._render_hover_contents(part) for part in contents]
            return "\n\n".join(p for p in parts if p)
        return ""

    def _format_symbols(
        self, relpath: str, symbols: Any, root: str, cap: int, doc: Any = None
    ) -> str:
        rows = []
        for sym in symbols or []:
            if not isinstance(sym, dict):
                continue
            name = sym.get("name", "<unnamed>")
            kind = sym.get("kind")
            kind_name = _SYMBOL_KINDS.get(int(kind), f"kind {kind}") if isinstance(
                kind, (int, float)
            ) else "Unknown"
            start = self._symbol_start(sym)
            # Pathless rows (hierarchical DocumentSymbol has no per-symbol
            # URI) belong to the queried file — _display_path falls back.
            path = self._display_path(sym.get("location") or sym, root, relpath)
            container = sym.get("containerName")
            in_part = f" in {container}" if container else ""
            line = int(start.get("line", 0)) + 1
            col = int(start.get("character", 0)) + 1
            pos = f"{path}:{line}:{col}"
            if doc is not None and path == relpath:
                pos = self._id_row(doc, line) or pos
            rows.append(f"{name} ({kind_name}){in_part} - {pos}")
        if not rows:
            return f"No symbols found in {relpath}."
        lines = [f"Document symbols for {relpath}:"]
        lines.extend(f"{i}. {row}" for i, row in enumerate(rows[:cap], 1))
        extra = len(rows) - cap
        if extra > 0:
            lines.append(f"- … and {extra} more")
        return "\n".join(lines)

    def _symbol_start(self, sym: dict) -> dict:
        """Best-available start position: selectionRange → range → location.range."""
        for candidate in (
            sym.get("selectionRange"),
            sym.get("range"),
            (sym.get("location") or {}).get("range"),
        ):
            if isinstance(candidate, dict) and isinstance(candidate.get("start"), dict):
                return candidate["start"]
        return {}
