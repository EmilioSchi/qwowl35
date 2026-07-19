"""Tree-sitter syntax checking → human-readable warnings.

Given a file path + source (or a bash command), parse with tree-sitter and walk
the tree for ``ERROR`` / missing nodes, turning each into a concise, 1-based,
position-anchored message. The model already sees tool output; appending these
warnings tells it deterministically that — and where — its code is malformed, so
it can fix the syntax before relying on the file.

Design rules:
- **Never raise.** Every public function wraps its body so a checker bug can
  never turn a successful tool result into an error.
- **No-op cleanly** when the optional ``tree-sitter-language-pack`` package (or a
  particular grammar) is unavailable, when the extension is unrecognised, or when
  the source is too large.
- **Cache parsers** per language so grammars load once per process.

Node access mirrors the defensive helpers in :mod:`tools.bash.analyzer`: tree
node members are read through :func:`_call` so the same code works whether a
given tree-sitter version exposes them as properties or methods.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

# Beyond this size we skip parsing: the model rarely reads such files whole, and
# parsing megabytes on every read/edit is not worth it.
_MAX_SOURCE_BYTES = 1_000_000
# How many issues we list in full before summarising the rest.
_SHOWN_ISSUES = 5
# Hard cap on issues collected during the walk, bounding work on pathologically
# broken files.
_COLLECT_CAP = 20
# Maximum length of an "unexpected ..." snippet.
_SNIPPET_CHARS = 24

# Curated extension → tree-sitter grammar name. The single place to extend
# language coverage. Ambiguous ``.h`` (C vs C++ headers) is deliberately omitted
# to avoid false positives. This is a sibling to ``language_for_extension`` in
# ``tools/files/hashline/commands/find_block.py`` (which returns block-strategy
# buckets, not grammar names) — kept separate on purpose.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".java": "java",
    ".rb": "ruby",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "bash",
    ".bash": "bash",
}

# Per-thread {language: parser-or-None} cache. tree-sitter ``Parser`` objects are
# unsendable (PyO3 panics if one is used on a thread other than its creator's), and
# the file tools run via ``asyncio.to_thread`` (a thread *pool*), so parsers must
# never be shared across threads. ``get_parser`` returns a fresh object per call,
# so a thread-local cache gives each thread its own reusable parsers. ``None`` is
# cached too, so a missing grammar is not retried on every call.
_PARSERS = threading.local()

# Whether this tree-sitter binding's ``Parser.parse`` wants ``bytes`` or ``str``;
# probed on first use (see :func:`_parse`). A plain string flag — safe to share.
_PARSE_MODE: str | None = None


def language_for_path(path: str | Path) -> str | None:
    """Tree-sitter grammar name for ``path``'s extension, or ``None``."""
    try:
        return _EXT_TO_LANG.get(Path(path).suffix.lower())
    except Exception:  # noqa: BLE001 - never raise to callers
        return None


def check_file(path: str | Path, source: str) -> list[str]:
    """Syntax-error messages for ``source`` of ``path``'s language.

    Empty when the language is unknown, tree-sitter is unavailable, the file is
    too large, or it parses clean. (Use :func:`syntax_report` for a block that
    also confirms a clean parse.)
    """
    try:
        return _check(language_for_path(path), source)[0]
    except Exception:  # noqa: BLE001 - syntax warnings are best-effort
        return []


def check_bash(command: str) -> list[str]:
    """Syntax-error messages for a bash ``command``, or an empty list."""
    try:
        if not command or not command.strip():
            return []
        return _check("bash", command)[0]
    except Exception:  # noqa: BLE001 - syntax warnings are best-effort
        return []


def check_file_structured(path: str | Path, source: str) -> list[tuple[int, int, str]]:
    """Structured syntax errors: ``(line, col, message)`` tuples, 1-based.

    Same detection as :func:`check_file` (it shares the parse + walk) but returns
    positions as ints, so callers can map an error to a line anchor without
    parsing message text. Empty when the language is unknown, tree-sitter is
    unavailable, the file is too large, or it parses clean. Never raises.
    """
    try:
        return _check_structured(language_for_path(path), source)[0]
    except Exception:  # noqa: BLE001 - best-effort
        return []


def check_file_structured_checked(
    path: str | Path, source: str
) -> tuple[list[tuple[int, int, str]], bool]:
    """:func:`check_file_structured` plus the ``checked`` flag.

    The validation router (``validate.py``) needs to distinguish "parsed clean"
    from "could not check" on its tree-sitter fallback path, the same way
    :func:`syntax_report` does via :func:`_check`. Never raises.
    """
    try:
        return _check_structured(language_for_path(path), source)
    except Exception:  # noqa: BLE001 - best-effort
        return [], False


def syntax_report(path: str | Path, source: str) -> str:
    """Status block for a file: error list, a clean-parse confirmation, or ``""``.

    Returns ``""`` only when we could not actually check (unknown language,
    tree-sitter unavailable, file too large) — so an OK line is reported exactly
    when the file really was parsed and found clean.
    """
    try:
        language = language_for_path(path)
        msgs, checked = _check(language, source)
        return format_report(language or "syntax", msgs, checked)
    except Exception:  # noqa: BLE001 - best-effort
        return ""


def format_warning_block(label: str, msgs: list[str]) -> str:
    """Render error ``msgs`` into the block appended to a tool result, or ``""``."""
    if not msgs:
        return ""
    shown = msgs[:_SHOWN_ISSUES]
    extra = len(msgs) - len(shown)
    lines = [f"Syntax check ({label}) — {len(msgs)} issue(s):"]
    lines.extend(f"- {m}" for m in shown)
    if extra > 0:
        lines.append(f"- … and {extra} more")
    return "\n".join(lines)


def format_report(label: str, msgs: list[str], checked: bool) -> str:
    """Error block when ``msgs``; a clean confirmation when ``checked``; else ``""``."""
    if msgs:
        return format_warning_block(label, msgs)
    if checked:
        return f"Syntax check ({label}): OK — no syntax errors."
    return ""


# --- internals --------------------------------------------------------------


def _parse_root(language: str | None, source: str) -> tuple[Any | None, bytes, bool]:
    """Return ``(root_node_or_None, source_bytes, checked)``.

    ``checked`` is True only when the source was actually parsed (language known,
    parser available, size within bounds, parse produced a tree with a root) — so
    callers can distinguish "clean" from "not checked". Shared by :func:`_check`
    (string messages) and :func:`_check_structured` (positions).
    """
    source_bytes = source.encode("utf-8", errors="replace")
    if not language:
        return None, source_bytes, False
    if len(source_bytes) > _MAX_SOURCE_BYTES:
        return None, source_bytes, False
    parser = _get_parser(language)
    if parser is None:
        return None, source_bytes, False
    tree = _parse(parser, source_bytes)
    if tree is None:
        return None, source_bytes, False
    root = _attr(tree, "root_node", None)
    return root, source_bytes, root is not None


def _check(language: str | None, source: str) -> tuple[list[str], bool]:
    """Return ``(error_messages, checked)``."""
    root, source_bytes, checked = _parse_root(language, source)
    if root is None:
        return [], checked
    if not bool(_attr(root, "has_error", False)):
        return [], True  # parsed clean
    issues: list[str] = []
    _collect(root, source_bytes, issues)
    if not issues:
        # ``has_error`` is set but no ERROR/missing leaf was localised (rare); give
        # a deterministic fallback rather than staying silent about a broken file.
        issues.append("syntax error detected (could not localise)")
    return issues, True


def _check_structured(
    language: str | None, source: str
) -> tuple[list[tuple[int, int, str]], bool]:
    """Structured counterpart of :func:`_check`: ``((line, col, message) tuples, checked)``."""
    root, source_bytes, checked = _parse_root(language, source)
    if root is None:
        return [], checked
    if not bool(_attr(root, "has_error", False)):
        return [], True  # parsed clean
    issues: list[tuple[int, int, str]] = []
    _collect_structured(root, source_bytes, issues)
    if not issues:
        issues.append((1, 1, "syntax error detected (could not localise)"))
    return issues, True


def _get_parser(language: str) -> Any | None:
    cache = getattr(_PARSERS, "by_lang", None)
    if cache is None:
        cache = {}
        _PARSERS.by_lang = cache
    if language in cache:
        return cache[language]
    parser: Any | None = None
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(language)
    except Exception:  # noqa: BLE001 - optional dependency / unknown grammar
        parser = None
    cache[language] = parser
    return parser


def _parse(parser: Any, source: bytes) -> Any | None:
    """Parse ``source`` bytes, tolerating bindings that want ``str`` vs ``bytes``.

    Different tree-sitter versions disagree on the argument type, so we probe once
    and remember the working form in ``_PARSE_MODE`` to avoid raising a
    ``TypeError`` on every subsequent parse.
    """
    global _PARSE_MODE
    try:
        if _PARSE_MODE == "str":
            return parser.parse(source.decode("utf8", errors="replace"))
        if _PARSE_MODE == "bytes":
            return parser.parse(source)
        try:
            tree = parser.parse(source)
            _PARSE_MODE = "bytes"
            return tree
        except TypeError:
            tree = parser.parse(source.decode("utf8", errors="replace"))
            _PARSE_MODE = "str"
            return tree
    except Exception:  # noqa: BLE001 - defensive
        return None


def _collect(node: Any, source: bytes, out: list[str]) -> None:
    """Record ERROR / missing nodes at or under ``node`` into ``out``.

    Evaluates ``node`` itself (some grammars make the *root* the ERROR node) then
    recurses, pruning clean subtrees via ``has_error``.
    """
    if len(out) >= _COLLECT_CAP:
        return
    if bool(_attr(node, "is_missing", False)):
        out.append(_missing_message(node, source))
        return
    if _node_kind(node) == "ERROR" or bool(_attr(node, "is_error", False)):
        out.append(_error_message(node, source))
        return  # Do not descend into a salvaged ERROR subtree.
    if not bool(_attr(node, "has_error", False)):
        return  # Clean subtree — nothing to report.
    for child in _node_children(node):
        if len(out) >= _COLLECT_CAP:
            return
        _collect(child, source, out)


def _collect_structured(node: Any, source: bytes, out: list[tuple[int, int, str]]) -> None:
    """Like :func:`_collect` but appends ``(line, col, message)`` tuples."""
    if len(out) >= _COLLECT_CAP:
        return
    if bool(_attr(node, "is_missing", False)):
        line, col = _position(node, source)
        out.append((line, col, _missing_message(node, source)))
        return
    if _node_kind(node) == "ERROR" or bool(_attr(node, "is_error", False)):
        line, col = _position(node, source)
        out.append((line, col, _error_message(node, source)))
        return  # Do not descend into a salvaged ERROR subtree.
    if not bool(_attr(node, "has_error", False)):
        return  # Clean subtree — nothing to report.
    for child in _node_children(node):
        if len(out) >= _COLLECT_CAP:
            return
        _collect_structured(child, source, out)


def _missing_message(node: Any, source: bytes) -> str:
    line, _col = _position(node, source)
    kind = _node_kind(node)
    if kind:
        return f"line {line}: missing '{kind}'"
    return f"line {line}: missing token"


def _error_message(node: Any, source: bytes) -> str:
    line, col = _position(node, source)
    snippet = _snippet(node, source)
    if snippet:
        return f"line {line}, col {col}: unexpected '{snippet}'"
    return f"line {line}, col {col}: syntax error"


def _snippet(node: Any, source: bytes) -> str:
    start, end = _byte_span(node)
    if end <= start:
        return ""
    try:
        text = source[start:end].decode("utf8", errors="replace")
    except Exception:  # noqa: BLE001 - defensive
        return ""
    text = " ".join(text.split())  # collapse newlines / runs of whitespace
    if len(text) > _SNIPPET_CHARS:
        text = text[:_SNIPPET_CHARS].rstrip() + "…"
    return text


def _byte_span(node: Any) -> tuple[int, int]:
    try:
        start = int(_attr(node, "start_byte", 0) or 0)
        end = int(_attr(node, "end_byte", 0) or 0)
    except Exception:  # noqa: BLE001 - defensive
        return 0, 0
    return start, end


def _position(node: Any, source: bytes) -> tuple[int, int]:
    """1-based (line, column) of ``node`` derived from its UTF-8 byte offset.

    The ``start_point`` member is unreliable across tree-sitter bindings (it is
    ``None`` when parsing from ``str``), but byte offsets are always populated, so
    we compute line/column directly from the source bytes. Columns are counted in
    characters, not bytes, so multi-byte text reports a sensible column.
    """
    start, _end = _byte_span(node)
    if start <= 0:
        return 1, 1
    try:
        prefix = source[:start]
        line = prefix.count(b"\n") + 1
        line_start = prefix.rfind(b"\n") + 1  # 0 when on the first line
        col = len(source[line_start:start].decode("utf8", errors="replace")) + 1
    except Exception:  # noqa: BLE001 - defensive
        return 1, 1
    return line, col


def _call(value: Any) -> Any:
    return value() if callable(value) else value


def _attr(node: Any, name: str, default: Any = None) -> Any:
    return _call(getattr(node, name, default))


def _node_kind(node: Any) -> str:
    kind = getattr(node, "type", None)
    if kind is not None:
        return str(_call(kind))
    return str(_call(getattr(node, "kind", "")) or "")


def _node_children(node: Any) -> list[Any]:
    children = getattr(node, "children", None)
    if children is not None and not callable(children):
        return list(children)
    if callable(children):
        return list(children())
    count = int(_call(getattr(node, "child_count", 0)) or 0)
    return [node.child(i) for i in range(count)]
