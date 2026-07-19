"""Whole-line comment pruning via tree-sitter comment nodes.

The language-aware pass behind ``compress_code``: every line that is entirely
comment (including lone lines, the shebang, and Python docstrings) is blanked
in place — the text goes, the line stays — and the total elided-line count is
returned to the caller rather than inserted in-place per run (repo rule:
nothing disappears silently, but the note lives once at the end of the whole
tool result, not scattered through the body; ``compress:false`` recovers the
original). Line NUMBERING is never altered: blanked lines keep their slot, so
LSP diagnostics references, offset/limit paging, and the model's own line
counting all stay aligned with the on-disk file. Code lines are NEVER
altered — a trailing ``x = 1  # note`` stays verbatim, and tree-sitter's
exact spans mean a string literal containing ``# not a comment`` can never be
mistaken for one.

Reuses the syntax checker's private defensive helpers rather than duplicating
them: ``_parse``'s str-vs-bytes probing, ``_attr``'s method-vs-property node
access, and ``_get_parser``'s per-thread cache are exactly the version-tolerant
machinery this walk needs, and a copy would drift. checker.py itself is
untouched.
"""

from __future__ import annotations

from tools.syntax.checker import (
    _attr,
    _byte_span,
    _get_parser,
    _node_children,
    _node_kind,
    _parse,
)

# Comment node kinds across the grammars detect.py can return. Most grammars
# use "comment"; rust and java split line/block, and rust nests doc_comment
# under those. A shebang parses as "comment" everywhere except javascript's
# hash_bang_line. Grammars without these kinds (e.g. json) make pruning a
# natural no-op.
_COMMENT_KINDS = frozenset(
    {"comment", "line_comment", "block_comment", "doc_comment", "hash_bang_line"}
)

# Python docstrings parse as statement-level string nodes, not comment nodes;
# one is prunable only when its parent is one of these (module body or the
# block of a def/class), which also catches stray bare-string no-op statements
# while leaving strings in assignments/calls untouched.
_DOCSTRING_PARENT_KINDS = frozenset({"module", "block"})


def prune_comment_lines(text: str, language: str) -> tuple[str, int]:
    """``(text, elided_count)``: comment-only lines blanked, or unchanged on failure.

    Elided lines are blanked in place (line numbering preserved) — no
    in-place marker — and the total count is returned so the caller can
    report it once, at the end of the whole tool result, instead of once per
    comment block.

    Error trees are accepted (no ``has_error`` gate): paged windows are
    routinely truncated mid-construct — exactly the large-file case where
    pruning matters — and the lexer identifies comment tokens robustly under
    error recovery. The per-line coverage check below re-validates every
    pruned line: a line is only elided when its every non-whitespace byte
    sits inside a comment span.
    """
    try:
        parser = _get_parser(language)
        if parser is None:
            return text, 0
        source_bytes = text.encode("utf-8", errors="replace")
        tree = _parse(parser, source_bytes)
        root = _attr(tree, "root_node") if tree is not None else None
        if root is None:
            return text, 0

        spans = _comment_spans(root, language)
        if not spans:
            return text, 0

        flags = _comment_only_lines(source_bytes, spans)
        lines = text.split("\n")
        out, elided = _elide_flagged(lines, flags)
        return "\n".join(out), elided
    except Exception:  # noqa: BLE001 - pruning is best-effort, fail open
        return text, 0


def _comment_spans(root, language: str) -> list[tuple[int, int]]:
    """Sorted, merged byte spans of every comment node under ``root``.

    Iterative stack walk: unlike the checker's ``has_error``-pruned recursion,
    every node must be visited, and deep trees would risk the recursion limit.
    Comment nodes are not descended into (their children, where a grammar has
    them, are part of the same span). For python the walk tracks each node's
    parent kind so docstrings — string-only expression_statements directly
    under a module or block — count as comments too; the language guard keeps
    json's string nodes (and every other grammar's) untouched.
    """
    spans: list[tuple[int, int]] = []
    stack = [(root, "")]
    while stack:
        node, parent_kind = stack.pop()
        kind = _node_kind(node)
        if kind in _COMMENT_KINDS or _is_docstring(language, node, kind, parent_kind):
            start, end = _byte_span(node)
            if end > start:
                spans.append((start, end))
            continue
        stack.extend((child, kind) for child in _node_children(node))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _is_docstring(language: str, node, kind: str, parent_kind: str) -> bool:
    """True for a python string statement in a docstring slot.

    Grammar versions differ: some wrap statement-level strings in an
    expression_statement, others (language-pack 1.8's python) put the string
    node directly under module/block. Both shapes are recognized; strings
    inside assignments, calls, etc. have other parents and never match.
    """
    if language != "python" or parent_kind not in _DOCSTRING_PARENT_KINDS:
        return False
    if kind == "string":
        return True
    if kind == "expression_statement":
        children = _node_children(node)
        return len(children) == 1 and _node_kind(children[0]) == "string"
    return False


def _comment_only_lines(source_bytes: bytes, spans: list[tuple[int, int]]) -> list[bool]:
    """Per-line flags: True when the line is entirely comment (plus whitespace).

    Computed on UTF-8 bytes because tree-sitter spans are byte offsets (same
    convention as checker._position). A line qualifies iff it is non-blank, at
    least one byte is covered by a comment span, and every uncovered byte is
    whitespace — which keeps trailing-comment lines (their code bytes are
    uncovered) and prunes the interior of whole-line block comments.
    """
    flags: list[bool] = []
    line_start = 0
    span_index = 0
    for line in source_bytes.split(b"\n"):
        line_end = line_start + len(line)
        # Advance past spans that end before this line.
        while span_index < len(spans) and spans[span_index][1] <= line_start:
            span_index += 1
        covered_any = False
        gaps_blank = True
        cursor = line_start
        i = span_index
        while i < len(spans) and spans[i][0] < line_end:
            start, end = spans[i]
            if start > cursor:
                if source_bytes[cursor:start].strip():
                    gaps_blank = False
                    break
            covered_any = True
            cursor = max(cursor, min(end, line_end))
            i += 1
        if gaps_blank and cursor < line_end and source_bytes[cursor:line_end].strip():
            gaps_blank = False
        flags.append(bool(line.strip()) and covered_any and gaps_blank)
        line_start = line_end + 1  # +1 for the split newline
    return flags


def _elide_flagged(lines: list[str], flags: list[bool]) -> tuple[list[str], int]:
    """Every flagged line blanked IN PLACE; returns (lines, total blanked).

    Blanking instead of deleting keeps the output's line numbering identical
    to the on-disk file. Deleting shifted everything below a comment block by
    its height, which broke every real-line reference the model holds: LSP
    diagnostics lines, offset/limit paging, its own line counting. Measured
    live: an explorer dismissed correct pylint findings ("line 127:
    learning_rate unused") because line 127 of its pruned view held different
    code, and reported "no unused variables" for a file with four.
    """
    out = list(lines)
    elided = 0
    for i, flagged in enumerate(flags):
        if flagged and i < len(out):
            out[i] = ""
            elided += 1
    return out, elided
