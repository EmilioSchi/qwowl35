"""Validation router: LSP semantic diagnostics first, tree-sitter fallback.

The single entry point the file tools call after a read or mutation. For
languages with a multilspy backend (``tools/lsp``) the check runs against a
real language server — semantic depth: unresolved symbols, type errors — and
is labelled ``<lang>, lsp``. Whenever LSP cannot answer (package or server
missing, server still booting, file outside the workspace, timeout, unsupported
language), the tree-sitter checker takes over. For files where LSP *should*
have answered (enabled, supported language, inside the workspace root) the
fallback label says so — ``<lang> — LSP unavailable, syntax-only`` — so a
clean result is never mistaken for a full semantic check; everywhere else
(tests in tempdirs, out-of-root files) the original labels and byte-identical
output are preserved, so every existing consumer and test keeps working.

The workspace root is captured at import time: the TUI never chdirs after
launch, and the headless runner's scratch dir lives under the launch dir. Files
outside the root (e.g. tests running in a tempdir) deterministically take the
tree-sitter path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .checker import check_file_structured_checked, format_report, language_for_path

_WORKSPACE_ROOT = os.getcwd()

# How many warning rows a report lists in full before summarising (errors are
# capped by the caller / checker._SHOWN_ISSUES as before).
_SHOWN_WARNINGS = 5


@dataclass(frozen=True)
class Validation:
    """Outcome of one validation pass, whichever layer answered."""

    errors: list[tuple[int, int, str]] = field(default_factory=list)
    warnings: list[tuple[int, int, str]] = field(default_factory=list)
    # e.g. "python, lsp" (LSP), "python" (tree-sitter by design), or
    # "python — LSP unavailable, syntax-only" (tree-sitter as a degraded fallback)
    label: str = "syntax"
    checked: bool = False  # False → nothing could be checked → report is ""

    def report(self) -> str:
        """The ``Syntax check (…)`` block for this outcome (or ``""``).

        Reuses :func:`checker.format_report`, feeding it whatever ``label``
        the router chose; LSP warnings are appended after the OK/error body
        without disturbing the first line (the TUI colors the block by its
        ``Syntax check (`` prefix and ``: OK`` text).
        """
        body = format_report(self.label, [m for _, _, m in self.errors], self.checked)
        if not body or not self.warnings:
            return body
        shown = self.warnings[:_SHOWN_WARNINGS]
        lines = [body, f"Warnings (not blocking) — {len(self.warnings)}:"]
        lines.extend(f"- {m}" for _, _, m in shown)
        extra = len(self.warnings) - len(shown)
        if extra > 0:
            lines.append(f"- … and {extra} more")
        return "\n".join(lines)


def validate_file(path: str | Path, source: str) -> Validation:
    """Validate ``source`` as the content of ``path``. Never raises."""
    try:
        if not source:
            return Validation()
        lsp = _lsp_result(path, source)
        if lsp is not None:
            language, (errors, warnings) = lsp
            return Validation(errors, warnings, f"{language}, lsp", True)
        issues, checked = check_file_structured_checked(path, source)
        label = language_for_path(path) or "syntax"
        # LSP was supposed to answer for this file but could not (booting,
        # timeout, crashed): say so, or a syntax-only OK reads as a full check.
        if checked and _lsp_expected(path):
            label = f"{label} — LSP unavailable, syntax-only"
        return Validation(issues, [], label, checked)
    except Exception:  # noqa: BLE001 - validation is best-effort
        return Validation()


def validation_report(path: str | Path, source: str) -> str:
    """Router-aware :func:`checker.syntax_report`: status block or ``""``."""
    try:
        return validate_file(path, source).report()
    except Exception:  # noqa: BLE001 - best-effort
        return ""


# How long a post-write validation may block for a language server that is
# still booting. A one-time cost per language per session (failed boots are
# cached and return immediately); read/edit hot paths never pay it.
LSP_BOOT_WAIT = 8.0


def warm_lsp(path: str | Path, wait: float = LSP_BOOT_WAIT) -> bool:
    """Block until ``path``'s language server is ready, up to ``wait`` seconds.

    WORKER THREADS ONLY — never call from the TUI event loop (a first boot can
    take the full ``wait``). Returns ``True`` iff the server is ready; ``False``
    fast when LSP is disabled, the language is unsupported, or the file is
    outside the workspace root (where :func:`lsp_check_file` would refuse
    anyway, so there is nothing worth booting). Never raises.
    """
    try:
        from ..lsp import get_ready_server, supported_language

        language = supported_language(path)
        if language is None:
            return False
        root_real = os.path.realpath(_WORKSPACE_ROOT)
        if not Path(os.path.realpath(str(path))).is_relative_to(root_real):
            return False
        return get_ready_server(language, root_real, wait).status == "ready"
    except Exception:  # noqa: BLE001 - warming is best-effort
        return False


def _lsp_expected(path: str | Path) -> bool:
    """Whether LSP was supposed to answer for ``path`` (so a fallback is a
    degradation worth flagging, not the designed route).

    True only when the language is supported AND LSP enabled (``supported_language``
    returns ``None`` when deliberately disabled) AND the file is inside the
    workspace root — outside-root files take tree-sitter by design.
    """
    try:
        from ..lsp import supported_language

        if supported_language(path) is None:
            return False
        root_real = os.path.realpath(_WORKSPACE_ROOT)
        return Path(os.path.realpath(str(path))).is_relative_to(root_real)
    except Exception:  # noqa: BLE001 - defensive
        return False


def _lsp_result(path: str | Path, source: str):
    """``(language, (errors, warnings))`` from the LSP layer, or ``None``.

    Fully guarded: ANY failure in the LSP layer (import, server, conversion)
    means "could not check semantically" and must land on the tree-sitter
    path, never abort validation altogether.
    """
    try:
        from ..lsp import lsp_check_file, supported_language

        # Content-first (same source lsp_check_file re-detects on), so a file
        # whose extension is unsupported but whose content is a supported
        # language still routes to LSP.
        language = supported_language(path, source)
        if language is None:
            return None
        result = lsp_check_file(path, source, _WORKSPACE_ROOT)
        if result is None:
            return None
        return language, result
    except Exception:  # noqa: BLE001 - defensive fallback to tree-sitter
        return None
