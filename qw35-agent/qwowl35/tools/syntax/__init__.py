"""Code validation for tool output: LSP semantic diagnostics + tree-sitter.

The file tools call :func:`validate_file` / :func:`validation_report`
(``validate.py``), which prefer the multilspy LSP layer (``tools/lsp``) and
fall back to the tree-sitter checker (``checker.py``) whenever LSP cannot
answer. The checker's entry points (:func:`check_file`, :func:`check_bash`,
:func:`format_warning_block`, …) remain exported for the bash analyzer and
direct callers. Every entry point degrades to a silent no-op when the optional
``multilspy`` / ``tree-sitter-language-pack`` packages are unavailable or the
language is unrecognised.
"""

from __future__ import annotations

from .checker import (
    check_bash,
    check_file,
    check_file_structured,
    check_file_structured_checked,
    format_report,
    format_warning_block,
    language_for_path,
    syntax_report,
)
from .validate import Validation, validate_file, validation_report, warm_lsp

__all__ = [
    "Validation",
    "warm_lsp",
    "check_bash",
    "check_file",
    "check_file_structured",
    "check_file_structured_checked",
    "format_report",
    "format_warning_block",
    "language_for_path",
    "syntax_report",
    "validate_file",
    "validation_report",
]
