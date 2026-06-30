"""Deterministic tree-sitter syntax checking for tool output.

Exposes :func:`check_file`, :func:`check_bash`, and :func:`format_warning_block`,
used to attach concise, position-anchored syntax warnings to read/edit/bash tool
results so the model learns deterministically *that* and *where* its code is
malformed. Every entry point degrades to a silent no-op when the optional
``tree-sitter-language-pack`` package is unavailable or the language is
unrecognised.
"""

from __future__ import annotations

from .checker import (
    check_bash,
    check_file,
    check_file_structured,
    format_report,
    format_warning_block,
    language_for_path,
    syntax_report,
)

__all__ = [
    "check_bash",
    "check_file",
    "check_file_structured",
    "format_report",
    "format_warning_block",
    "language_for_path",
    "syntax_report",
]
