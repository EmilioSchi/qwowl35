"""Diagnostics presentation layer: section grammar + per-agent dedup memory.

This package isolates HOW diagnostics reach a model's context, away from HOW
they are produced (``tools/syntax`` routes validation, ``tools/lsp`` talks to
language servers — both unchanged and imported one-way from here, never back).

Two concerns live here and nowhere else:

- **Section grammar** (:mod:`render`): every diagnostics block a tool appends
  (``Syntax check (…)``, ``LSP diagnostics (…)``) is a *trailing section* joined
  to the tool body with one blank line. ``split_trailing_section`` is the single
  splitter shared by the compressor (so code-cutting logic never touches
  diagnostics), the TUI renderers (so diagnostics never render as diff/code
  lines), and the orchestrator (no more ``partition("\\n\\n")`` guesswork).
- **Per-agent memory** (:mod:`memory`): each running agent instance remembers
  which diagnostic lines it has already been shown and is not shown them again;
  headline counts stay honest so a persisting error state is never mistaken for
  a clean file. A freshly spawned agent starts with an empty memory and sees
  everything once.
"""

from .memory import DiagnosticsMemory, SiftedDiagnostics
from .render import (
    ALL_UNCHANGED,
    SECTION_PREFIXES,
    clean_validation_report,
    is_section_start,
    join_section,
    split_trailing_section,
    unchanged_note,
    validation_report_with_memory,
)

__all__ = [
    "DiagnosticsMemory",
    "SiftedDiagnostics",
    "ALL_UNCHANGED",
    "SECTION_PREFIXES",
    "clean_validation_report",
    "validation_report_with_memory",
    "is_section_start",
    "join_section",
    "split_trailing_section",
    "unchanged_note",
]
