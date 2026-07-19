"""Section grammar shared by every diagnostics producer and consumer.

A diagnostics block is always a *trailing section*: it starts at column 0 with
one of :data:`SECTION_PREFIXES`, sits after exactly one blank separator line,
and runs to the end of the tool result. This module is the one place that
grammar is defined; producers join with :func:`join_section` and consumers
split with :func:`split_trailing_section`, so:

- the output compressor carves the section off before any code-cutting logic
  (comment pruning, repeat collapsing) and re-attaches it verbatim;
- the TUI carves it off before diff/code rendering, so diagnostics can never
  be mistaken for removed file lines or source code;
- the orchestrator separates an opened file's body from its diagnostics
  without guessing at blank lines.

Also home to the shared dedup wording, so every dialect summarises suppressed
rows with the same sentence and tests can pin one phrasing.
"""

from __future__ import annotations

# Column-0 header prefixes that can open a diagnostics section. Every section
# builder in the codebase starts its first line with one of these.
SECTION_PREFIXES: tuple[str, ...] = ("Syntax check (", "LSP diagnostics (")


def is_section_start(line: str) -> bool:
    """Whether ``line`` (uncut, column 0) opens a diagnostics section."""
    return line.startswith(SECTION_PREFIXES)


def join_section(body: str, section: str) -> str:
    """Canonically attach a diagnostics ``section`` to a tool-result ``body``."""
    if not section:
        return body
    if not body:
        return section
    return f"{body}\n\n{section}"


def split_trailing_section(text: str) -> tuple[str, str]:
    """Split ``text`` into ``(body, section)``; ``section`` is ``""`` when absent.

    The section starts at the LAST column-0 line matching
    :data:`SECTION_PREFIXES` that follows a blank line (or opens the text) and
    extends to the end. Taking the last candidate keeps file bodies that merely
    *mention* a header (docs, this repo's own tests) from truncating the real
    section; a false positive can only ever shield extra trailing text from
    compression, never corrupt it. ``join_section(body, section)`` restores
    ``text`` exactly.
    """
    if not text:
        return text, ""
    lines = text.split("\n")
    start: int | None = None
    for i, line in enumerate(lines):
        if is_section_start(line) and (i == 0 or lines[i - 1] == ""):
            start = i
    if start is None:
        return text, ""
    section = "\n".join(lines[start:])
    body_lines = lines[:start]
    if body_lines and body_lines[-1] == "":
        body_lines = body_lines[:-1]  # drop the canonical blank separator
    return "\n".join(body_lines), section


# One consistent phrasing for suppressed rows across every dialect. Explicit
# about WHERE the details are (earlier in this same context) so a small model
# does not go re-reading the file to find them.
def unchanged_note(count: int, noun: str = "issue") -> str:
    """Summary bullet for rows suppressed because this agent already saw them."""
    return f"- {count} unchanged {noun}(s) already reported above (not repeated)"


# Clause each dialect embeds in its own header when EVERY row was suppressed —
# the section collapses to one honest line instead of going silent (silence
# would read as "clean" and un-flag a still-broken file).
ALL_UNCHANGED = "all unchanged and already reported above"


def validation_report_with_memory(validation, sifted, cap: int = 5) -> str:
    """``Validation.report()`` with per-agent dedup, both cases.

    Same block shapes as the plain report — ``Syntax check (…) — N issue(s):``
    with capped bullets and the ``Warnings (not blocking) — N:`` tail, or the
    clean confirmation — except rows this agent already saw are summarised.
    Headline counts always describe the file's CURRENT state. Falls back to
    ``validation.report()`` verbatim when ``sifted`` is None.
    """
    if sifted is None:
        return validation.report()
    if not validation.errors:
        return clean_validation_report(validation, sifted, cap)
    total = len(validation.errors)
    if sifted.all_prior:
        return f"Syntax check ({validation.label}) — {total} issue(s), {ALL_UNCHANGED}."
    lines = [f"Syntax check ({validation.label}) — {total} issue(s):"]
    shown = sifted.errors[:cap]
    lines.extend(f"- {message}" for _line, _col, message in shown)
    extra = len(sifted.errors) - len(shown)
    if extra > 0:
        lines.append(f"- … and {extra} more")
    if sifted.prior_errors:
        lines.append(unchanged_note(sifted.prior_errors))
    warn_shown = sifted.warnings[:cap]
    if warn_shown or sifted.prior_warnings:
        lines.append(f"Warnings (not blocking) — {len(validation.warnings)}:")
        lines.extend(f"- {message}" for _line, _col, message in warn_shown)
        warn_extra = len(sifted.warnings) - len(warn_shown)
        if warn_extra > 0:
            lines.append(f"- … and {warn_extra} more")
        if sifted.prior_warnings:
            lines.append(unchanged_note(sifted.prior_warnings, "warning"))
    sifted.mark_rendered(shown, warn_shown)
    return "\n".join(lines)


def clean_validation_report(validation, sifted, cap: int = 5) -> str:
    """No-errors confirmation with the riding warnings deduped per agent.

    Mirrors ``Validation.report()`` for the clean case byte-for-byte — the OK
    line (the TUI colors on its ``: OK`` text), then the ``Warnings (not
    blocking) — N:`` list with the same cap and ``… and N more`` summary — so
    the only difference is that rows this agent already saw are summarised.
    Falls back to ``validation.report()`` verbatim when there is no memory or
    nothing to dedup. Shared by every dialect: the clean shape is the one part
    of the section grammar that never differed per surface.
    """
    if sifted is None or not validation.warnings:
        return validation.report()
    if not validation.checked:
        return ""
    base = f"Syntax check ({validation.label}): OK — no syntax errors."
    if not sifted.warnings:  # every warning already shown, unchanged
        return f"{base}\nWarnings (not blocking) — {len(validation.warnings)}: {ALL_UNCHANGED}."
    lines = [base, f"Warnings (not blocking) — {len(validation.warnings)}:"]
    shown = sifted.warnings[:cap]
    lines.extend(f"- {message}" for _line, _col, message in shown)
    extra = len(sifted.warnings) - len(shown)
    if extra > 0:
        lines.append(f"- … and {extra} more")
    if sifted.prior_warnings:
        lines.append(unchanged_note(sifted.prior_warnings, "warning"))
    sifted.mark_rendered(warnings=shown)
    return "\n".join(lines)
