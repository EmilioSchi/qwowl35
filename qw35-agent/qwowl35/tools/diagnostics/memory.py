"""Per-agent memory of diagnostic lines already shown.

The same broken file is validated on every read and every mutation, so without
memory the identical ``- line 32, col 61: [no-member] …`` rows flood an agent's
context on each tool call. This store remembers, per agent instance, which
rendered diagnostic rows that agent has already seen and lets the section
builders list only the new ones (headline counts always stay current, so a
still-broken file is never mistaken for a clean one).

Scoping is the caller's job: one ``DiagnosticsMemory`` per running agent
instance, following its *context* lifetime — a freshly spawned editor or
explorer gets a fresh memory (its context holds no earlier diagnostics), the
PLAN executors share one for the lifetime of their shared context, and /clear
resets everything.

Identity of a diagnostic row is ``(file, message, anchored line content)``:

- ``message`` already embeds line, column, text, and source — an issue that
  moves to another line is genuinely new information and re-shows;
- the anchored line's current content is included so an edit that rewrites the
  diagnosed line while leaving the same message re-shows too. This matters for
  the hashline dialect, whose ``edit id:`` echo hashes the line content: a
  suppressed row must never strand the editor with a stale id.

Rows that disappear from a validation (the issue got fixed) are evicted, so a
later regression of the very same row is reported again — and the store stays
bounded by the number of live diagnostics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

# (line, col, message) — the tuple shape produced by tools/syntax/validate.
Entry = tuple[int, int, str]


class HasDiagnostics(Protocol):
    """Anything shaped like ``tools.syntax.validate.Validation``."""

    errors: list[Entry]
    warnings: list[Entry]


class DiagnosticsMemory:
    """What one agent instance has already been shown, keyed per file."""

    def __init__(self) -> None:
        self._seen: dict[str, set[str]] = {}

    def clear(self) -> None:
        """Forget everything (the owning agent's context was reset)."""
        self._seen.clear()

    def sift(self, file: str, validation: HasDiagnostics, source_text: str) -> "SiftedDiagnostics":
        """Split ``validation`` into new vs already-shown rows for ``file``.

        Prunes rows no longer reported (fixed issues re-arm), then partitions
        the current errors and warnings. Nothing is marked as shown yet — the
        section builder decides what it actually renders (caps apply) and calls
        :meth:`SiftedDiagnostics.mark_rendered` with exactly that subset, so a
        row summarised as "… and N more" still counts as unseen next time.
        """
        file = self._normalize(file)
        lines = source_text.splitlines()

        def anchor(line_no: int) -> str:
            return lines[line_no - 1] if 1 <= line_no <= len(lines) else ""

        seen = self._seen.setdefault(file, set())
        current = {
            self._key(message, anchor(line_no))
            for line_no, _col, message in [*validation.errors, *validation.warnings]
        }
        seen.intersection_update(current)

        def partition(entries: Sequence[Entry]) -> tuple[list[Entry], int]:
            new: list[Entry] = []
            prior = 0
            for entry in entries:
                if self._key(entry[2], anchor(entry[0])) in seen:
                    prior += 1
                else:
                    new.append(entry)
            return new, prior

        new_errors, prior_errors = partition(validation.errors)
        new_warnings, prior_warnings = partition(validation.warnings)
        return SiftedDiagnostics(
            errors=new_errors,
            warnings=new_warnings,
            prior_errors=prior_errors,
            prior_warnings=prior_warnings,
            _seen=seen,
            _anchor=anchor,
        )

    @staticmethod
    def _key(message: str, anchor_content: str) -> str:
        return f"{message}\x00{anchor_content.rstrip()}"

    @staticmethod
    def _normalize(file: str) -> str:
        # The same file may arrive as given by the model (hashline) or resolved
        # (inspect_file); one canonical key keeps cross-surface dedup working.
        return os.path.abspath(file)


@dataclass
class SiftedDiagnostics:
    """One validation pass partitioned into new vs previously shown rows."""

    errors: list[Entry]
    warnings: list[Entry]
    prior_errors: int
    prior_warnings: int
    _seen: set[str] = field(repr=False, default_factory=set)
    _anchor: Callable[[int], str] = field(repr=False, default=lambda _line: "")

    @property
    def all_prior(self) -> bool:
        """True when something was suppressed and nothing new remains."""
        return not self.errors and not self.warnings and (
            self.prior_errors > 0 or self.prior_warnings > 0
        )

    def mark_rendered(
        self, errors: Sequence[Entry] = (), warnings: Sequence[Entry] = ()
    ) -> None:
        """Record the rows a section builder actually listed in full."""
        for line_no, _col, message in [*errors, *warnings]:
            self._seen.add(DiagnosticsMemory._key(message, self._anchor(line_no)))
