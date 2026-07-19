"""Tests for the diagnostics presentation layer (tools/diagnostics).

Run directly: ``python qwowl35/tests/diagnostics_test.py``. Pure functions and
in-memory state — no LSP, no disk beyond nothing at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.diagnostics import (  # noqa: E402
    ALL_UNCHANGED,
    DiagnosticsMemory,
    clean_validation_report,
    join_section,
    split_trailing_section,
    unchanged_note,
    validation_report_with_memory,
)
from tools.syntax import Validation  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


# --- section grammar ---------------------------------------------------------


def test_split_returns_text_verbatim_without_section() -> None:
    body = "def main():\n    return 1"
    assert_equal(split_trailing_section(body), (body, ""), "no section, no split")
    assert_equal(split_trailing_section(""), ("", ""), "empty text")


def test_split_and_join_round_trip() -> None:
    body = "def main():\n    return compute()"
    section = "Syntax check (python, lsp) — 1 issue(s):\n- line 2, col 12: boom (pylint)"
    text = join_section(body, section)
    assert_equal(text, f"{body}\n\n{section}", "canonical join is one blank line")
    assert_equal(split_trailing_section(text), (body, section), "split restores parts")
    lsp = "LSP diagnostics (python, lsp) — 1 error(s), 0 warning(s):\n- line 3: x"
    assert_equal(
        split_trailing_section(join_section(body, lsp)), (body, lsp), "both prefixes split"
    )


def test_split_section_opening_the_text() -> None:
    section = "Syntax check (python): OK — no syntax errors."
    assert_equal(split_trailing_section(section), ("", section), "section-only text")
    assert_equal(join_section("", section), section, "join with empty body")


def test_split_ignores_indented_and_mid_paragraph_headers() -> None:
    # Headers quoted inside code (indented) or mid-paragraph (no preceding
    # blank line) never start a section.
    body = (
        'print("Syntax check (demo) — 1 issue(s):")\n'
        "text above\nSyntax check (not after a blank line)"
    )
    assert_equal(split_trailing_section(body), (body, ""), "no false section")
    indented = "code\n\n    LSP diagnostics (python) — quoted, indented"
    assert_equal(split_trailing_section(indented), (indented, ""), "indented ignored")


def test_split_takes_the_last_candidate() -> None:
    # A file body that *documents* the grammar must not truncate the real
    # trailing section: the LAST candidate wins.
    body = (
        "docs about the block:\n\n"
        "Syntax check (example) — quoted in a README\n"
        "more prose"
    )
    real = "Syntax check (python, lsp) — 2 issue(s):\n- line 1, col 1: a\n- line 9, col 2: b"
    text = join_section(body, real)
    got_body, got_section = split_trailing_section(text)
    assert_equal(got_section, real, "the real (last) section is carved")
    assert_equal(got_body, body, "the documenting body stays intact")


# --- per-agent memory --------------------------------------------------------


_SOURCE = "import os\nx = foo\ny = bar\n"


def _validation(errors, warnings=()):
    return Validation(list(errors), list(warnings), "python, lsp", True)


def test_sift_partitions_and_only_rendered_rows_are_remembered() -> None:
    memory = DiagnosticsMemory()
    errors = [
        (2, 5, "line 2, col 5: Undefined name 'foo' (Pyflakes)"),
        (3, 5, "line 3, col 5: Undefined name 'bar' (Pyflakes)"),
    ]
    first = memory.sift("/w/app.py", _validation(errors), _SOURCE)
    assert_equal(first.errors, errors, "everything new on first sight")
    assert_equal(first.prior_errors, 0, "nothing prior yet")
    # Only the first row was actually listed (a cap, say) — the second must
    # still count as unseen next time.
    first.mark_rendered(errors[:1])
    second = memory.sift("/w/app.py", _validation(errors), _SOURCE)
    assert_equal(second.errors, errors[1:], "unrendered row still new")
    assert_equal(second.prior_errors, 1, "rendered row suppressed")
    second.mark_rendered(second.errors)
    third = memory.sift("/w/app.py", _validation(errors), _SOURCE)
    assert_true(third.all_prior, "all rows suppressed once all were rendered")
    assert_equal(third.prior_errors, 2, "count stays honest")


def test_fixed_rows_are_evicted_so_regressions_reshow() -> None:
    memory = DiagnosticsMemory()
    row = (2, 5, "line 2, col 5: Undefined name 'foo' (Pyflakes)")
    sifted = memory.sift("/w/app.py", _validation([row]), _SOURCE)
    sifted.mark_rendered(sifted.errors)
    # The issue disappears (fixed) — the seen-set prunes it…
    memory.sift("/w/app.py", _validation([]), _SOURCE)
    # …so the very same row re-shows when it regresses later.
    again = memory.sift("/w/app.py", _validation([row]), _SOURCE)
    assert_equal(again.errors, [row], "regressed row is new again")
    assert_equal(again.prior_errors, 0, "not suppressed after eviction")


def test_changed_anchor_content_reshows_the_row() -> None:
    memory = DiagnosticsMemory()
    row = (2, 5, "line 2, col 5: Undefined name 'foo' (Pyflakes)")
    sifted = memory.sift("/w/app.py", _validation([row]), _SOURCE)
    sifted.mark_rendered(sifted.errors)
    # Same message, but line 2's content changed (an edit rewrote it): the
    # hashline id the agent holds is stale, so the row must re-show.
    edited = "import os\nx = foo  # tweaked\ny = bar\n"
    again = memory.sift("/w/app.py", _validation([row]), edited)
    assert_equal(again.errors, [row], "row re-shows with fresh anchor")
    unchanged = memory.sift("/w/app.py", _validation([row]), edited)
    assert_true(unchanged.errors == [row], "not marked: previous sift never rendered")
    unchanged.mark_rendered(unchanged.errors)
    assert_true(
        memory.sift("/w/app.py", _validation([row]), edited).all_prior,
        "suppressed once rendered against the new content",
    )


def test_memory_keys_per_file_and_clear_forgets() -> None:
    memory = DiagnosticsMemory()
    row = (1, 1, "line 1, col 1: boom (pylint)")
    a = memory.sift("/w/a.py", _validation([row]), "boom()\n")
    a.mark_rendered(a.errors)
    b = memory.sift("/w/b.py", _validation([row]), "boom()\n")
    assert_equal(b.errors, [row], "files do not cross-suppress")
    memory.clear()
    again = memory.sift("/w/a.py", _validation([row]), "boom()\n")
    assert_equal(again.errors, [row], "clear() forgets everything")


def test_warnings_partition_independently() -> None:
    memory = DiagnosticsMemory()
    err = (2, 5, "line 2, col 5: Undefined name 'foo' (Pyflakes)")
    warn = (1, 1, "line 1, col 1: 'os' imported but unused (Pyflakes)")
    sifted = memory.sift("/w/app.py", _validation([err], [warn]), _SOURCE)
    sifted.mark_rendered([err], [warn])
    again = memory.sift("/w/app.py", _validation([err], [warn]), _SOURCE)
    assert_true(again.all_prior, "errors and warnings both suppressed")
    assert_equal(
        (again.prior_errors, again.prior_warnings), (1, 1), "counts split by kind"
    )


# --- report builders ---------------------------------------------------------


def test_first_pass_report_matches_plain_validation_report() -> None:
    # With a fresh memory the deduped report must be byte-identical to
    # Validation.report() — rendering is unchanged until something repeats.
    errors = [(i, 1, f"line {i}, col 1: boom {i} (pylint)") for i in range(1, 8)]
    warnings = [(1, 2, "line 1, col 2: unused (pylint)")]
    v = _validation(errors, warnings)
    sifted = DiagnosticsMemory().sift("/w/app.py", v, _SOURCE)
    assert_equal(
        validation_report_with_memory(v, sifted), v.report(), "first pass identical"
    )
    clean = _validation([], warnings)
    sifted_clean = DiagnosticsMemory().sift("/w/app.py", clean, _SOURCE)
    assert_equal(
        clean_validation_report(clean, sifted_clean),
        clean.report(),
        "clean first pass identical",
    )
    assert_equal(
        validation_report_with_memory(v, None), v.report(), "no memory → verbatim"
    )


def test_repeat_report_collapses_to_honest_one_liner() -> None:
    errors = [(2, 5, "line 2, col 5: Undefined name 'foo' (Pyflakes)")]
    v = _validation(errors)
    memory = DiagnosticsMemory()
    validation_report_with_memory(v, memory.sift("/w/app.py", v, _SOURCE))
    repeat = validation_report_with_memory(v, memory.sift("/w/app.py", v, _SOURCE))
    assert_equal(
        repeat,
        f"Syntax check (python, lsp) — 1 issue(s), {ALL_UNCHANGED}.",
        "all-unchanged single line keeps the header prefix and honest count",
    )


def test_partial_repeat_lists_only_new_rows_with_note() -> None:
    old = (2, 5, "line 2, col 5: Undefined name 'foo' (Pyflakes)")
    new = (3, 5, "line 3, col 5: Undefined name 'bar' (Pyflakes)")
    memory = DiagnosticsMemory()
    v1 = _validation([old])
    validation_report_with_memory(v1, memory.sift("/w/app.py", v1, _SOURCE))
    v2 = _validation([old, new])
    report = validation_report_with_memory(v2, memory.sift("/w/app.py", v2, _SOURCE))
    assert_true(report.startswith("Syntax check (python, lsp) — 2 issue(s):"), report)
    assert_true(f"- {new[2]}" in report, "new row listed")
    assert_true(f"- {old[2]}" not in report, "old row suppressed")
    assert_true(unchanged_note(1) in report, "suppressed rows summarised")


def main() -> None:
    test_split_returns_text_verbatim_without_section()
    test_split_and_join_round_trip()
    test_split_section_opening_the_text()
    test_split_ignores_indented_and_mid_paragraph_headers()
    test_split_takes_the_last_candidate()
    test_sift_partitions_and_only_rendered_rows_are_remembered()
    test_fixed_rows_are_evicted_so_regressions_reshow()
    test_changed_anchor_content_reshows_the_row()
    test_memory_keys_per_file_and_clear_forgets()
    test_warnings_partition_independently()
    test_first_pass_report_matches_plain_validation_report()
    test_repeat_report_collapses_to_honest_one_liner()
    test_partial_repeat_lists_only_new_rows_with_note()
    print("diagnostics_test: all tests passed")


if __name__ == "__main__":
    main()
