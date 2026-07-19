"""Tests for the tree-sitter syntax checker and its tool wiring."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.files.adapter import HashlineTools  # noqa: E402
from tools.syntax import checker  # noqa: E402
from tools.syntax.checker import (  # noqa: E402
    check_bash,
    check_file,
    check_file_structured,
    format_warning_block,
    language_for_path,
    syntax_report,
)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def _anchor(output: str, line_no: int) -> str:
    match = re.search(rf"^{line_no}([0-9a-f]{{2}})\|", output, re.MULTILINE)
    if not match:
        raise AssertionError(f"id for line {line_no} not found in:\n{output}")
    return f"{line_no}{match.group(1)}"


def test_language_for_path() -> None:
    assert_equal(language_for_path("a.py"), "python", ".py → python")
    assert_equal(language_for_path("a.TS"), "typescript", "case-insensitive .ts")
    assert_equal(language_for_path("a.tsx"), "tsx", ".tsx → tsx")
    assert_equal(language_for_path("a.rs"), "rust", ".rs → rust")
    # .h is deliberately omitted (C vs C++ header ambiguity → false positives).
    assert_equal(language_for_path("a.h"), None, ".h is intentionally unmapped")
    assert_equal(language_for_path("notes.md"), None, "unknown extension → None")


def test_python_missing_colon_reports_line() -> None:
    msgs = check_file("m.py", "a = 1\nb = 2\ndef f()\n    return 1\n")
    assert_true(msgs, f"expected a python syntax warning, got {msgs!r}")
    assert_true(any("line 3" in m for m in msgs), f"error should point at line 3: {msgs}")


def test_js_unbalanced_brace_reports_missing() -> None:
    msgs = check_file("app.js", "function f() {\n  return 1;\n")
    assert_true(msgs, f"expected a js syntax warning, got {msgs!r}")
    assert_true(any("missing" in m and "}" in m for m in msgs), f"missing brace: {msgs}")


def test_root_node_error_is_localized() -> None:
    # Some grammars (e.g. YAML) make the *root* the ERROR node; the walk must
    # still report a line rather than falling through to "could not localise".
    msgs = check_file("c.yml", "a: [1, 2\n")
    assert_true(msgs, f"expected a yaml syntax warning, got {msgs!r}")
    assert_true(all("could not localis" not in m for m in msgs), f"should localise: {msgs}")
    assert_true(any("line 1" in m for m in msgs), f"should name a line: {msgs}")


def test_clean_files_have_no_warnings() -> None:
    assert_equal(check_file("m.py", "def f():\n    return 1\n"), [], "valid python")
    assert_equal(check_file("c.json", '{"a": 1, "b": [2, 3]}\n'), [], "valid json")
    assert_equal(check_file("app.ts", "const x: number = 1;\n"), [], "valid ts")


def test_unknown_extension_is_noop() -> None:
    assert_equal(check_file("notes.md", "this is { not ] valid code"), [], "unknown ext → []")


def test_check_bash() -> None:
    bad = check_bash('echo "unterminated')
    assert_true(bad, f"expected a bash syntax warning, got {bad!r}")
    assert_equal(check_bash("echo hi && ls -la"), [], "clean command → []")
    assert_equal(check_bash(""), [], "empty command → []")
    assert_equal(check_bash("   "), [], "whitespace command → []")


def test_format_warning_block() -> None:
    assert_equal(format_warning_block("python", []), "", "no messages → empty block")
    block = format_warning_block("python", ["line 3: missing ':'"])
    assert_true(block.startswith("Syntax check (python) — 1 issue(s):"), f"header: {block}")
    assert_true("- line 3: missing ':'" in block, f"bullet: {block}")
    many = format_warning_block("python", [f"line {i}: e" for i in range(8)])
    assert_true("… and 3 more" in many, f"should summarise overflow: {many}")


def test_large_source_is_skipped() -> None:
    huge_and_broken = "(" * 1_000_001  # over the 1 MB guard
    assert_equal(check_file("big.py", huge_and_broken), [], "oversized source → []")


def test_no_tree_sitter_degrades_gracefully() -> None:
    original = checker._get_parser
    checker._get_parser = lambda language: None  # type: ignore[assignment]
    try:
        assert_equal(check_file("m.py", "def f(\n"), [], "no parser → no file warnings")
        assert_equal(check_bash('echo "x'), [], "no parser → no bash warnings")
    finally:
        checker._get_parser = original  # type: ignore[assignment]


def test_thread_safe_parser_cache() -> None:
    # tree-sitter Parser objects are unsendable across threads (PyO3 panics, and
    # the panic is not an ``Exception`` so it bypasses defensive catches). The file
    # tools run on a thread pool, so the checker caches parsers per-thread. Seed the
    # cache on the main thread, then parse on another — this must not panic.
    import threading

    assert_true(check_file("seed.py", "def f(\n"), "seed main-thread parser cache")
    results: list = []
    errors: list = []

    def worker() -> None:
        try:
            results.append(check_file("w.py", "def g(\n"))
            results.append(check_file("w.py", "def g():\n    return 1\n"))
        except BaseException as exc:  # noqa: BLE001 - catch the PyO3 panic too
            errors.append(repr(exc))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert_equal(errors, [], "no cross-thread parser panic")
    assert_true(results and results[0], "worker detected the broken file")
    assert_equal(results[1], [], "worker parsed the clean file")


def test_read_appends_syntax_block() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("broken.py").write_text("def f()\n    return 1\n", encoding="utf-8")
            tools = HashlineTools()
            shown = tools.execute("read_file", {"file_path": os.path.abspath("broken.py")})
            assert_true("issue(s)" in shown and "Syntax check (python)" in shown, f"read should warn: {shown}")
            # A clean recognized file gets a positive OK confirmation.
            Path("ok.py").write_text("def g():\n    return 1\n", encoding="utf-8")
            clean = tools.execute("read_file", {"file_path": os.path.abspath("ok.py")})
            assert_true("Syntax check (python): OK" in clean, f"clean read should confirm OK: {clean}")
            assert_true("issue(s)" not in clean, f"clean read has no error list: {clean}")
            # Unknown language gets neither warning nor OK line.
            Path("notes.md").write_text("# hi {[(\n", encoding="utf-8")
            md = tools.execute("read_file", {"file_path": os.path.abspath("notes.md")})
            assert_true("Syntax check" not in md, f"unknown ext stays silent: {md}")
        finally:
            os.chdir(cwd)


def test_edit_introducing_error_warns_and_clean_edit_does_not() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            tools = HashlineTools()
            shown = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})

            # Clean edit: line 2 stays valid → no syntax block.
            line2 = _anchor(shown, 2)
            clean = tools.execute(
                "replace",
                {"file": "m.py", "id": line2, "content": "    return 2"},
            )
            assert_true("Syntax check (python): OK" in clean, f"clean edit should confirm OK: {clean}")

            # Breaking edit: drop the colon on line 1 → file no longer parses.
            reread = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})
            line1 = _anchor(reread, 1)
            broken = tools.execute(
                "replace",
                {"file": "m.py", "id": line1, "content": "def f("},
            )
            assert_true("issue(s)" in broken, f"breaking edit should list issues: {broken}")
        finally:
            os.chdir(cwd)


def test_check_file_structured_returns_positions() -> None:
    errors = check_file_structured("m.py", "a = 1\nb = 2\ndef f()\n    return 1\n")
    assert_true(errors, f"expected structured errors, got {errors!r}")
    first = errors[0]
    assert_equal(len(first), 3, "each error is (line, col, message)")
    line, col, message = first
    assert_true(isinstance(line, int) and isinstance(col, int), "line/col are ints")
    assert_true(any(e[0] == 3 for e in errors), f"error should point at line 3: {errors}")
    assert_true(isinstance(message, str) and message, "message is a non-empty string")
    # Clean / unknown / oversized → empty, mirroring check_file.
    assert_equal(check_file_structured("m.py", "def f():\n    return 1\n"), [], "clean → []")
    assert_equal(check_file_structured("notes.md", "{ not ] code"), [], "unknown ext → []")
    assert_equal(check_file_structured("big.py", "(" * 1_000_001), [], "oversized → []")


def test_structured_and_string_checkers_agree() -> None:
    source = "def f()\n    return 1\n"
    structured = check_file_structured("m.py", source)
    strings = check_file("m.py", source)
    assert_equal(len(structured), len(strings), "same issue count both ways")
    # The structured message text matches the string-checker message.
    assert_equal([e[2] for e in structured], strings, "structured messages equal string messages")


def test_syntax_report_confirms_clean_and_skips_unknown() -> None:
    assert_true("OK" in syntax_report("m.py", "def f():\n    return 1\n"), "clean python → OK")
    assert_true("issue(s)" in syntax_report("m.py", "def f(\n"), "broken python → issues")
    assert_equal(syntax_report("notes.md", "x { ["), "", "unknown language → no report")
    assert_equal(syntax_report("big.py", "(" * 1_000_001), "", "too large → no report (not OK)")


def main() -> None:
    test_language_for_path()
    test_python_missing_colon_reports_line()
    test_js_unbalanced_brace_reports_missing()
    test_root_node_error_is_localized()
    test_clean_files_have_no_warnings()
    test_unknown_extension_is_noop()
    test_check_bash()
    test_format_warning_block()
    test_large_source_is_skipped()
    test_no_tree_sitter_degrades_gracefully()
    test_thread_safe_parser_cache()
    test_read_appends_syntax_block()
    test_edit_introducing_error_warns_and_clean_edit_does_not()
    test_check_file_structured_returns_positions()
    test_structured_and_string_checkers_agree()
    test_syntax_report_confirms_clean_and_skips_unknown()
    print("tools/syntax tests passed")


if __name__ == "__main__":
    main()
