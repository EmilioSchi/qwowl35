"""Tests for F2 (syntax error → edit anchors + attention marker) and F3
(post-mutation auto-delete of adjacent identical lines)."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.files import hashline  # noqa: E402,F401  (package import for monkeypatch path)
from tools.files.adapter import HashlineTools, TOOL_ATTENTION_MARKER  # noqa: E402
from tools.files.hashline import tool_calling  # noqa: E402


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


def _in_tmp(fn):
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            fn()
        finally:
            os.chdir(cwd)


# --- F2: syntax errors as marked, anchored, fixable rows --------------------


def test_syntax_error_read_is_marked_and_anchored() -> None:
    def body() -> None:
        Path("broken.py").write_text("a = 1\nb = 2\ndef f()\n    return 1\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("beginTransaction", {"file": "broken.py"})
        assert_true(shown.startswith(TOOL_ATTENTION_MARKER), f"syntax error flags attention: {shown!r}")
        clean = shown[len(TOOL_ATTENTION_MARKER):]
        assert_true("Syntax check (python)" in clean and "issue(s)" in clean, f"keeps issue header: {clean}")
        # The error row (line 3) is offered as a ready edit anchor.
        assert_true(re.search(r"edit id: 3[0-9a-f]{2}\|", clean), f"line-3 edit id present: {clean}")

    _in_tmp(body)


def test_clean_read_has_no_marker_and_keeps_ok() -> None:
    def body() -> None:
        Path("ok.py").write_text("def g():\n    return 1\n", encoding="utf-8")
        tools = HashlineTools()
        clean = tools.execute("beginTransaction", {"file": "ok.py"})
        assert_true(not clean.startswith(TOOL_ATTENTION_MARKER), "clean read is not flagged")
        assert_true("Syntax check (python): OK" in clean, f"clean read confirms OK: {clean}")

    _in_tmp(body)


def test_mutation_introducing_error_is_marked_and_anchored() -> None:
    def body() -> None:
        Path("m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("beginTransaction", {"file": "m.py"})
        broken = tools.execute("edit", {"file": "m.py", "id": _anchor(shown, 1), "content": "def f("})
        assert_true(broken.startswith(TOOL_ATTENTION_MARKER), f"breaking edit flags attention: {broken!r}")
        body_text = broken[len(TOOL_ATTENTION_MARKER):]
        assert_true("issue(s)" in body_text, f"lists the issue: {body_text}")
        assert_true("edit id:" in body_text, f"offers an edit id: {body_text}")

    _in_tmp(body)


# --- F3: post-mutation auto-delete of adjacent identical lines ---------------


def test_edit_leaving_duplicate_is_autodeleted() -> None:
    def body() -> None:
        Path("m.py").write_text("x = 1\nx = 2\ny = 3\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("beginTransaction", {"file": "m.py"})
        # Edit line 2 to equal line 1 → an adjacent duplicate the pass removes.
        result = tools.execute("edit", {"file": "m.py", "id": _anchor(shown, 2), "content": "x = 1"})
        assert_true("Removed 1 adjacent duplicate line(s): 2." in result, f"reports the removal: {result}")
        assert_equal(Path("m.py").read_text(encoding="utf-8"), "x = 1\ny = 3\n", "duplicate deleted on disk")

    _in_tmp(body)


def test_read_does_not_autodelete() -> None:
    def body() -> None:
        Path("d.txt").write_text("a\na\nb\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("beginTransaction", {"file": "d.txt"})
        assert_equal(Path("d.txt").read_text(encoding="utf-8"), "a\na\nb\n", "read must not mutate the file")

    _in_tmp(body)


def test_all_exact_collapses_blank_double_line() -> None:
    # Exercises the (non-default) all_exact branch: it collapses a PEP8 double-blank
    # run. The shipped default is "smart", so this test pins the policy explicitly.
    def body() -> None:
        original = tool_calling.DUP_POLICY
        tool_calling.DUP_POLICY = "all_exact"
        try:
            Path("m.py").write_text("x = 1\n\n\ny = 2\n", encoding="utf-8")
            tools = HashlineTools()
            shown = tools.execute("beginTransaction", {"file": "m.py"})
            tools.execute("edit", {"file": "m.py", "id": _anchor(shown, 1), "content": "x = 0"})
            assert_equal(Path("m.py").read_text(encoding="utf-8"), "x = 0\n\ny = 2\n", "all_exact collapses blanks")
        finally:
            tool_calling.DUP_POLICY = original

    _in_tmp(body)


def test_dup_policy_smart_spares_blanks_but_drops_real_dups() -> None:
    def body() -> None:
        original = tool_calling.DUP_POLICY
        tool_calling.DUP_POLICY = "smart"
        try:
            # Blank double line is spared; a real duplicate content line is removed.
            Path("m.py").write_text("x = 1\n\n\nvalue = 42\nvalue = 42\n", encoding="utf-8")
            tools = HashlineTools()
            shown = tools.execute("beginTransaction", {"file": "m.py"})
            tools.execute("edit", {"file": "m.py", "id": _anchor(shown, 1), "content": "x = 0"})
            text = Path("m.py").read_text(encoding="utf-8")
            assert_true("\n\n\n" in text, f"smart policy spares the blank double line: {text!r}")
            assert_equal(text.count("value = 42"), 1, "smart policy still drops a real duplicate")
        finally:
            tool_calling.DUP_POLICY = original

    _in_tmp(body)


def test_smart_default_spares_blanks_and_brackets() -> None:
    # Locks the shipped default (DUP_POLICY == "smart"): blank double-lines and
    # single-bracket lines survive an auto-dedup pass while a real duplicate collapses.
    def body() -> None:
        assert_equal(tool_calling.DUP_POLICY, "smart", "smart is the shipped default")
        # 1:a=1  2:blank  3:blank  4:}  5:}  6:dup  7:dup  (.txt skips the syntax check)
        Path("t.txt").write_text("a = 1\n\n\n}\n}\ndup\ndup\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("beginTransaction", {"file": "t.txt"})
        result = tools.execute("edit", {"file": "t.txt", "id": _anchor(shown, 1), "content": "a = 0"})
        assert_equal(
            Path("t.txt").read_text(encoding="utf-8"),
            "a = 0\n\n\n}\n}\ndup\n",
            "smart spares blank double-line and stacked single brackets, drops the real duplicate",
        )
        assert_true("Removed 1 adjacent duplicate line(s)" in result, f"only the real dup is reported: {result}")

    _in_tmp(body)


def test_combined_syntax_error_and_duplicate() -> None:
    def body() -> None:
        Path("m.py").write_text("def f():\n    return 1\n    return 1\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("beginTransaction", {"file": "m.py"})
        # Break the header (syntax error) — the duplicate body lines are auto-removed.
        result = tools.execute("edit", {"file": "m.py", "id": _anchor(shown, 1), "content": "def f("})
        assert_true(result.startswith(TOOL_ATTENTION_MARKER), "combined result is flagged")
        text = result[len(TOOL_ATTENTION_MARKER):]
        assert_true("Removed 1 adjacent duplicate line(s)" in text, f"dedup reported: {text}")
        assert_true("Syntax check (python)" in text, f"syntax block present: {text}")
        assert_true(
            text.index("Removed") < text.index("Syntax check"),
            "dedup note precedes the trailing syntax block",
        )

    _in_tmp(body)


def main() -> None:
    test_syntax_error_read_is_marked_and_anchored()
    test_clean_read_has_no_marker_and_keeps_ok()
    test_mutation_introducing_error_is_marked_and_anchored()
    test_edit_leaving_duplicate_is_autodeleted()
    test_read_does_not_autodelete()
    test_all_exact_collapses_blank_double_line()
    test_dup_policy_smart_spares_blanks_but_drops_real_dups()
    test_smart_default_spares_blanks_and_brackets()
    test_combined_syntax_error_and_duplicate()
    print("file tool attention tests passed")


if __name__ == "__main__":
    main()
