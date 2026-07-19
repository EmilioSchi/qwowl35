"""Tests for the F1 redundant full-read gate (byte-size 40% threshold)."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.files.adapter import HashlineTools, TOOL_ATTENTION_MARKER  # noqa: E402


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def _line_id(output: str, line_no: int) -> str:
    match = re.search(rf"^{line_no}([0-9a-f]{{2}})\|", output, re.MULTILINE)
    if not match:
        raise AssertionError(f"id for line {line_no} not found in:\n{output}")
    return f"{line_no}{match.group(1)}"


def _has_body(output: str) -> bool:
    """Whether the output contains at least one rendered ``<line><hash>|`` id row."""
    return re.search(r"(?m)^\d+[0-9a-f]{2}\|", output) is not None


def _in_tmp(fn):
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            fn()
        finally:
            os.chdir(cwd)


def test_first_full_read_is_not_suppressed() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\nc\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        assert_true("Skipped re-opening" not in shown, f"first open must run: {shown}")
        assert_true(_has_body(shown), f"first open shows ids: {shown}")

    _in_tmp(body)


def test_identical_full_reread_is_always_served() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\nc\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        again = tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        assert_true("Skipped re-opening" not in again, f"redundant open always served: {again}")
        assert_true(_has_body(again), f"re-read returns the body: {again}")

    _in_tmp(body)


def test_force_arg_bypasses_suppression() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\nc\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        forced = tools.execute("read_file", {"file_path": "m.txt", "_force": True})
        assert_true("Skipped re-opening" not in forced, f"_force bypasses gate: {forced}")
        assert_true(_has_body(forced), f"_force returns the body: {forced}")

    _in_tmp(body)


def test_repeated_full_read_always_serves() -> None:
    # Every read_file returns the file body — no suppression gate.
    def body() -> None:
        Path("m.txt").write_text("a\nb\nc\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        first = tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        assert_true("Skipped re-opening" not in first, "first re-read always served")
        assert_true(_has_body(first), "first re-read has the body")
        second = tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        assert_true("Skipped re-opening" not in second, f"third read also served: {second}")
        assert_true(_has_body(second), "third read has the body")

    _in_tmp(body)


def test_small_change_is_always_served() -> None:
    def body() -> None:
        Path("m.txt").write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        # Same-length in-place change → always served (no suppression gate).
        Path("m.txt").write_text("line1\nlineX\nline3\nline4\nline5\n", encoding="utf-8")
        again = tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        assert_true("Skipped re-opening" not in again, f"small change always served: {again}")
        assert_true(_has_body(again), "re-read returns the body with fresh ids")

    _in_tmp(body)


def test_large_change_over_gate_is_not_suppressed() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        Path("m.txt").write_text("a\nb\nc\nd\ne\nf\ng\n", encoding="utf-8")  # > 40% bigger
        again = tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        assert_true("Skipped re-opening" not in again, f"large change re-open served: {again}")
        assert_true(_has_body(again), "served open has the body")

    _in_tmp(body)


def test_edit_invalidates_suppression() -> None:
    # A mutation that changes the file drops the gate baseline so a follow-up
    # read_file returns the current file (the common open → edit → re-open flow).
    def body() -> None:
        Path("m.txt").write_text("aaaa\nbbbb\ncccc\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        tools.execute("replace",{"file": "m.txt", "id": _line_id(shown, 2), "content": "dddd"})
        reread = tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        assert_true("Skipped re-opening" not in reread, f"re-open after edit served: {reread}")
        assert_true("|dddd" in reread, "re-open reflects the edit")

    _in_tmp(body)


def test_deleted_file_after_read_is_not_suppressed() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        os.remove("m.txt")
        result = tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        assert_true("Skipped re-opening" not in result, f"deleted file is not suppressed: {result}")
        assert_true(result.startswith("Error"), f"deleted open surfaces the real error: {result}")

    _in_tmp(body)


def test_emptied_file_is_not_suppressed() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        Path("m.txt").write_text("", encoding="utf-8")
        result = tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        assert_true("Skipped re-opening" not in result, f"emptied file is not suppressed: {result}")

    _in_tmp(body)


def main() -> None:
    test_first_full_read_is_not_suppressed()
    test_identical_full_reread_is_always_served()
    test_force_arg_bypasses_suppression()
    test_repeated_full_read_always_serves()
    test_small_change_is_always_served()
    test_large_change_over_gate_is_not_suppressed()
    test_edit_invalidates_suppression()
    test_deleted_file_after_read_is_not_suppressed()
    test_emptied_file_is_not_suppressed()
    print("redundant read gate tests passed")


if __name__ == "__main__":
    main()
