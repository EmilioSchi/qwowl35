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


def _anchor(output: str, line_no: int) -> str:
    match = re.search(rf"^{line_no}:([0-9a-f]{{2}})\|", output, re.MULTILINE)
    if not match:
        raise AssertionError(f"anchor for line {line_no} not found in:\n{output}")
    return f"{line_no}:{match.group(1)}"


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
        shown = tools.execute("read", {"file": "m.txt"})
        assert_true("Skipped re-reading" not in shown, f"first read must run: {shown}")
        assert_true("1:" in shown, f"first read shows anchors: {shown}")

    _in_tmp(body)


def test_identical_full_reread_is_suppressed() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\nc\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("read", {"file": "m.txt"})
        again = tools.execute("read", {"file": "m.txt"})
        assert_true("Skipped re-reading" in again, f"redundant read suppressed: {again}")
        assert_true("unchanged since your last read" in again, f"explains why: {again}")
        assert_true("1:" not in again, f"no file body in a suppressed read: {again}")
        assert_true(not again.startswith(TOOL_ATTENTION_MARKER), "suppression is not an error")

    _in_tmp(body)


def test_force_arg_bypasses_suppression() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\nc\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("read", {"file": "m.txt"})
        forced = tools.execute("read", {"file": "m.txt", "_force": True})
        assert_true("Skipped re-reading" not in forced, f"_force bypasses gate: {forced}")
        assert_true("1:" in forced, f"_force returns the body: {forced}")

    _in_tmp(body)


def test_read_again_after_suppression_serves() -> None:
    # Prompt-free escape: the second redundant read of an unchanged file is served.
    def body() -> None:
        Path("m.txt").write_text("a\nb\nc\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("read", {"file": "m.txt"})
        first = tools.execute("read", {"file": "m.txt"})
        assert_true("Skipped re-reading" in first, "first redundant read suppressed")
        second = tools.execute("read", {"file": "m.txt"})
        assert_true("Skipped re-reading" not in second, f"asking again serves it: {second}")
        assert_true("1:" in second, "served read has the body")

    _in_tmp(body)


def test_anchored_read_always_runs() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\nc\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("read", {"file": "m.txt"})
        anchor = _anchor(shown, 2)
        snippet = tools.execute("read", {"file": "m.txt", "anchor": anchor})
        assert_true("Skipped re-reading" not in snippet, f"anchored read never suppressed: {snippet}")
        assert_true("2:" in snippet, "anchored read returns the snippet")

    _in_tmp(body)


def test_anchored_read_does_not_reset_baseline() -> None:
    # An anchored snippet does not refresh the full-read gate baseline: a follow-up
    # full read of the still-unchanged file is still suppressed, and the model is
    # still considered to hold current anchors.
    def body() -> None:
        Path("m.txt").write_text("a\nb\nc\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("read", {"file": "m.txt"})
        assert_true(tools.has_current_anchors("m.txt"), "holds anchors after full read")
        tools.execute("read", {"file": "m.txt", "anchor": _anchor(shown, 1)})
        assert_true(tools.has_current_anchors("m.txt"), "still holds anchors after anchored read")
        again = tools.execute("read", {"file": "m.txt"})
        assert_true("Skipped re-reading" in again, f"full reread still suppressed: {again}")

    _in_tmp(body)


def test_small_change_under_gate_is_suppressed_with_drift_note() -> None:
    def body() -> None:
        Path("m.txt").write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("read", {"file": "m.txt"})
        # Same-length in-place change → byte delta ~0% < 40% → suppressed, but the
        # note warns anchors near the edit may be stale.
        Path("m.txt").write_text("line1\nlineX\nline3\nline4\nline5\n", encoding="utf-8")
        again = tools.execute("read", {"file": "m.txt"})
        assert_true("Skipped re-reading" in again, f"small change suppressed: {again}")
        assert_true("changed only" in again, f"drift note present: {again}")

    _in_tmp(body)


def test_large_change_over_gate_is_not_suppressed() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("read", {"file": "m.txt"})
        Path("m.txt").write_text("a\nb\nc\nd\ne\nf\ng\n", encoding="utf-8")  # > 40% bigger
        again = tools.execute("read", {"file": "m.txt"})
        assert_true("Skipped re-reading" not in again, f"large change re-read served: {again}")
        assert_true("1:" in again, "served read has the body")

    _in_tmp(body)


def test_edit_invalidates_suppression() -> None:
    # A mutation that changes the file drops the gate baseline so a follow-up full
    # read returns the current file (the common read → edit → re-read flow).
    def body() -> None:
        Path("m.txt").write_text("aaaa\nbbbb\ncccc\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("read", {"file": "m.txt"})
        tools.execute("edit", {"file": "m.txt", "anchor": _anchor(shown, 2), "content": "dddd"})
        reread = tools.execute("read", {"file": "m.txt"})
        assert_true("Skipped re-reading" not in reread, f"reread after edit served: {reread}")
        assert_true("|dddd" in reread, "reread reflects the edit")

    _in_tmp(body)


def test_deleted_file_after_read_is_not_suppressed() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("read", {"file": "m.txt"})
        os.remove("m.txt")
        result = tools.execute("read", {"file": "m.txt"})
        assert_true("Skipped re-reading" not in result, f"deleted file is not suppressed: {result}")
        assert_true(result.startswith("Error"), f"deleted read surfaces the real error: {result}")

    _in_tmp(body)


def test_emptied_file_is_not_suppressed() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\n", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("read", {"file": "m.txt"})
        Path("m.txt").write_text("", encoding="utf-8")
        result = tools.execute("read", {"file": "m.txt"})
        assert_true("Skipped re-reading" not in result, f"emptied file is not suppressed: {result}")

    _in_tmp(body)


def main() -> None:
    test_first_full_read_is_not_suppressed()
    test_identical_full_reread_is_suppressed()
    test_force_arg_bypasses_suppression()
    test_read_again_after_suppression_serves()
    test_anchored_read_always_runs()
    test_anchored_read_does_not_reset_baseline()
    test_small_change_under_gate_is_suppressed_with_drift_note()
    test_large_change_over_gate_is_not_suppressed()
    test_edit_invalidates_suppression()
    test_deleted_file_after_read_is_not_suppressed()
    test_emptied_file_is_not_suppressed()
    print("redundant read gate tests passed")


if __name__ == "__main__":
    main()
