"""Tests for the message-history container (pure, no Textual).

Every test points ``HistoryConfig.file`` at a throwaway temp dir — never the real
cache/home — so nothing on the developer's machine is touched.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from history import HistoryConfig, MessageHistory  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def _cfg(tmp: str, **kw) -> HistoryConfig:
    return HistoryConfig(file=Path(tmp) / "history", **kw)


def test_append_and_load_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp)
        h = MessageHistory(cfg)
        h.append("first")
        h.append("second")
        reloaded = MessageHistory(cfg)
        assert_equal(reloaded.entries, ["first", "second"], "roundtrip")


def test_consecutive_dup_skipped_but_repeats_kept() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        h = MessageHistory(_cfg(tmp))
        h.append("a")
        h.append("a")  # consecutive dup -> dropped
        assert_equal(h.entries, ["a"], "consecutive dup dropped")
        h.append("b")
        h.append("a")  # not consecutive -> kept
        assert_equal(h.entries, ["a", "b", "a"], "non-consecutive repeat kept")


def test_rstrip_newlines_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        h = MessageHistory(_cfg(tmp))
        h.append("x\n\n")
        h.append("a\nb")  # internal newline preserved
        assert_equal(h.entries, ["x", "a\nb"], "rstrip newlines only")


def test_empty_not_appended() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        h = MessageHistory(_cfg(tmp))
        h.append("")
        h.append("\n")
        assert_equal(h.entries, [], "empty submissions dropped")


def test_trim_to_max() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        h = MessageHistory(_cfg(tmp, max_entries=3))
        for c in "abcde":
            h.append(c)
        assert_equal(h.entries, ["c", "d", "e"], "trim keeps last max_entries")


def test_multiline_entry_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp)
        MessageHistory(cfg).append("line1\nline2\nline3")
        assert_equal(
            MessageHistory(cfg).entries, ["line1\nline2\nline3"], "multiline survives reload"
        )


def test_malformed_and_blank_lines_skipped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp)
        cfg.file.write_text('\n"valid"\nnot json\n"also"\n', encoding="utf-8")
        assert_equal(MessageHistory(cfg).entries, ["valid", "also"], "skip blank/malformed")


def test_prev_next_draft_cycle() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        h = MessageHistory(_cfg(tmp))
        h.append("a")
        h.append("b")
        assert_equal(h.prev("draft"), "b", "first prev -> newest")
        assert_equal(h.prev("draft"), "a", "second prev -> older")
        assert_equal(h.prev("draft"), "a", "prev at oldest is idempotent")
        assert_equal(h.next(), "b", "next -> newer")
        assert_equal(h.next(), "draft", "next past newest restores draft")
        assert_equal(h.next(), None, "next while in draft is a no-op")


def test_next_empty_draft_returns_empty_string() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        h = MessageHistory(_cfg(tmp))
        h.append("a")
        assert_equal(h.prev(""), "a", "prev with empty draft")
        assert_equal(h.next(), "", "empty draft restored as '' (not None)")


def test_prev_on_empty_history_returns_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert_equal(MessageHistory(_cfg(tmp)).prev("x"), None, "prev on empty -> None")


def test_next_without_prev_returns_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        h = MessageHistory(_cfg(tmp))
        h.append("a")
        assert_equal(h.next(), None, "next without prior prev -> None")


def test_reset_navigation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        h = MessageHistory(_cfg(tmp))
        h.append("a")
        h.prev("draft")
        h.reset_navigation()
        assert_equal(h.next(), None, "reset drops back to draft mode")


def test_disabled_no_disk_writes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp, enabled=False)
        h = MessageHistory(cfg)
        h.append("a")
        h.append("b")
        assert_equal(h.entries, ["a", "b"], "in-memory append works when disabled")
        assert_true(not cfg.file.exists(), "disabled history never creates the file")


def test_concurrent_append_merges() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp)
        a = MessageHistory(cfg)
        b = MessageHistory(cfg)
        a.append("x")
        b.append("y")  # re-reads under lock, must not clobber "x"
        assert_equal(MessageHistory(cfg).entries, ["x", "y"], "concurrent writes merge")


def test_lazy_creation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp)
        MessageHistory(cfg)  # construction alone must not create anything
        assert_true(not cfg.file.exists(), "construction creates nothing")
        MessageHistory(cfg).append("a")
        assert_true(cfg.file.exists(), "first append creates the file")


def main() -> None:
    test_append_and_load_roundtrip()
    test_consecutive_dup_skipped_but_repeats_kept()
    test_rstrip_newlines_only()
    test_empty_not_appended()
    test_trim_to_max()
    test_multiline_entry_roundtrip()
    test_malformed_and_blank_lines_skipped()
    test_prev_next_draft_cycle()
    test_next_empty_draft_returns_empty_string()
    test_prev_on_empty_history_returns_none()
    test_next_without_prev_returns_none()
    test_reset_navigation()
    test_disabled_no_disk_writes()
    test_concurrent_append_merges()
    test_lazy_creation()
    print("history tests passed")


if __name__ == "__main__":
    main()
