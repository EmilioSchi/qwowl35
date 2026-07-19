"""Tests for the Markdown session export (sessions/export.py).

Run directly: ``python qwowl35/tests/sessions_export_test.py``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sessions.export import main as export_main  # noqa: E402
from sessions.export import render_session_markdown  # noqa: E402
from sessions.restore import load_session  # noqa: E402
from sessions.store import SessionStore  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_in(needle: str, haystack: str, label: str) -> None:
    if needle not in haystack:
        raise AssertionError(f"{label}: {needle!r} not found in output")


def _seed(root: Path) -> SessionStore:
    """One NORMAL turn: assistant text, a tool call + its result, a closer."""
    store = SessionStore(root=root)
    turn = store.begin_turn("fix the parser")
    turn.meta["mode"] = "normal"
    turn.meta["outcome"] = "parser fixed"
    turn.record(
        "assistant",
        content="Running the tests.",
        tool_calls=[{"id": "c1", "name": "bash", "arguments": {"command": "pytest"}}],
    )
    turn.record("tool_result", id="c1", name="bash", result="all tests passed", is_error=False)
    turn.record("assistant", content="Done — parser fixed.", tool_calls=[])
    turn.meta["ok"] = True
    turn.flush_meta()
    store.end_turn(turn, ok=True)
    return store


def test_render_includes_goal_content_and_tool_io() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = _seed(root)
        turns = load_session(root, store.session_hash)
        md = render_session_markdown(turns, title="session")

        assert_in("fix the parser", md, "goal rendered")
        assert_in("Running the tests.", md, "assistant content rendered")
        assert_in("Done — parser fixed.", md, "closing assistant content rendered")
        assert_in("`bash`", md, "tool name rendered")
        assert_in("pytest", md, "tool arguments rendered")
        assert_in("all tests passed", md, "tool result rendered")
        assert_in("parser fixed", md, "outcome rendered")


def test_result_pairs_with_its_call_exactly_once() -> None:
    # The result is rendered inline with the call that issued it (paired by id),
    # never a second time as a standalone block.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = _seed(root)
        turns = load_session(root, store.session_hash)
        md = render_session_markdown(turns)
        assert_true(md.count("all tests passed") == 1, "result rendered exactly once")


def test_cli_writes_markdown_file_for_hash() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = _seed(root)
        out = Path(tmp) / "log.md"
        code = export_main([store.session_hash, "--root", str(root), "-o", str(out)])
        assert_true(code == 0, "cli exit 0")
        assert_in("fix the parser", out.read_text(), "cli output has the goal")


def test_cli_defaults_to_latest_session() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _seed(root)
        out = Path(tmp) / "latest.md"
        code = export_main(["--root", str(root), "-o", str(out)])
        assert_true(code == 0, "cli exit 0 for latest")
        assert_in("fix the parser", out.read_text(), "latest session rendered")


def test_cli_reports_empty_root() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        code = export_main(["--root", str(Path(tmp))])
        assert_true(code == 1, "empty root -> non-zero exit")


def main() -> None:
    test_render_includes_goal_content_and_tool_io()
    test_result_pairs_with_its_call_exactly_once()
    test_cli_writes_markdown_file_for_hash()
    test_cli_defaults_to_latest_session()
    test_cli_reports_empty_root()
    print("sessions export tests passed")


if __name__ == "__main__":
    main()
