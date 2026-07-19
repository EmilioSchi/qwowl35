"""Tests for the session read side: summaries, turn loading, orchestrator
rehydration, and the display replay.

Run directly: ``python qwowl35/tests/sessions_restore_test.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402
from modes import Mode  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from sessions.replay import replay_into_chat  # noqa: E402
from sessions.restore import list_session_summaries, load_session  # noqa: E402
from sessions.store import SessionStore  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


CHAT_DELTA = [
    {"role": "user", "content": "hi there"},
    {"role": "assistant", "content": "hello!"},
]


def _seed_session(root: Path, age_seconds: float = 0.0) -> SessionStore:
    """A realistic two-turn session: one CHAT turn, one NORMAL turn with
    display records, plus an incomplete third turn dir (crash mid-turn)."""
    store = SessionStore(root=root)

    chat_turn = store.begin_turn("hi there")
    chat_turn.meta["mode"] = "chat"
    chat_turn.meta["outcome"] = "hello!"
    chat_turn.meta["chat_messages"] = CHAT_DELTA
    chat_turn.record("assistant", content="hello!", tool_calls=[])
    chat_turn.meta["ok"] = True
    chat_turn.flush_meta()
    store.end_turn(chat_turn, ok=True)

    work_turn = store.begin_turn("fix the bug")
    work_turn.meta["mode"] = "normal"
    work_turn.meta["outcome"] = "patched agent.py"
    work_turn.record("assistant", content="On it.", tool_calls=[])
    work_turn.record(
        "tool_result",
        id="c1",
        name="run_shell_command",
        result="tests passed\nall green",
        is_error=False,
    )
    work_turn.record("assistant", content="Done, tests pass.", tool_calls=[])
    work_turn.meta["ok"] = True
    work_turn.flush_meta()
    store.end_turn(work_turn, ok=True)

    (store.session_dir / "turns" / "0003").mkdir()

    if age_seconds:
        meta_path = store.session_dir / "session.json"
        meta = json.loads(meta_path.read_text())
        meta["last_active_ts"] = time.time() - age_seconds
        meta_path.write_text(json.dumps(meta))
    return store


class RecordingChat:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def add_user(self, text: str) -> None:
        self.calls.append(("user", text))

    def add_system(self, text: str) -> None:
        self.calls.append(("system", text))

    def add_assistant_chunk(self, text: str) -> None:
        self.calls.append(("assistant_chunk", text))

    def flush_assistant(self) -> None:
        self.calls.append(("flush_assistant",))


class FakeUI:
    def __init__(self) -> None:
        self.chat = RecordingChat()

    def set_state(self, *a, **k) -> None: ...
    def set_warning(self, *a, **k) -> None: ...
    def set_mode(self, *a, **k) -> None: ...


def test_summaries_newest_first_and_filtered() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        older = _seed_session(root, age_seconds=3600)
        newer = _seed_session(root)

        empty = SessionStore(root=root)
        empty._upsert_session_meta()

        summaries = list_session_summaries(root)
        hashes = [s.session_hash for s in summaries]
        assert_equal(hashes, [newer.session_hash, older.session_hash],
                     "newest first, empty session skipped")
        assert_equal(summaries[0].first_goal, "hi there", "first goal surfaced")
        assert_equal(summaries[0].turn_count, 2, "turn count surfaced")
        assert_equal(summaries[0].last_mode, "normal", "last mode surfaced")

        summaries = list_session_summaries(root, exclude=newer.session_hash)
        assert_equal([s.session_hash for s in summaries], [older.session_hash],
                     "exclude drops the live session")


def test_summaries_fall_back_without_session_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = _seed_session(root)
        (store.session_dir / "session.json").unlink()

        summaries = list_session_summaries(root)
        assert_equal(len(summaries), 1, "session still listed")
        assert_equal(summaries[0].first_goal, "hi there",
                     "first goal recovered from turn meta")
        assert_equal(summaries[0].turn_count, 2,
                     "turn count recovered by counting meta files")


def test_load_session_ordered_and_tolerant() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = _seed_session(root)

        turns = load_session(root, store.session_hash)
        assert_equal(len(turns), 2, "incomplete turn dir skipped")
        assert_equal([t.goal for t in turns], ["hi there", "fix the bug"],
                     "turns in conversation order")
        assert_equal(turns[0].chat_messages, CHAT_DELTA, "chat delta verbatim")
        assert_equal(turns[1].chat_messages, [], "non-chat turn has no delta")
        assert_equal(turns[1].outcome, "patched agent.py", "outcome loaded")
        kinds = [rec["kind"] for rec in turns[1].display_records]
        assert_equal(kinds, ["assistant", "tool_result", "assistant"],
                     "display records in stream order")

        assert_equal(load_session(root, "0" * 64), [], "unknown session is empty")


def test_orchestrator_restore_rehydrates_state() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = _seed_session(root)
        turns = load_session(root, store.session_hash)

        ui = FakeUI()
        orch = Orchestrator(client=None, config=Config(), ui=ui)
        orch.turn_log.append(("stale goal", "stale outcome"))
        orch.chat_messages.append({"role": "user", "content": "stale"})

        orch.restore(turns)

        assert_equal(orch.turn_log,
                     [("hi there", "hello!"), ("fix the bug", "patched agent.py")],
                     "turn log rebuilt in order")
        assert_equal([m["role"] for m in orch.chat_messages],
                     ["system", "user", "assistant"],
                     "chat lineage is fresh system + verbatim delta")
        assert_equal(orch.chat_messages[1:], CHAT_DELTA, "chat delta re-extended")
        notes = orch._session_notes()
        assert_true("patched agent.py" in notes, "session notes see restored outcomes")


def test_replay_into_chat_renders_display_records() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = _seed_session(root)
        turns = load_session(root, store.session_hash)

        chat = RecordingChat()
        replay_into_chat(chat, turns)

        assert_equal(chat.calls[0], ("system", "restored session · 2 turn(s)"),
                     "restore header first")
        assert_equal(chat.calls[1], ("user", "hi there"), "turn goal as user message")
        assert_true(("assistant_chunk", "hello!") in chat.calls, "assistant replayed")
        tool_lines = [c for c in chat.calls
                      if c[0] == "system" and c[1].startswith("[run_shell_command]")]
        assert_equal(tool_lines,
                     [("system", "[run_shell_command] ok · tests passed")],
                     "tool result as one static system line")
        flushes = [c for c in chat.calls if c[0] == "flush_assistant"]
        assert_equal(len(flushes), 3, "every assistant message flushed")


def test_aborted_turn_still_restores_with_empty_outcome() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = SessionStore(root=root)
        store.begin_turn("crashed mid-stream")

        turns = load_session(root, store.session_hash)
        assert_equal(len(turns), 1, "aborted turn still recorded")
        assert_equal(turns[0].goal, "crashed mid-stream", "goal survives the abort")
        assert_equal(turns[0].outcome, "", "outcome empty, not missing")
        assert_true(not turns[0].ok, "aborted turn is not ok")

        summaries = list_session_summaries(root)
        assert_equal([s.session_hash for s in summaries], [store.session_hash],
                     "session with only an aborted turn is listed")


def test_restore_supports_resuming_turn() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = _seed_session(root)

        resumed = SessionStore(root=root)
        resumed.attach(store.session_hash)
        turn = resumed.begin_turn("follow-up")
        assert_equal(turn.path.name, "0004",
                     "new turns continue numbering past the incomplete dir")


def main() -> None:
    test_summaries_newest_first_and_filtered()
    test_summaries_fall_back_without_session_json()
    test_load_session_ordered_and_tolerant()
    test_orchestrator_restore_rehydrates_state()
    test_replay_into_chat_renders_display_records()
    test_aborted_turn_still_restores_with_empty_outcome()
    test_restore_supports_resuming_turn()
    print("sessions restore tests passed")


if __name__ == "__main__":
    main()
