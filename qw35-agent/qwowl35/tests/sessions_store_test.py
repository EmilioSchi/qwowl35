"""Tests for the session store: hashing, turn allocation, artifacts, and
the age/count/orphan cleanup.

Run directly: ``python qwowl35/tests/sessions_store_test.py``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sessions.store import SessionStore, generate_session_hash  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def _age_session(root: Path, session_hash: str, seconds: float) -> None:
    meta_path = root / session_hash / "session.json"
    meta = json.loads(meta_path.read_text())
    meta["last_active_ts"] = time.time() - seconds
    meta_path.write_text(json.dumps(meta))


def test_session_hash_is_unique_sha256_hex() -> None:
    first = generate_session_hash()
    second = generate_session_hash()
    assert_equal(len(first), 64, "sha256 hex length")
    assert_true(all(c in "0123456789abcdef" for c in first), "lowercase hex")
    assert_true(first != second, "hashes distinct across calls")


def test_turn_dirs_increment_in_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(root=Path(tmp))
        first = store.begin_turn("first goal")
        store.end_turn(first, ok=True)
        second = store.begin_turn("second goal")
        store.end_turn(second, ok=True)

        assert_equal(first.path.name, "0001", "first turn dir")
        assert_equal(second.path.name, "0002", "second turn dir")
        assert_true(first.path.parent == store.session_dir / "turns", "turns/ layout")


def test_turn_dir_saves_artifacts_and_meta() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(root=Path(tmp))
        turn = store.begin_turn("fix the parser")
        turn.save("plan.md", "Approved plan: 1. fix it")
        turn.record_timings({"session_path": "checkpoint", "prompt_eval_count": 40,
                             "cached_prompt_tokens": 900})
        turn.record_timings({"session_path": "extend", "prompt_eval_count": 10,
                             "cached_prompt_tokens": 950})
        turn.meta["ok"] = True
        turn.flush_meta()
        store.end_turn(turn, ok=True)

        assert_true((turn.path / "plan.md").exists(), "artifact written")
        meta = json.loads((turn.path / "meta.json").read_text())
        assert_equal(meta["goal"], "fix the parser", "goal recorded")
        assert_equal(meta["session_paths"], {"checkpoint": 1, "extend": 1}, "paths tallied")
        assert_equal(meta["total_prompt_eval_count"], 50, "prefill totals summed")
        assert_equal(meta["total_cached_prompt_tokens"], 1850, "cached totals summed")


def test_transcript_holds_only_conversation_records() -> None:
    # The transcript is a restore-display record: begin_turn/end_turn no longer
    # emit meta/state/turn_end bookkeeping (that lives in meta.json), and the
    # heavy request/raw_chunk records are gone entirely. Only the conversation
    # records written through turn.record land in the file.
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(root=Path(tmp))
        turn = store.begin_turn("trace me")
        turn.record("assistant", content="on it", tool_calls=[])
        turn.record("tool_result", id="c1", name="bash", result="ok", is_error=False)
        turn.meta["mode"] = "normal"
        turn.meta["outcome"] = "done"
        turn.flush_meta()
        store.end_turn(turn, ok=True)

        lines = (turn.path / "transcript.jsonl").read_text().splitlines()
        kinds = [json.loads(line)["kind"] for line in lines]
        assert_equal(kinds, ["assistant", "tool_result"],
                     "only conversation records in the transcript")

        meta = json.loads((turn.path / "meta.json").read_text())
        assert_equal(meta["goal"], "trace me", "goal in meta.json")
        assert_equal(meta["mode"], "normal", "mode in meta.json")
        assert_equal(meta["outcome"], "done", "outcome in meta.json")


def test_session_json_upserted() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(root=Path(tmp))
        turn = store.begin_turn("first goal")
        turn.meta["mode"] = "chat"
        store.end_turn(turn, ok=True)

        meta = json.loads((store.session_dir / "session.json").read_text())
        assert_equal(meta["hash"], store.session_hash, "hash recorded")
        assert_equal(meta["first_goal"], "first goal", "first goal kept")
        assert_equal(meta["last_mode"], "chat", "last mode recorded")
        assert_equal(meta["turn_count"], 1, "turn count recorded")
        assert_true(isinstance(meta["last_active_ts"], float), "sortable timestamp")

        second = store.begin_turn("second goal")
        store.end_turn(second, ok=False)
        meta = json.loads((store.session_dir / "session.json").read_text())
        assert_equal(meta["first_goal"], "first goal", "first goal not overwritten")
        assert_equal(meta["turn_count"], 2, "turn count advanced")


def test_begin_turn_flushes_meta_immediately() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(root=Path(tmp))
        turn = store.begin_turn("aborted goal")

        meta = json.loads((turn.path / "meta.json").read_text())
        assert_equal(meta["goal"], "aborted goal",
                     "goal on disk before the turn completes")
        assert_true("started" in meta, "start stamp flushed early")


def test_rotate_starts_fresh_session() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(root=Path(tmp))
        turn = store.begin_turn("before clear")
        store.end_turn(turn, ok=True)
        old_hash = store.session_hash
        old_dir = store.session_dir

        new_hash = store.rotate()
        assert_true(new_hash != old_hash, "rotation mints a new hash")
        assert_true(old_dir.exists(), "previous session left restorable")

        turn = store.begin_turn("after clear")
        assert_equal(turn.path.name, "0001", "numbering restarts")
        assert_true(turn.path.is_relative_to(store.root / new_hash),
                    "new turns land in the new session dir")


def test_fork_copies_prefix_and_continues_numbering() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = SessionStore(root=root)
        for i, mode in enumerate(("chat", "web", "normal"), start=1):
            turn = source.begin_turn(f"goal {i}")
            turn.meta["mode"] = mode
            turn.flush_meta()
            source.end_turn(turn, ok=True)

        store = SessionStore(root=root)
        fork_hash = store.fork(source.session_hash, ["0001", "0002"])

        assert_equal(store.session_hash, fork_hash, "store attached to the fork")
        fork_turns = sorted(
            p.name for p in (store.session_dir / "turns").iterdir()
        )
        assert_equal(fork_turns, ["0001", "0002"], "prefix copied")
        assert_true(
            (store.session_dir / "turns" / "0001" / "transcript.jsonl").exists(),
            "transcripts copied with the turns",
        )
        meta = json.loads((store.session_dir / "session.json").read_text())
        assert_equal(meta["restored_from"],
                     {"session": source.session_hash, "turns": 2},
                     "origin stamped")
        assert_equal(meta["first_goal"], "goal 1", "first goal from the prefix")
        assert_equal(meta["last_mode"], "web", "last mode from the prefix")
        assert_equal(meta["turn_count"], 2, "turn count matches the copies")

        turn = store.begin_turn("after fork")
        assert_equal(turn.path.name, "0003", "numbering continues past the copies")
        source_turns = sorted(
            p.name for p in (source.session_dir / "turns").iterdir()
        )
        assert_equal(source_turns, ["0001", "0002", "0003"], "source untouched")


def test_attach_resumes_numbering() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        original = SessionStore(root=Path(tmp))
        for goal in ("one", "two"):
            turn = original.begin_turn(goal)
            original.end_turn(turn, ok=True)

        resumed = SessionStore(root=Path(tmp))
        resumed.attach(original.session_hash)
        turn = resumed.begin_turn("three")
        assert_equal(turn.path.name, "0003", "numbering continues after attach")
        assert_equal(resumed.session_dir, original.session_dir, "same session dir")


def test_cleanup_removes_old_excess_and_orphans() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        old = SessionStore(root=root)
        turn = old.begin_turn("old session")
        old.end_turn(turn, ok=True)
        _age_session(root, old.session_hash, 40 * 86400)

        fresh = SessionStore(root=root)
        turn = fresh.begin_turn("fresh session")
        fresh.end_turn(turn, ok=True)

        stale_orphan = root / generate_session_hash()
        (stale_orphan / "turns" / "0001").mkdir(parents=True)
        past = time.time() - 2 * 3600
        os.utime(stale_orphan, (past, past))

        young_orphan = root / generate_session_hash()
        (young_orphan / "turns" / "0001").mkdir(parents=True)

        stranger = root / "not-a-session"
        stranger.mkdir()

        current = SessionStore(root=root)
        turn = current.begin_turn("current session")
        current.end_turn(turn, ok=True)
        _age_session(root, current.session_hash, 40 * 86400)

        removed = current.cleanup(max_age_days=30, max_sessions=50)
        assert_equal(removed, 2, "old session and stale orphan removed")
        assert_true(not (root / old.session_hash).exists(), "old session gone")
        assert_true(not stale_orphan.exists(), "stale orphan gone")
        assert_true(young_orphan.exists(), "orphan inside grace window kept")
        assert_true((root / fresh.session_hash).exists(), "fresh session kept")
        assert_true((root / current.session_hash).exists(), "attached session spared")
        assert_true(stranger.exists(), "unrecognized dirs untouched")

        extra_hashes = []
        for i in range(3):
            extra = SessionStore(root=root)
            turn = extra.begin_turn(f"extra {i}")
            extra.end_turn(turn, ok=True)
            extra_hashes.append(extra.session_hash)
        removed = current.cleanup(max_age_days=30, max_sessions=2)
        assert_equal(removed, 2, "count cap enforced")


def test_cleanup_sweeps_legacy_runs_tree() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "sessions"
        legacy = Path(tmp) / "runs"
        (legacy / "20260101-000000-abcdef").mkdir(parents=True)

        store = SessionStore(root=root)
        removed = store.cleanup()
        assert_true(removed >= 1, "legacy tree counted")
        assert_true(not legacy.exists(), "legacy runs tree swept")


def main() -> None:
    test_session_hash_is_unique_sha256_hex()
    test_turn_dirs_increment_in_order()
    test_turn_dir_saves_artifacts_and_meta()
    test_transcript_holds_only_conversation_records()
    test_session_json_upserted()
    test_begin_turn_flushes_meta_immediately()
    test_rotate_starts_fresh_session()
    test_fork_copies_prefix_and_continues_numbering()
    test_attach_resumes_numbering()
    test_cleanup_removes_old_excess_and_orphans()
    test_cleanup_sweeps_legacy_runs_tree()
    print("sessions store tests passed")


if __name__ == "__main__":
    main()
