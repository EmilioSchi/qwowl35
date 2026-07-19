"""Tests for the ``/sessions`` command: dispatch, the picker modal, restore
into the live app, and the busy guard.

Run directly: ``python qwowl35/tests/sessions_command_test.py``. The dispatch
test uses fakes (no TUI); the rest drive the real app headless via Textual's
Pilot against a temp-root session store.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.widgets import Static  # noqa: E402

from app import QwowlApp  # noqa: E402
from modes import Mode  # noqa: E402
from sessions.restore import RestoredTurn, load_session  # noqa: E402
from sessions.store import SessionStore  # noqa: E402
from widgets.session_selector import (  # noqa: E402
    SessionSelector,
    TurnSelector,
    render_restore_payload,
)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def _seed_session(
    root: Path, mode: str = "chat", age_seconds: float = 0.0, turns: int = 1
) -> SessionStore:
    store = SessionStore(root=root)
    for i in range(turns):
        goal = "hi there" if i == 0 else f"goal {i + 1}"
        outcome = "hello!" if i == 0 else f"done {i + 1}"
        turn = store.begin_turn(goal)
        turn.meta["mode"] = mode
        turn.meta["outcome"] = outcome
        if mode == "chat":
            turn.meta["chat_messages"] = [
                {"role": "user", "content": goal},
                {"role": "assistant", "content": outcome},
            ]
        turn.record("assistant", content=outcome, tool_calls=[])
        turn.meta["ok"] = True
        turn.flush_meta()
        store.end_turn(turn, ok=True)
    if age_seconds:
        meta_path = store.session_dir / "session.json"
        meta = json.loads(meta_path.read_text())
        meta["last_active_ts"] = time.time() - age_seconds
        meta_path.write_text(json.dumps(meta))
    return store


async def _pick_first_session(pilot, app) -> None:
    app._dispatch_command("/sessions")
    await _pause_until(
        pilot, lambda: isinstance(app.screen, SessionSelector), "session picker opened"
    )
    await pilot.press("enter")
    await _pause_until(
        pilot, lambda: isinstance(app.screen, TurnSelector), "turn picker opened"
    )


def _preview_text(app) -> str:
    try:
        body = app.screen.query_one("#preview-body", Static)
    except Exception:
        return ""
    renderable = body.render()
    return renderable.plain if hasattr(renderable, "plain") else str(renderable)


async def _pause_until(pilot, predicate, label: str, tries: int = 100) -> None:
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for: {label}")


def test_restore_payload_render() -> None:
    turns = [
        RestoredTurn(
            goal="hi there", mode="chat", outcome="hello!", ok=True, turn_dir="0001",
            chat_messages=[
                {"role": "user", "content": "hi there"},
                {"role": "assistant", "content": "hello!"},
            ],
        ),
        RestoredTurn(
            goal="fix the bug", mode="normal", outcome="patched", ok=True,
            turn_dir="0002", chat_messages=[],
        ),
    ]
    text = render_restore_payload(turns).plain
    assert_true("turn 1" in text and "turn 2" in text, "both turns rendered")
    assert_true("[chat]" in text and "[normal]" in text, "modes shown")
    assert_true("hi there" in text, "goal rendered")
    assert_true("hello!" in text, "outcome rendered")
    assert_true("chat_messages: 2" in text, "chat message count shown")
    assert_true("assistant ▸ hello!" in text, "chat message previewed")
    assert_true("(none — non-CHAT turn)" in text, "non-chat turn marked")
    assert_true("on restore" in text, "restore summary present")
    assert_true("no restorable turns" in render_restore_payload([]).plain,
                "empty payload handled")


def test_session_preview_shows_restore_payload() -> None:
    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _seed_session(root)  # one CHAT turn: goal "hi there", outcome "hello!"

            app = QwowlApp()
            app._session_store = SessionStore(root=root)
            async with app.run_test() as pilot:
                app._dispatch_command("/sessions")
                await _pause_until(
                    pilot,
                    lambda: isinstance(app.screen, SessionSelector),
                    "session picker opened",
                )
                await _pause_until(
                    pilot,
                    lambda: "hi there" in _preview_text(app),
                    "preview populated with the restore payload",
                )
                text = _preview_text(app)
                assert_true("hello!" in text, "outcome shown in preview")
                assert_true("chat_messages: 2" in text,
                            "verbatim chat messages counted in preview")

    asyncio.run(scenario())


def test_dispatch_routes_sessions_command() -> None:
    app = QwowlApp.__new__(QwowlApp)
    calls: list[str] = []
    app._open_session_selector = lambda: calls.append("sessions")
    assert_true(app._dispatch_command("/sessions"), "/sessions handled")
    assert_equal(calls, ["sessions"], "/sessions routed")
    for text in ("/session", "/SESSIONS", "/sessions now", "sessions"):
        assert_true(not app._dispatch_command(text), f"{text!r} not a command")
    assert_equal(calls, ["sessions"], "no extra handler fired")


def test_sessions_picker_restores_selected_session() -> None:
    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seeded = _seed_session(root)

            app = QwowlApp()
            app._session_store = SessionStore(root=root)
            async with app.run_test() as pilot:
                await _pick_first_session(pilot, app)
                await pilot.press("enter")
                await _pause_until(
                    pilot, lambda: app.agent.turn_log, "restore completed"
                )

                assert_equal(app.agent.turn_log, [("hi there", "hello!")],
                             "turn log restored")
                assert_equal(
                    [m["role"] for m in app.agent.chat_messages],
                    ["system", "user", "assistant"],
                    "chat lineage restored",
                )
                assert_true(len(app.chat.children) > 0, "transcript repopulated")
                assert_equal(app._session_store.session_hash, seeded.session_hash,
                             "store attached to the restored session")
                assert_equal(app.mode, Mode.CHAT,
                             "resumes in CHAT when the session ended there")

    asyncio.run(scenario())


def test_restore_resumes_the_session_last_mode() -> None:
    async def scenario() -> None:
        for seeded_mode, expected in [
            ("web", Mode.WEB),
            ("plan", Mode.PLAN),
            ("normal", Mode.NORMAL),
            ("", Mode.NORMAL),
        ]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _seed_session(root, mode=seeded_mode)

                app = QwowlApp()
                app._session_store = SessionStore(root=root)
                async with app.run_test() as pilot:
                    await _pick_first_session(pilot, app)
                    await pilot.press("enter")
                    await _pause_until(
                        pilot, lambda: app.agent.turn_log, "restore completed"
                    )
                    assert_equal(
                        app.mode, expected,
                        f"mode {seeded_mode or '(missing)'} resumes as {expected.value}",
                    )
                    assert_equal(
                        app.status.state.mode, expected,
                        "status bar box follows the resumed mode",
                    )

    asyncio.run(scenario())


def test_partial_restore_branches_to_fresh_session() -> None:
    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seeded = _seed_session(root, turns=3)

            app = QwowlApp()
            app._session_store = SessionStore(root=root)
            async with app.run_test() as pilot:
                await _pick_first_session(pilot, app)
                await pilot.press("up")
                await pilot.press("up")
                await pilot.press("enter")
                await _pause_until(
                    pilot, lambda: app.agent.turn_log, "restore completed"
                )

                assert_equal(app.agent.turn_log, [("hi there", "hello!")],
                             "only the first turn restored")
                assert_equal(app.agent.chat_messages[1:],
                             [{"role": "user", "content": "hi there"},
                              {"role": "assistant", "content": "hello!"}],
                             "chat lineage cut at the chosen turn")
                assert_true(
                    app._session_store.session_hash != seeded.session_hash,
                    "partial restore forks onto a fresh session",
                )
                fork_dir = app._session_store.session_dir
                fork_turns = sorted(p.name for p in (fork_dir / "turns").iterdir())
                assert_equal(fork_turns, ["0001"],
                             "prefix turn dirs copied into the fork")
                fork_meta = json.loads((fork_dir / "session.json").read_text())
                assert_equal(
                    fork_meta["restored_from"],
                    {"session": seeded.session_hash, "turns": 1},
                    "fork stamped with its origin",
                )
                reloaded = load_session(root, app._session_store.session_hash)
                assert_equal([t.goal for t in reloaded], ["hi there"],
                             "the fork is restorable on its own")
                original_turns = sorted(
                    p.name for p in (root / seeded.session_hash / "turns").iterdir()
                )
                assert_equal(original_turns, ["0001", "0002", "0003"],
                             "original session left intact")

    asyncio.run(scenario())


def test_turn_picker_escape_returns_to_session_list() -> None:
    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _seed_session(root, turns=2)

            app = QwowlApp()
            app._session_store = SessionStore(root=root)
            own_hash = app._session_store.session_hash
            async with app.run_test() as pilot:
                await _pick_first_session(pilot, app)

                await pilot.press("escape")
                await _pause_until(
                    pilot,
                    lambda: isinstance(app.screen, SessionSelector),
                    "escape steps back to the session list",
                )

                await pilot.press("escape")
                await _pause_until(
                    pilot,
                    lambda: not isinstance(
                        app.screen, (SessionSelector, TurnSelector)
                    ),
                    "second escape closes the flow",
                )
                assert_equal(app.agent.turn_log, [], "nothing restored")
                assert_equal(app._session_store.session_hash, own_hash,
                             "store untouched")

    asyncio.run(scenario())


def test_sessions_picker_escape_leaves_state_untouched() -> None:
    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _seed_session(root)

            app = QwowlApp()
            app._session_store = SessionStore(root=root)
            own_hash = app._session_store.session_hash
            async with app.run_test() as pilot:
                app._dispatch_command("/sessions")
                await _pause_until(
                    pilot,
                    lambda: isinstance(app.screen, SessionSelector),
                    "picker opened",
                )

                await pilot.press("escape")
                await _pause_until(
                    pilot,
                    lambda: not isinstance(app.screen, SessionSelector),
                    "picker dismissed",
                )

                assert_equal(app.agent.turn_log, [], "turn log untouched")
                assert_equal(len(app.chat.children), 0, "transcript untouched")
                assert_equal(app._session_store.session_hash, own_hash,
                             "store still on its own session")
                assert_equal(app.mode, Mode.NORMAL, "mode untouched")

    asyncio.run(scenario())


def test_sessions_locked_while_busy() -> None:
    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _seed_session(root)

            app = QwowlApp()
            app._session_store = SessionStore(root=root)
            async with app.run_test() as pilot:
                app._busy = True
                app._dispatch_command("/sessions")
                for _ in range(10):
                    await pilot.pause()
                assert_true(not isinstance(app.screen, SessionSelector),
                            "picker refused while a turn runs")

    asyncio.run(scenario())


def test_clear_makes_current_session_restorable() -> None:
    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = QwowlApp()
            app._session_store = SessionStore(root=root)
            app.agent.session_store = app._session_store
            async with app.run_test() as pilot:
                turn = app._session_store.begin_turn("hi there")
                turn.meta["mode"] = "chat"
                turn.meta["outcome"] = "hello!"
                turn.meta["ok"] = True
                turn.flush_meta()
                app._session_store.end_turn(turn, ok=True)
                pre_clear_hash = app._session_store.session_hash

                app._dispatch_command("/sessions")
                for _ in range(10):
                    await pilot.pause()
                assert_true(not isinstance(app.screen, SessionSelector),
                            "live session alone is not listed")

                app._dispatch_command("/clear")
                await pilot.pause()
                assert_true(app._session_store.session_hash != pre_clear_hash,
                            "/clear rotated to a fresh session")

                await _pick_first_session(pilot, app)
                await pilot.press("enter")
                await _pause_until(
                    pilot, lambda: app.agent.turn_log, "restore completed"
                )
                assert_equal(app.agent.turn_log, [("hi there", "hello!")],
                             "pre-clear conversation restored")
                assert_equal(app._session_store.session_hash, pre_clear_hash,
                             "store re-attached to the pre-clear session")

    asyncio.run(scenario())


def test_no_past_sessions_shows_notice_without_modal() -> None:
    async def scenario() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = QwowlApp()
            app._session_store = SessionStore(root=Path(tmp))
            async with app.run_test() as pilot:
                app._dispatch_command("/sessions")
                for _ in range(10):
                    await pilot.pause()
                assert_true(not isinstance(app.screen, SessionSelector),
                            "no picker for an empty store")

    asyncio.run(scenario())


def main() -> None:
    test_restore_payload_render()
    test_session_preview_shows_restore_payload()
    test_dispatch_routes_sessions_command()
    test_sessions_picker_restores_selected_session()
    test_restore_resumes_the_session_last_mode()
    test_partial_restore_branches_to_fresh_session()
    test_turn_picker_escape_returns_to_session_list()
    test_sessions_picker_escape_leaves_state_untouched()
    test_sessions_locked_while_busy()
    test_clear_makes_current_session_restorable()
    test_no_past_sessions_shows_notice_without_modal()
    print("sessions command tests passed")


if __name__ == "__main__":
    main()
