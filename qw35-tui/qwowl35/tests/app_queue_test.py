"""Tests for qwowl35 busy-time message queue behavior."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import QwowlApp, format_queued_user_batch  # noqa: E402


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


class FakePanel:
    def __init__(self) -> None:
        self.display = False
        self.content = ""

    def update(self, content) -> None:
        self.content = content


class FakePrompt:
    def __init__(self) -> None:
        self.history: list[str] = []
        self.cleared = False

    def append_history(self, text: str) -> None:
        self.history.append(text)

    def clear(self) -> None:
        self.cleared = True


def _plain(content) -> str:
    return content.plain if hasattr(content, "plain") else str(content)


def make_app(*, busy: bool):
    app = QwowlApp.__new__(QwowlApp)
    app._busy = busy
    app._queued_messages = []
    app.queue_panel = FakePanel()
    app.started_turns = []
    app._run_turn = app.started_turns.append
    return app


def test_format_queued_user_batch_merges_in_order() -> None:
    batch = format_queued_user_batch(["first", "second\nline"])

    assert_equal(
        batch,
        "first\n\nsecond\nline",
        "merged queue format",
    )


def test_busy_submission_queues_and_updates_display() -> None:
    app = make_app(busy=True)
    prompt = FakePrompt()

    app._handle_submission(prompt, "  queued message  ")

    assert_equal(prompt.history, ["  queued message  "], "busy submission stored in history")
    assert_true(prompt.cleared, "busy submission clears prompt")
    assert_equal(app.started_turns, [], "busy submission does not start a worker")
    assert_equal(app._queued_messages, ["queued message"], "busy submission queued")
    assert_true(app.queue_panel.display, "queue panel shown")
    assert_true("Queue" in _plain(app.queue_panel.content), "queue label shown")
    assert_true("queued message" in _plain(app.queue_panel.content), "preview shown")
    assert_true("1. queued message" not in _plain(app.queue_panel.content), "preview is not numbered")


def test_idle_submission_starts_turn_without_queueing() -> None:
    app = make_app(busy=False)
    prompt = FakePrompt()

    app._handle_submission(prompt, "  hello  ")

    assert_equal(app.started_turns, ["hello"], "idle submission starts turn")
    assert_equal(app._queued_messages, [], "idle submission not queued")
    assert_equal(prompt.history, ["  hello  "], "idle submission stored in history")
    assert_true(prompt.cleared, "idle submission clears prompt")


def test_pop_queued_user_batch_merges_and_hides_display() -> None:
    app = make_app(busy=True)
    app._enqueue_message("first")
    app._enqueue_message("second")

    batch = app.pop_queued_user_batch()

    assert_equal(batch, "first\n\nsecond", "queued batch")
    assert_equal(app._queued_messages, [], "queue cleared")
    assert_true(not app.queue_panel.display, "queue panel hidden")
    assert_equal(app.queue_panel.content, "", "queue display cleared")


def main() -> None:
    test_format_queued_user_batch_merges_in_order()
    test_busy_submission_queues_and_updates_display()
    test_idle_submission_starts_turn_without_queueing()
    test_pop_queued_user_batch_merges_and_hides_display()
    print("app queue tests passed")


if __name__ == "__main__":
    main()
