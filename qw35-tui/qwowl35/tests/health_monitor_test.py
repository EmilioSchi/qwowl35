"""Regression guard for the health-monitor worker group.

The health poller (`_monitor_health`) and the turn runner (`_run_turn`) are both
`@work(exclusive=True)`. If they share a worker group, submitting a message
while the server is still connecting cancels the health poller — so when the
server comes up later, `update_health` is never called again and the footer's
context size/percentage stays stuck at "?". They must live in distinct groups.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_mod  # noqa: E402
from app import QwowlApp  # noqa: E402


def _raw_monitor():
    """The undecorated `_monitor_health` coroutine, pulled from the @work closure."""
    cells = dict(zip(
        QwowlApp._monitor_health.__code__.co_freevars,
        QwowlApp._monitor_health.__closure__,
    ))
    return cells["method"].cell_contents


class FakeMonitorApp:
    """Minimal stand-in exposing only what `_monitor_health` touches."""

    def __init__(self, probes: list[bool]) -> None:
        self._probes = list(probes)
        self._busy = False
        self.actions: list[str] = []

    async def _check_health(self) -> bool:
        return self._probes.pop(0)

    def set_state(self, state) -> None:
        self.actions.append(f"state:{getattr(state, 'value', state)}")

    def set_info(self, message: str) -> None:
        self.actions.append(f"info:{message}")

    def set_error(self, code: str, message: str) -> None:
        self.actions.append(f"error:{code}")


def _drive(probes: list[bool], busy_at: dict[int, bool] | None = None) -> list[str]:
    """Run the monitor over a scripted probe sequence, then stop it.

    `asyncio.sleep` is patched to advance a tick counter (optionally flipping
    `_busy` before the next probe) and to raise once the script is exhausted so
    the otherwise-infinite loop terminates.
    """
    fake = FakeMonitorApp(probes)
    busy_at = busy_at or {}
    tick = {"n": 0}
    real_sleep = asyncio.sleep

    async def fake_sleep(_seconds):
        tick["n"] += 1
        if not fake._probes:
            raise asyncio.CancelledError
        if tick["n"] in busy_at:
            fake._busy = busy_at[tick["n"]]
        await real_sleep(0)

    app_mod.asyncio.sleep = fake_sleep
    try:
        try:
            asyncio.run(_raw_monitor()(fake))
        except asyncio.CancelledError:
            pass
    finally:
        app_mod.asyncio.sleep = real_sleep
    return fake.actions


def test_first_connect_settles_into_waiting() -> None:
    actions = _drive([True, True])
    assert actions == ["state:waiting"], f"only the initial connect reacts, got {actions}"


def test_disconnect_after_connect_is_tracked() -> None:
    # Was the bug: the monitor returned on first success and never noticed the
    # server going away. It must now flip to an offline error.
    actions = _drive([True, False, False])
    assert actions == ["state:waiting", "error:offline"], (
        f"a mid-session disconnect must surface offline, got {actions}"
    )


def test_reconnect_flashes_connected() -> None:
    actions = _drive([True, False, True, True])
    assert actions == ["state:waiting", "error:offline", "info:connected"], (
        f"recovery should flash 'connected', got {actions}"
    )


def test_edge_while_busy_is_deferred_until_idle() -> None:
    # Server drops on probe 2, but a turn is generating (busy) — the mascot is
    # left to the generation/stream handler. Once idle (tick 2), the still-down
    # server is reflected as offline.
    actions = _drive([True, False, False], busy_at={1: True, 2: False})
    assert actions == ["state:waiting", "error:offline"], (
        f"a drop seen while busy must apply once idle, got {actions}"
    )


def _work_config(method) -> dict:
    """Pull the @work decorator's bound config out of the wrapper closure."""
    freevars = method.__code__.co_freevars
    cells = method.__closure__ or ()
    return {name: cell.cell_contents for name, cell in zip(freevars, cells)}


def test_health_and_turn_workers_are_in_distinct_groups() -> None:
    health = _work_config(QwowlApp._monitor_health)
    turn = _work_config(QwowlApp._run_turn)

    assert health.get("exclusive") is True, "health monitor should be exclusive"
    assert turn.get("exclusive") is True, "turn runner should be exclusive"
    assert health.get("group") != turn.get("group"), (
        "health monitor and turn runner must use different worker groups, "
        "otherwise running a turn cancels the health poller and the footer "
        f"never recovers (both are {health.get('group')!r})"
    )
