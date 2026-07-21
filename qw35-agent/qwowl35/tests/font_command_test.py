"""Tests for the ``/fonts`` command: the picker flow (Pilot) and the commit
side effects (preference saved; active.css rewritten only under webgui).

Run directly: ``python qwowl35/tests/font_command_test.py``. The preference
file is redirected to a temp path so the tests never touch the user's real
config dir, and ``QWOWL35_WEBUI_FONTS_DIR`` is controlled per scenario.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import webfonts  # noqa: E402
from app import QwowlApp  # noqa: E402
from webfonts import preference  # noqa: E402
from widgets.font_selector import FontSelector  # noqa: E402


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


class _Env:
    """Isolate the pref file to a temp path; clear both webfonts env vars."""

    def __enter__(self) -> "_Env":
        self._dir = tempfile.mkdtemp()
        self._orig_path = preference._pref_path
        preference._pref_path = lambda: Path(self._dir) / "font.json"
        self._saved = {
            var: os.environ.pop(var, None)
            for var in (preference.ENV_VAR, webfonts.FONTS_DIR_ENV)
        }
        return self

    def __exit__(self, *exc) -> None:
        preference._pref_path = self._orig_path
        for var, value in self._saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


def test_fonts_picker_commit_saves_second_family() -> None:
    async def scenario() -> None:
        with _Env():
            app = QwowlApp()
            async with app.run_test() as pilot:
                app._dispatch_command("/fonts")
                await pilot.pause()
                await pilot.pause()
                assert_true(isinstance(app.screen, FontSelector), "picker opened")

                await pilot.press("down", "enter")
                await pilot.pause()
                await pilot.pause()
                assert_true(not isinstance(app.screen, FontSelector), "picker dismissed")
                assert_equal(
                    preference.load(), webfonts.CATALOG[1].slug, "second family persisted"
                )

    asyncio.run(scenario())


def test_fonts_picker_escape_saves_nothing() -> None:
    async def scenario() -> None:
        with _Env():
            app = QwowlApp()
            async with app.run_test() as pilot:
                app._dispatch_command("/fonts")
                await pilot.pause()
                await pilot.pause()
                await pilot.press("down", "escape")
                await pilot.pause()
                await pilot.pause()
                assert_true(not isinstance(app.screen, FontSelector), "picker dismissed")
                assert_true(
                    not preference._pref_path().exists(), "escape wrote no preference"
                )

    asyncio.run(scenario())


def test_highlight_previews_and_escape_reverts_under_webgui() -> None:
    # With FONTS_DIR_ENV set (webgui: the server exported the scratch statics
    # dir), moving the highlight rewrites the served active.css live, and
    # escape restores the committed family.
    async def scenario() -> None:
        with _Env(), tempfile.TemporaryDirectory() as fonts_dir:
            os.environ[webfonts.FONTS_DIR_ENV] = fonts_dir
            app = QwowlApp()
            async with app.run_test() as pilot:
                app._dispatch_command("/fonts")
                await pilot.pause()
                await pilot.pause()
                await pilot.press("down")
                await pilot.pause()
                assert_equal(
                    (Path(fonts_dir) / "active.css").read_text(),
                    webfonts.active_css(webfonts.CATALOG[1]),
                    "highlight previewed live",
                )
                await pilot.press("escape")
                await pilot.pause()
                await pilot.pause()
            assert_equal(
                (Path(fonts_dir) / "active.css").read_text(),
                webfonts.active_css(webfonts.CATALOG[0]),
                "escape reverted the preview",
            )
            assert_true(not preference._pref_path().exists(), "nothing persisted")

    asyncio.run(scenario())


def test_commit_rewrites_active_css_only_under_webgui() -> None:
    # A commit rewrites the served active.css when FONTS_DIR_ENV is set;
    # without it (plain TUI), the commit only persists the preference.
    async def scenario() -> None:
        with _Env(), tempfile.TemporaryDirectory() as fonts_dir:
            os.environ[webfonts.FONTS_DIR_ENV] = fonts_dir
            app = QwowlApp()
            async with app.run_test() as pilot:
                app._dispatch_command("/fonts")
                await pilot.pause()
                await pilot.pause()
                await pilot.press("down", "enter")
                await pilot.pause()
                await pilot.pause()
            expected = webfonts.active_css(webfonts.CATALOG[1])
            assert_equal(
                (Path(fonts_dir) / "active.css").read_text(), expected, "active.css rewritten"
            )

        with _Env():
            app = QwowlApp()
            async with app.run_test() as pilot:
                app._dispatch_command("/fonts")
                await pilot.pause()
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                await pilot.pause()
            assert_equal(preference.load(), webfonts.CATALOG[0].slug, "preference saved")

    asyncio.run(scenario())


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("font command tests passed")


if __name__ == "__main__":
    main()
