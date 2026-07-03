"""Tests for cross-launch theme persistence (``theme.preference``).

Run directly: ``python qwowl35/tests/theme_preference_test.py``. No TUI — a fake
catalog stands in for the real one, and the on-disk file is redirected to a temp
path so the tests never touch the user's real config dir.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theme import preference  # noqa: E402


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


class _FakeCatalog:
    """Minimal stand-in: 'qwowl' (dark only) and 'tokyo' (dark + light)."""

    names = ["qwowl", "tokyo"]

    def available_modes(self, name: str) -> list[str]:
        return {"qwowl": ["dark"], "tokyo": ["dark", "light"]}.get(name, [])


class _Env:
    """Context manager: isolate the pref file to a temp path and clear the env var."""

    def __enter__(self) -> "_Env":
        self._dir = tempfile.mkdtemp()
        self._orig_path = preference._pref_path
        preference._pref_path = lambda: Path(self._dir) / "theme.json"
        self._orig_env = os.environ.pop(preference.ENV_VAR, None)
        return self

    def __exit__(self, *exc) -> None:
        preference._pref_path = self._orig_path
        if self._orig_env is None:
            os.environ.pop(preference.ENV_VAR, None)
        else:
            os.environ[preference.ENV_VAR] = self._orig_env


CAT = _FakeCatalog()


def test_default_when_nothing_saved() -> None:
    with _Env():
        assert_equal(
            preference.load(CAT, default_name="qwowl"), ("qwowl", "dark"), "clean start default"
        )


def test_save_then_load_roundtrips() -> None:
    with _Env():
        preference.save("tokyo", "light")
        assert_equal(preference.load(CAT, default_name="qwowl"), ("tokyo", "light"), "roundtrip")


def test_env_overrides_saved_file() -> None:
    with _Env():
        preference.save("tokyo", "light")
        os.environ[preference.ENV_VAR] = "qwowl"
        assert_equal(preference.load(CAT, default_name="tokyo"), ("qwowl", "dark"), "env wins")


def test_env_name_mode_parsed() -> None:
    with _Env():
        os.environ[preference.ENV_VAR] = "tokyo:light"
        assert_equal(preference.load(CAT, default_name="qwowl"), ("tokyo", "light"), "name:mode")


def test_unknown_env_falls_back_to_file() -> None:
    with _Env():
        preference.save("tokyo", "dark")
        os.environ[preference.ENV_VAR] = "does-not-exist"
        assert_equal(
            preference.load(CAT, default_name="qwowl"), ("tokyo", "dark"), "bad env -> saved file"
        )


def test_unavailable_mode_snaps_to_available() -> None:
    with _Env():
        # 'qwowl' has no light mode; a saved/typed light value must snap to dark.
        os.environ[preference.ENV_VAR] = "qwowl:light"
        assert_equal(
            preference.load(CAT, default_name="tokyo"), ("qwowl", "dark"), "mode snapped"
        )


def test_corrupt_file_falls_back_to_default() -> None:
    with _Env():
        preference._pref_path().write_text("{not json", encoding="utf-8")
        assert_equal(
            preference.load(CAT, default_name="qwowl"), ("qwowl", "dark"), "corrupt -> default"
        )


def test_save_never_raises_on_bad_dir() -> None:
    with _Env():
        # Point at an unwritable path; save must swallow the error.
        preference._pref_path = lambda: Path("/nonexistent-root/xyz/theme.json")
        preference.save("tokyo", "dark")  # must not raise


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("theme preference tests passed")


if __name__ == "__main__":
    main()
