"""Tests for the web-UI font catalog, active.css generation, and the persisted
``/fonts`` preference (``webfonts`` package).

Run directly: ``python qwowl35/tests/webfonts_test.py``. No TUI — the on-disk
preference file is redirected to a temp path so the tests never touch the
user's real config dir. The catalog test reads the real ``webui/fonts/`` tree,
pinning the per-family directory layout the server copies at startup.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import webfonts  # noqa: E402
from webfonts import preference  # noqa: E402

FONTS_ROOT = Path(__file__).resolve().parent.parent / "webui" / "fonts"


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


class _Env:
    """Context manager: isolate the pref file to a temp path and clear the env var."""

    def __enter__(self) -> "_Env":
        self._dir = tempfile.mkdtemp()
        self._orig_path = preference._pref_path
        preference._pref_path = lambda: Path(self._dir) / "font.json"
        self._orig_env = os.environ.pop(preference.ENV_VAR, None)
        return self

    def __exit__(self, *exc) -> None:
        preference._pref_path = self._orig_path
        if self._orig_env is None:
            os.environ.pop(preference.ENV_VAR, None)
        else:
            os.environ[preference.ENV_VAR] = self._orig_env


# --------------------------------------------------------------------------- #
# Catalog ↔ vendored files (pins the webui/fonts/<slug>/ layout)
# --------------------------------------------------------------------------- #
def test_catalog_slugs_unique_and_default_present() -> None:
    slugs = webfonts.slugs()
    assert_equal(len(slugs), len(set(slugs)), "slugs are unique")
    assert_true(webfonts.DEFAULT_SLUG in slugs, "default slug is in the catalog")
    assert_true(webfonts.get("no-such-family") is None, "unknown slug -> None")


def test_every_catalog_file_is_vendored() -> None:
    for family in webfonts.CATALOG:
        family_dir = FONTS_ROOT / family.slug
        assert_true(family_dir.is_dir(), f"{family.slug}: webui/fonts dir exists")
        for file in family.files:
            assert_true(
                (family_dir / file.filename).is_file(),
                f"{family.slug}/{file.filename} exists on disk",
            )
        # Every family ships its license/provenance next to the woff2 files.
        extras = [p.name for p in family_dir.iterdir() if not p.name.endswith(".woff2")]
        assert_true(extras, f"{family.slug}: a license/info file is vendored")


# --------------------------------------------------------------------------- #
# active.css generation
# --------------------------------------------------------------------------- #
def test_active_css_declares_family_and_roboto_alias() -> None:
    family = webfonts.get(webfonts.DEFAULT_SLUG)
    css = webfonts.active_css(family)
    # One block per file per name: the real family + the "Roboto Mono" alias
    # that js/textual.js hardcodes.
    assert_equal(css.count("@font-face"), 2 * len(family.files), "two blocks per file")
    assert_equal(css.count(f'font-family: "{family.css_family}"'), len(family.files), "real name")
    assert_equal(css.count('font-family: "Roboto Mono"'), len(family.files), "alias")


def test_active_css_urls_are_relative_to_the_stylesheet() -> None:
    # active.css is written at runtime, not rendered through Jinja: URLs must be
    # slug-relative (resolved against /static/fonts/active.css), never templated.
    for family in webfonts.CATALOG:
        css = webfonts.active_css(family)
        assert_true("{{" not in css, f"{family.slug}: no template syntax leaks")
        for file in family.files:
            assert_true(
                f'url("{family.slug}/{file.filename}")' in css,
                f"{family.slug}/{file.filename} referenced slug-relative",
            )


def test_active_css_regular_only_family_emits_two_blocks() -> None:
    family = webfonts.get("proggyclean")
    assert_equal(len(family.files), 1, "proggyclean ships only a Regular cut")
    assert_equal(webfonts.active_css(family).count("@font-face"), 2, "one face per name")


def test_write_active_css_writes_and_rewrites() -> None:
    import json

    with tempfile.TemporaryDirectory() as tmp:
        first = webfonts.get("mononoki")
        second = webfonts.get("terminus")
        webfonts.write_active_css(tmp, first)
        assert_equal((Path(tmp) / "active.css").read_text(), webfonts.active_css(first), "written")
        # The manifest the page polls for live apply names the slug + family.
        assert_equal(
            json.loads((Path(tmp) / "active.json").read_text()),
            {"slug": "mononoki", "family": first.css_family},
            "manifest written",
        )
        webfonts.write_active_css(tmp, second)
        assert_equal((Path(tmp) / "active.css").read_text(), webfonts.active_css(second), "rewritten")
        assert_equal(
            json.loads((Path(tmp) / "active.json").read_text())["slug"], "terminus",
            "manifest follows",
        )
        leftovers = [p for p in Path(tmp).iterdir() if p.name not in ("active.css", "active.json")]
        assert_equal(leftovers, [], "no temp files left behind")


def test_write_active_css_never_raises_on_bad_dir() -> None:
    webfonts.write_active_css("/nonexistent-root/xyz", webfonts.CATALOG[0])  # must not raise


# --------------------------------------------------------------------------- #
# Preference persistence
# --------------------------------------------------------------------------- #
def test_default_when_nothing_saved() -> None:
    with _Env():
        assert_equal(preference.load(), webfonts.DEFAULT_SLUG, "clean start default")


def test_save_then_load_roundtrips() -> None:
    with _Env():
        preference.save("terminus")
        assert_equal(preference.load(), "terminus", "roundtrip")


def test_env_overrides_saved_file() -> None:
    with _Env():
        preference.save("terminus")
        os.environ[preference.ENV_VAR] = "meslo"
        assert_equal(preference.load(), "meslo", "env wins")


def test_unknown_env_falls_back_to_file() -> None:
    with _Env():
        preference.save("gohu")
        os.environ[preference.ENV_VAR] = "does-not-exist"
        assert_equal(preference.load(), "gohu", "bad env -> saved file")


def test_unknown_saved_slug_falls_back_to_default() -> None:
    with _Env():
        preference._pref_path().write_text('{"slug": "removed-family"}', encoding="utf-8")
        assert_equal(preference.load(), webfonts.DEFAULT_SLUG, "unknown slug -> default")


def test_corrupt_file_falls_back_to_default() -> None:
    with _Env():
        preference._pref_path().write_text("{not json", encoding="utf-8")
        assert_equal(preference.load(), webfonts.DEFAULT_SLUG, "corrupt -> default")


def test_save_never_raises_on_bad_dir() -> None:
    with _Env():
        preference._pref_path = lambda: Path("/nonexistent-root/xyz/font.json")
        preference.save("terminus")  # must not raise


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("webfonts tests passed")


if __name__ == "__main__":
    main()
