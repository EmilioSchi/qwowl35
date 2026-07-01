"""Tests for the opencode theme resolver and registry.

Run directly: ``python qwowl35/tests/theme_loader_test.py``. Pure logic — no TUI,
no network. Validates the color math, the palette derivation over real bundled
theme files, and the catalog wiring.
"""

from __future__ import annotations

import re
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import theme  # noqa: E402
from theme import loader as tl  # noqa: E402
from theme import registry as theme_registry  # noqa: E402

_HEX = re.compile(r"^#[0-9a-f]{6}$")


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def _lum(hex_color: str) -> float:
    r, g, b = tl.hex_to_rgb(hex_color)
    return 0.299 * r + 0.587 * g + 0.114 * b


# --------------------------------------------------------------------------- #
# Color math
# --------------------------------------------------------------------------- #
def test_hex_to_rgb_forms() -> None:
    assert_equal(tl.hex_to_rgb("#ffffff"), (255, 255, 255), "6-digit")
    assert_equal(tl.hex_to_rgb("#fff"), (255, 255, 255), "3-digit shorthand")
    assert_equal(tl.hex_to_rgb("#112233ff"), (0x11, 0x22, 0x33), "8-digit drops alpha")
    assert_equal(tl.hex_to_rgb("C7C7C7"), (0xC7, 0xC7, 0xC7), "missing hash tolerated")
    assert_equal(tl.hex_to_rgb("var(--x)"), None, "non-hex -> None")
    assert_equal(tl.hex_to_rgb("#12"), None, "wrong length -> None")
    assert_equal(tl.hex_to_rgb(None), None, "None -> None")


def test_normalize_hex() -> None:
    assert_equal(tl.normalize_hex("#100F0F"), "#100f0f", "uppercase lowered")
    assert_equal(tl.normalize_hex("#e4e4e45e"), "#e4e4e4", "alpha stripped")
    assert_equal(tl.normalize_hex("#abc"), "#aabbcc", "shorthand expanded")
    assert_equal(tl.normalize_hex("nope"), None, "invalid -> None")


def test_mix() -> None:
    assert_equal(tl.mix("#000000", "#ffffff", 0.5), "#808080", "midpoint")
    assert_equal(tl.mix("#000000", "#ffffff", 0.0), "#000000", "alpha 0 -> base")
    assert_equal(tl.mix("#000000", "#ffffff", 1.0), "#ffffff", "alpha 1 -> overlay")
    assert_equal(tl.mix("#000000", "nope", 0.5), "#000000", "non-hex overlay -> base fallback")


# --------------------------------------------------------------------------- #
# resolve_theme
# --------------------------------------------------------------------------- #
_MINIMAL = {
    "dark": {
        "palette": {
            "neutral": "#101010",
            "ink": "#e0e0e0",
            "primary": "#3366cc",
            "success": "#33cc66",
            "warning": "#ccaa33",
            "error": "#cc3333",
            "info": "#33aacc",
        },
        "overrides": {},
    },
    "light": {
        "palette": {
            "neutral": "#f0f0f0",
            "ink": "#202020",
            "primary": "#2255bb",
            "success": "#229955",
            "warning": "#997722",
            "error": "#aa2222",
            "info": "#2288aa",
        },
        "overrides": {"text-weak": "#777777"},
    },
}


def test_resolve_all_tokens_valid_hex() -> None:
    for mode in ("dark", "light"):
        pal = tl.resolve_theme(_MINIMAL, mode)
        for f in fields(theme.Palette):
            val = getattr(pal, f.name)
            assert_true(_HEX.match(val), f"{mode}/{f.name} is #rrggbb, got {val!r}")


def test_resolve_dark_vs_light_orientation() -> None:
    d = tl.resolve_theme(_MINIMAL, "dark")
    lt = tl.resolve_theme(_MINIMAL, "light")
    assert_true(_lum(d.BG_BASE) < _lum(d.FG_BRIGHT), "dark: bg darker than fg")
    assert_true(_lum(lt.BG_BASE) > _lum(lt.FG_BRIGHT), "light: bg lighter than fg")
    assert_equal(d.BG_BASE, "#101010", "dark BG_BASE from neutral")
    assert_equal(d.FG_BRIGHT, "#e0e0e0", "dark FG_BRIGHT from ink")


def test_resolve_accent_falls_back_to_primary() -> None:
    # No 'accent' key -> ACCENT should equal primary.
    pal = tl.resolve_theme(_MINIMAL, "dark")
    assert_equal(pal.ACCENT, "#3366cc", "ACCENT falls back to primary")
    with_accent = {"dark": {"palette": {**_MINIMAL["dark"]["palette"], "accent": "#ff9900"}}}
    assert_equal(tl.resolve_theme(with_accent, "dark").ACCENT, "#ff9900", "accent used when present")


def test_resolve_honors_text_weak_override() -> None:
    lt = tl.resolve_theme(_MINIMAL, "light")
    assert_equal(lt.FG_DIM, "#777777", "text-weak override used for FG_DIM")


def test_resolve_ignores_nonhex_and_missing() -> None:
    junk = {
        "dark": {
            "palette": {"neutral": "#101010", "ink": "#e0e0e0"},  # missing primary etc.
            "overrides": {"text-weak": "var(--nope)"},           # non-hex ignored
        }
    }
    pal = tl.resolve_theme(junk, "dark")
    for f in fields(theme.Palette):
        assert_true(_HEX.match(getattr(pal, f.name)), f"{f.name} still valid despite gaps")
    # missing primary -> ACCENT falls back to the built-in default accent
    assert_equal(pal.ACCENT, theme.DEFAULT.ACCENT, "missing primary -> default accent")
    # non-hex text-weak ignored -> FG_DIM derived (not the junk string)
    assert_true(_HEX.match(pal.FG_DIM), "non-hex text-weak ignored")


def test_missing_mode_falls_back() -> None:
    only_dark = {"dark": _MINIMAL["dark"]}
    pal = tl.resolve_theme(only_dark, "light")  # light absent -> use dark
    assert_equal(pal.BG_BASE, "#101010", "missing mode falls back to available")


# --------------------------------------------------------------------------- #
# theme.py live palette
# --------------------------------------------------------------------------- #
def test_getattr_follows_active() -> None:
    pal = tl.resolve_theme(_MINIMAL, "dark")
    try:
        theme.set_active(pal)
        assert_equal(theme.BG_BASE, pal.BG_BASE, "theme.BG_BASE follows active")
        assert_equal(theme.ACCENT, pal.ACCENT, "theme.ACCENT follows active")
    finally:
        theme.set_active(theme.DEFAULT)
    assert_equal(theme.BG_BASE, theme.DEFAULT.BG_BASE, "restored to default")


def test_css_variables_cover_all_tokens() -> None:
    variables = theme.to_css_variables(theme.DEFAULT)
    assert_equal(len(variables), len(fields(theme.Palette)), "one css var per token")
    assert_equal(variables["bg-base"], theme.DEFAULT.BG_BASE, "bg-base var")
    assert_equal(variables["scroll-bar-hover"], theme.DEFAULT.SCROLL_BAR_HOVER, "kebab-cased")


def test_getattr_rejects_unknown() -> None:
    try:
        theme.NOT_A_TOKEN  # noqa: B018
    except AttributeError:
        return
    raise AssertionError("unknown attribute should raise AttributeError")


# --------------------------------------------------------------------------- #
# Registry over the real bundled themes
# --------------------------------------------------------------------------- #
def test_catalog_lists_builtin_first_and_all_bundled() -> None:
    cat = theme_registry.load_catalog()
    assert_equal(cat.names[0], theme_registry.BUILTIN_NAME, "built-in default listed first")
    assert_true(len(cat.names) >= 2, "catalog has themes")
    # Every listed theme resolves to valid tokens in every mode it offers.
    for name in cat.names:
        for mode in cat.available_modes(name):
            pal = cat.palette(name, mode)
            for f in fields(theme.Palette):
                assert_true(_HEX.match(getattr(pal, f.name)), f"{name}/{mode}/{f.name}")


def test_catalog_textual_name_and_mode_fallback() -> None:
    cat = theme_registry.load_catalog()
    assert_equal(cat.available_modes(theme_registry.BUILTIN_NAME), ["dark"], "builtin dark-only")
    # builtin has no light -> textual_name falls back to its dark registration
    assert_equal(
        cat.textual_name(theme_registry.BUILTIN_NAME, "light"),
        theme_registry.BUILTIN_NAME,
        "builtin light falls back to dark name",
    )


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("theme loader/registry tests passed")


if __name__ == "__main__":
    main()
