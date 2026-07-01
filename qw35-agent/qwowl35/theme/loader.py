"""Compact port of opencode's theme resolver.

Reads an opencode "desktop-theme" JSON — the schema shipped at
``anomalyco/opencode`` (``{ name, id, light:{palette,overrides},
dark:{palette,overrides} }``) — and derives the small set of colors this app
draws from (a :class:`theme.Palette`).

Unlike opencode's full engine (``resolve.ts`` + ``color.ts``, which expands a
seed palette through OKLch color-space math), this is a deliberately small
linear-RGB derivation: the app only needs ~20 tokens, so we interpolate them
between the theme's background (``neutral``) and foreground (``ink``) plus its
semantic colors. It mirrors the *spirit* of opencode's grey/muted generation and
its ``tint`` blend, not the exact math.

Each mode's ``palette`` provides: ``neutral, ink, primary, success, warning,
error, info`` (always) and ``accent, diffAdd, diffDelete, interactive``
(optional). ``overrides`` are mostly ``syntax-*``/``markdown-*`` tokens the app
doesn't use; we honor only ``text-weak`` (a hand-tuned muted text color) when it
is a plain hex.
"""

from __future__ import annotations

from typing import Any

from . import DEFAULT, Palette

Mode = str  # "dark" | "light"


# --------------------------------------------------------------------------- #
# Color math (linear RGB; hex in, hex out)
# --------------------------------------------------------------------------- #
def hex_to_rgb(value: str) -> tuple[int, int, int] | None:
    """Parse ``#rgb`` / ``#rrggbb`` / ``#rrggbbaa`` → (r, g, b). ``None`` if invalid.

    Alpha is dropped (the app uses opaque fills). Tolerant of a missing ``#``.
    """
    if not isinstance(value, str):
        return None
    s = value.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) not in (6, 8):
        return None
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """(r, g, b) floats/ints → ``#rrggbb`` (rounded, clamped to 0-255)."""
    def clamp(x: float) -> int:
        return max(0, min(255, round(x)))

    return f"#{clamp(r):02x}{clamp(g):02x}{clamp(b):02x}"


def normalize_hex(value: Any) -> str | None:
    """Canonicalize any parseable color to opaque lowercase ``#rrggbb``.

    Drops alpha (the app uses opaque fills) and normalizes case/shorthand, so
    themes that ship ``#100F0F`` or an 8-digit ``#rrggbbaa`` come out uniform.
    Returns ``None`` if ``value`` isn't a usable hex color.
    """
    rgb = hex_to_rgb(value)
    return rgb_to_hex(*rgb) if rgb is not None else None


def mix(base_hex: str, overlay_hex: str, alpha: float) -> str:
    """Linear interpolation ``base → overlay`` by ``alpha`` (opencode ``tint``).

    Falls back to ``base_hex`` if either color is unparseable.
    """
    base = hex_to_rgb(base_hex)
    overlay = hex_to_rgb(overlay_hex)
    if base is None or overlay is None:
        return base_hex
    return rgb_to_hex(
        base[0] + (overlay[0] - base[0]) * alpha,
        base[1] + (overlay[1] - base[1]) * alpha,
        base[2] + (overlay[2] - base[2]) * alpha,
    )


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def _hex_or(value: Any, fallback: str) -> str:
    """Return ``value`` normalized to ``#rrggbb`` if usable, else ``fallback``."""
    norm = normalize_hex(value)
    return norm if norm is not None else fallback


def resolve_theme(theme_json: dict[str, Any], mode: Mode) -> theme.Palette:
    """Derive a :class:`theme.Palette` from one opencode theme JSON in ``mode``.

    ``mode`` must be ``"dark"`` or ``"light"``. Missing optional palette keys and
    non-hex override values fall back gracefully; the result is always valid.
    """
    section = theme_json.get(mode) or theme_json.get("dark") or theme_json.get("light") or {}
    palette = section.get("palette", {}) if isinstance(section, dict) else {}
    overrides = section.get("overrides", {}) if isinstance(section, dict) else {}

    d = DEFAULT  # per-token fallbacks when a source color is absent

    neutral = _hex_or(palette.get("neutral"), d.BG_BASE)   # background base
    ink = _hex_or(palette.get("ink"), d.FG_BRIGHT)         # foreground base
    primary = _hex_or(palette.get("primary"), d.ACCENT)
    accent = _hex_or(palette.get("accent"), primary)
    success = _hex_or(palette.get("success"), d.SUCCESS)
    warning = _hex_or(palette.get("warning"), d.WARNING)
    error = _hex_or(palette.get("error"), d.ERROR)
    diff_add = _hex_or(palette.get("diffAdd"), success)
    diff_delete = _hex_or(palette.get("diffDelete"), error)

    # A hand-tuned muted text color, when the theme provides one; else derived.
    text_weak = _hex_or(overrides.get("text-weak"), mix(ink, neutral, 0.30))

    return Palette(
        BG_BASE=neutral,
        BG_SURFACE=mix(neutral, ink, 0.06),      # subtle lift toward text
        FG_BRIGHT=ink,
        FG_DIM=text_weak,
        FG_MUTED=mix(ink, neutral, 0.38),
        FG_FAINT=mix(ink, neutral, 0.48),
        FG_GHOST=mix(ink, neutral, 0.58),
        ACCENT=accent,
        WARNING=warning,
        ERROR=error,
        ERROR_SOFT=mix(error, ink, 0.30),
        SUCCESS=success,
        SUCCESS_SOFT=mix(success, ink, 0.30),
        SCROLL_BAR=mix(neutral, ink, 0.20),
        SCROLL_BAR_HOVER=mix(neutral, ink, 0.30),
        SCROLL_BAR_ACTIVE=mix(neutral, ink, 0.40),
        CODE_BG=neutral,
        DIFF_ADD_BG=mix(neutral, diff_add, 0.15),
        DIFF_REMOVE_BG=mix(neutral, diff_delete, 0.15),
        DIFF_CONTEXT_BG=neutral,
    )
