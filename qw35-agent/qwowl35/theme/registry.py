"""Discover bundled opencode themes and register them with the Textual app.

Themes live in ``qwowl35/theme/themes/*.json`` (opencode desktop-theme JSON).
Each is resolved (via :mod:`theme.loader`) into a :class:`theme.Palette` per mode and
wrapped as a ``textual.theme.Theme`` whose ``variables`` carry the app's custom
CSS tokens (``$bg-base`` …). The current soft-black/teal palette is registered
first as the built-in default ``"qwowl"`` (dark only) so there is always a
working theme even with an empty ``themes/`` directory.

Dark variants register under the theme's bare name (``"tokyonight"``); light
variants under ``"<name>-light"``. The selector shows the bare names and toggles
mode; :meth:`ThemeCatalog.textual_name` resolves a (name, mode) pair to the
registered Textual theme name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from textual.theme import Theme

from . import DEFAULT, Palette, to_css_variables
from . import loader

THEMES_DIR = Path(__file__).resolve().parent / "themes"
BUILTIN_NAME = "qwowl"
MODES = ("dark", "light")


def _to_theme(name: str, palette: Palette, *, dark: bool) -> Theme:
    """Wrap a resolved palette as a Textual Theme carrying the app's CSS vars."""
    return Theme(
        name=name,
        dark=dark,
        primary=palette.ACCENT,
        secondary=palette.ACCENT,
        accent=palette.ACCENT,
        foreground=palette.FG_BRIGHT,
        background=palette.BG_BASE,
        surface=palette.BG_SURFACE,
        panel=palette.CODE_BG,
        success=palette.SUCCESS,
        warning=palette.WARNING,
        error=palette.ERROR,
        variables=to_css_variables(palette),
    )


class ThemeCatalog:
    """User-facing theme names mapped to their resolved palettes per mode."""

    def __init__(self, palettes: dict[str, dict[str, Palette]]) -> None:
        self._palettes = palettes
        others = sorted(n for n in palettes if n != BUILTIN_NAME)
        self.names: list[str] = ([BUILTIN_NAME] if BUILTIN_NAME in palettes else []) + others

    def available_modes(self, name: str) -> list[str]:
        return [m for m in MODES if m in self._palettes.get(name, {})]

    def palette(self, name: str, mode: str) -> Palette:
        """Resolved palette for (name, mode), falling back to the first available mode."""
        modes = self._palettes[name]
        return modes.get(mode) or next(iter(modes.values()))

    def textual_name(self, name: str, mode: str) -> str:
        """Registered Textual theme name for (name, mode) — nearest available mode."""
        avail = self.available_modes(name)
        chosen = mode if mode in avail else (avail[0] if avail else "dark")
        return name if chosen == "dark" else f"{name}-{chosen}"

    def register_all(self, app: Any) -> None:
        """Register every (name, mode) as a Textual theme on ``app``."""
        for name, modes in self._palettes.items():
            for mode, palette in modes.items():
                app.register_theme(
                    _to_theme(self.textual_name(name, mode), palette, dark=(mode == "dark"))
                )


def load_catalog() -> ThemeCatalog:
    """Build the catalog: built-in default plus every ``themes/*.json``."""
    palettes: dict[str, dict[str, Palette]] = {BUILTIN_NAME: {"dark": DEFAULT}}
    if THEMES_DIR.is_dir():
        for path in sorted(THEMES_DIR.glob("*.json")):
            try:
                data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            modes = {
                mode: loader.resolve_theme(data, mode)
                for mode in MODES
                if isinstance(data.get(mode), dict)
            }
            if modes:
                palettes[path.stem] = modes
    return ThemeCatalog(palettes)
