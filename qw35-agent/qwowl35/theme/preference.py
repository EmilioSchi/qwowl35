"""Persist the user's last-chosen theme so it survives across app launches.

The ``/theme`` picker is otherwise session-only. This module remembers the last
committed ``(name, mode)`` on disk and replays it on the next launch, mirroring
the best-effort, atomic-write approach of :mod:`history`.

Two inputs feed the resolved startup theme, highest priority first:

1. The ``QWOWL35_THEME`` environment variable — a manual override for scripts or
   one-off sessions. Value is ``"<name>"`` or ``"<name>:<mode>"`` (e.g.
   ``tokyonight`` or ``tokyonight:light``). The qwowl-specific name avoids the
   collision a generic ``THEME`` would risk. A process can't persist its *own*
   environment, so this is an override only — it is never written back.
2. The persisted preference file, written whenever a ``/theme`` choice is
   committed. Lives under the OS config dir (``platformdirs.user_config_dir``
   — e.g. ``~/Library/Application Support/qwowl35`` on macOS) since a theme is a
   preference, not disposable cache.

Either source is validated against the live catalog; anything unknown (a deleted
theme, a typo'd env value, a corrupt file) falls back to the built-in default so
a bad value can never leave the app themeless.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import platformdirs

ENV_VAR = "QWOWL35_THEME"
_FILE_NAME = "theme.json"


def _pref_path() -> Path:
    """Path to the persisted preference file (dir created best-effort)."""
    base = Path(platformdirs.user_config_dir("qwowl35"))
    base.mkdir(parents=True, exist_ok=True)
    return base / _FILE_NAME


def _split(value: str) -> tuple[str, str | None]:
    """Parse ``"name"`` or ``"name:mode"`` → ``(name, mode | None)``."""
    name, sep, mode = value.strip().partition(":")
    return name.strip(), (mode.strip() or None) if sep else None


def _validate(catalog: Any, name: str, mode: str | None) -> tuple[str, str] | None:
    """Resolve ``(name, mode)`` against the catalog, or ``None`` if unknown.

    An unknown name is rejected; an unknown/omitted mode is snapped to the
    theme's nearest available mode so a name-only value still resolves.
    """
    if not name or name not in catalog.names:
        return None
    available = catalog.available_modes(name)
    if mode not in available:
        mode = available[0] if available else "dark"
    return name, mode


def load(catalog: Any, *, default_name: str, default_mode: str = "dark") -> tuple[str, str]:
    """Resolve the startup theme: env override, then saved file, then default."""
    env = os.environ.get(ENV_VAR)
    if env:
        resolved = _validate(catalog, *_split(env))
        if resolved is not None:
            return resolved

    try:
        data: dict[str, Any] = json.loads(_pref_path().read_text(encoding="utf-8"))
        resolved = _validate(catalog, data.get("name", ""), data.get("mode"))
        if resolved is not None:
            return resolved
    except (OSError, ValueError, TypeError):
        pass

    return _validate(catalog, default_name, default_mode) or (default_name, default_mode)


def save(name: str, mode: str) -> None:
    """Persist ``(name, mode)`` atomically. Best-effort — never raises."""
    try:
        path = _pref_path()
        payload = json.dumps({"name": name, "mode": mode})
        # Write to a sibling temp file then atomically rename, so a crash
        # mid-write can't leave a half-written (unparseable) preference.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".theme-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, path)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except OSError:
        pass
