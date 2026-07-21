"""Persist the user's last-chosen web-UI font so it survives across launches.

Mirrors :mod:`theme.preference` (same config dir, same atomic best-effort
writes), minus the mode axis. Two inputs feed the resolved startup font,
highest priority first:

1. The ``QWOWL35_FONT`` environment variable — a manual override for scripts or
   one-off sessions; value is a catalog slug (e.g. ``terminus``). Never written
   back.
2. The persisted preference file, written whenever a ``/fonts`` choice is
   committed.

Either source is validated against :data:`webfonts.CATALOG`; anything unknown
(a removed family, a typo'd env value, a corrupt file) falls back to the
default so a bad value can never leave the page fontless.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import platformdirs

from webfonts import DEFAULT_SLUG, slugs

ENV_VAR = "QWOWL35_FONT"
_FILE_NAME = "font.json"


def _pref_path() -> Path:
    """Path to the persisted preference file (dir created best-effort)."""
    base = Path(platformdirs.user_config_dir("qwowl35"))
    base.mkdir(parents=True, exist_ok=True)
    return base / _FILE_NAME


def load(default_slug: str = DEFAULT_SLUG) -> str:
    """Resolve the startup font slug: env override, then saved file, then default."""
    known = slugs()

    env = os.environ.get(ENV_VAR, "").strip()
    if env in known:
        return env

    try:
        data: dict[str, Any] = json.loads(_pref_path().read_text(encoding="utf-8"))
        saved = data.get("slug", "")
        if saved in known:
            return saved
    except (OSError, ValueError, TypeError, AttributeError):
        pass

    return default_slug


def save(slug: str) -> None:
    """Persist ``slug`` atomically. Best-effort — never raises."""
    try:
        path = _pref_path()
        payload = json.dumps({"slug": slug})
        # Write to a sibling temp file then atomically rename, so a crash
        # mid-write can't leave a half-written (unparseable) preference.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".font-", suffix=".tmp")
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
