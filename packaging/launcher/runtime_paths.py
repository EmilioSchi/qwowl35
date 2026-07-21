"""Path resolution for the qw35.app launcher, frozen (PyInstaller) or dev.

Dev mode runs this package straight from the repo checkout
(`python packaging/launcher/qw35_launcher.py`), resolving against the repo
tree; frozen mode resolves against the PyInstaller bundle (`sys._MEIPASS`
points at the onedir `_internal` directory, where the spec ships `qwowl35/`,
`setup_page/` and `bin/qw35`).
"""

from __future__ import annotations

import os
import sys

_LAUNCHER_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_LAUNCHER_DIR))


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _resources() -> str:
    return sys._MEIPASS if is_frozen() else REPO_ROOT


def qwowl_pkg_dir() -> str:
    if is_frozen():
        return os.path.join(_resources(), "qwowl35")
    return os.path.join(REPO_ROOT, "qw35-agent", "qwowl35")


def server_binary() -> str:
    if is_frozen():
        return os.path.join(_resources(), "bin", "qw35")
    return os.path.join(REPO_ROOT, "target", "release", "qw35")


def setup_page_dir() -> str:
    if is_frozen():
        return os.path.join(_resources(), "setup_page")
    return os.path.join(_LAUNCHER_DIR, "setup_page")


def app_support_dir() -> str:
    # Launcher-only test override; the agent's own config stays CLI-only.
    override = os.environ.get("QW35_APP_SUPPORT_DIR")
    if override:
        return override
    return os.path.expanduser("~/Library/Application Support/qw35")


def gguf_dir() -> str:
    return os.path.join(app_support_dir(), "gguf")


def log_dir() -> str:
    return os.path.expanduser("~/Library/Logs/qw35")


def self_invoke_argv() -> list[str]:
    """Argv prefix that re-runs this launcher, frozen or dev."""
    if is_frozen():
        return [sys.executable]
    return [sys.executable, os.path.join(_LAUNCHER_DIR, "qw35_launcher.py")]
