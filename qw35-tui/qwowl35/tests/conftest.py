"""Pytest bootstrap for the consolidated test suite.

The package modules import each other by bare name (flat sys.path quirk), so
put the ``qwowl35`` package dir (this folder's parent) on ``sys.path`` before
any test module is collected. Each test also does this insert itself so it can
still be run directly (``python qwowl35/tests/<name>.py``); this just makes the
same thing happen for pytest regardless of import mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
