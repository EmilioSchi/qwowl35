"""Entry point: ``python -m qwowl35`` or ``python qwowl35/__main__.py``.

The ``tools`` package and ``mascot`` module are imported with bare absolute
names, so this directory must be on ``sys.path``. We insert it here before
importing anything else, which makes both launch styles work.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def main() -> None:
    import argparse

    from app import QwowlApp
    from config import load_config

    parser = argparse.ArgumentParser(prog="qwowl35", description="Minimal qw35 coding agent TUI")
    parser.add_argument("--base-url", help="qw35-server base URL (default http://127.0.0.1:8080)")
    parser.add_argument(
        "--think",
        choices=["auto", "on", "off"],
        help="thinking mode: auto defers to the server --mode default, on requests "
        "thinking, off disables it (default auto)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        help="optional thinking budget when --think on: low/medium/high cap the reasoning "
        "budget at 4/10/16%% of max_tokens, xhigh keeps the 16%% backstop (only sent when given)",
    )
    parser.add_argument(
        "--restricted-bash",
        action="store_true",
        default=None,
        help="run the bash tool in restricted mode",
    )
    parsed = parser.parse_args()

    config = load_config(
        base_url=parsed.base_url,
        think=parsed.think,
        reasoning_effort=parsed.reasoning_effort,
        restricted_bash=parsed.restricted_bash,
    )
    QwowlApp(config).run()


if __name__ == "__main__":
    main()
