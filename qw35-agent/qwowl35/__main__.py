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
    parser.add_argument(
        "--no-compress",
        dest="compress",
        action="store_false",
        default=None,
        help="disable tool-output compression (full raw tool results)",
    )
    parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        default=None,
        help="disable the query-aware semantic rerank of web results "
        "(statistical compression only)",
    )
    parser.add_argument(
        "--rerank-scorer",
        choices=["cross-encoder", "bm25"],
        help="rerank scorer: cross-encoder = the server's native reranker via "
        "/v1/rerank (default; qw35 auto-loads the reranker GGUF when present; "
        "degrades to bm25 when the server has no reranker), bm25 = lexical only",
    )
    parser.add_argument(
        "--no-lsp",
        dest="lsp",
        action="store_false",
        default=None,
        help="disable LSP semantic diagnostics on read/edit results "
        "(tree-sitter syntax checks only)",
    )
    parsed = parser.parse_args()

    config = load_config(
        base_url=parsed.base_url,
        think=parsed.think,
        reasoning_effort=parsed.reasoning_effort,
        restricted_bash=parsed.restricted_bash,
        compress=parsed.compress,
        rerank=parsed.rerank,
        rerank_scorer=parsed.rerank_scorer,
        lsp=parsed.lsp,
    )
    QwowlApp(config).run()


if __name__ == "__main__":
    main()
