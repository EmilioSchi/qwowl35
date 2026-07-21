#!/usr/bin/env python3
"""qw35.app entry point — GUI launcher and frozen child dispatcher.

Modes (see packaging/README.md):
  qw35_launcher.py                                  GUI: setup window -> downloads -> server -> agent
  qw35_launcher.py --qw35-dispatch agent-serve ...  textual-serve host (child of the GUI)
  qw35_launcher.py --qw35-dispatch agent-tui ...    one qwowl35 app instance (child of agent-serve,
                                                    one per browser tab)

The dispatch modes exist because qwowl35's own child-spawn re-invokes
`sys.executable __main__.py`, which cannot work from a PyInstaller bundle;
here the frozen executable re-invokes itself instead.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import runtime_paths  # noqa: E402


def _redirect_stdio() -> None:
    """Windowed PyInstaller apps run with sys.stdout/stderr = None; any stray
    print() would raise. Send both to a log file instead."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    log_dir = runtime_paths.log_dir()
    os.makedirs(log_dir, exist_ok=True)
    log = open(os.path.join(log_dir, "launcher.log"), "a", buffering=1)
    sys.stdout = sys.stdout or log
    sys.stderr = sys.stderr or log


def _run_agent_tui(rest: list[str]) -> None:
    """The tail of qwowl35.__main__.main(): load config, run the Textual app.

    Duplicated (~10 lines) rather than imported — qwowl35/__main__.py builds
    child commands from its own file path, which is frozen-hostile.
    """
    import argparse

    pkg = runtime_paths.qwowl_pkg_dir()
    if pkg not in sys.path:
        sys.path.insert(0, pkg)

    parser = argparse.ArgumentParser(prog="qw35 --qw35-dispatch agent-tui")
    parser.add_argument("--base-url")
    ns = parser.parse_args(rest)

    from app import QwowlApp
    from config import load_config

    QwowlApp(load_config(base_url=ns.base_url)).run()


def _read_setup_html() -> str:
    page_dir = runtime_paths.setup_page_dir()

    def read(name: str) -> str:
        with open(os.path.join(page_dir, name), encoding="utf-8") as fh:
            return fh.read()

    return (
        read("index.html")
        .replace("/*{{CSS}}*/", read("setup.css"))
        .replace("/*{{JS}}*/", read("setup.js"))
    )


def run_gui() -> None:
    import atexit
    import signal

    import webview

    from setup_api import SetupApi

    api = SetupApi()
    window = webview.create_window(
        "qw35",
        html=_read_setup_html(),
        js_api=api,
        width=1200,
        height=800,
        min_size=(700, 500),
    )
    api.bind_window(window)
    # Reap the server/agent trees on every exit path: window close, Cmd-Q
    # (atexit), and SIGTERM/SIGINT (kill, Ctrl+C) — shutdown() is idempotent.
    window.events.closed += api.shutdown
    atexit.register(api.shutdown)

    def _on_signal(signum, frame):
        api.shutdown()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    webview.start()  # returns when the window closes; must run on the main thread
    api.shutdown()


def main() -> None:
    _redirect_stdio()
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "--qw35-dispatch":
        mode, rest = argv[1], argv[2:]
        if mode == "agent-tui":
            _run_agent_tui(rest)
        elif mode == "agent-serve":
            import procs

            procs.run_agent_serve(rest)
        else:
            sys.exit(f"unknown dispatch mode: {mode}")
        return
    run_gui()


if __name__ == "__main__":
    main()
