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


def _strip_ui_args(argv: list[str]) -> list[str]:
    """Drop --ui/--ui-port (and their values) from an argv slice."""
    out: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
        elif arg in ("--ui", "--ui-port"):
            skip = True
        elif not arg.startswith(("--ui=", "--ui-port=")):
            out.append(arg)
    return out


def _child_command(extra: list[str]) -> list[str]:
    """Relaunch this entry point with the caller's flags, minus the UI choice."""
    return [sys.executable, os.path.join(_HERE, "__main__.py"), *_strip_ui_args(sys.argv[1:]), *extra]


def _require_textual_serve():
    try:
        from textual_serve.server import Server
    except ImportError:
        sys.exit(
            "--ui webgui/gui needs the optional textual-serve package:\n"
            "    pip install textual-serve"
        )
    return Server


def _webui_overrides(tmp_dir: str) -> dict:
    """Statics/templates overriding textual-serve's stock page with webui/.

    The page is served from a merged scratch copy: the package's bundled assets
    (js/css/images) plus the Mononoki Nerd Font files vendored in webui/fonts/,
    with webui/app_index.html as the template. Falls back to the stock page
    (Roboto Mono from Google Fonts) when webui/ is incomplete.
    """
    import shutil

    import textual_serve

    webui = os.path.join(_HERE, "webui")
    fonts_dir = os.path.join(webui, "fonts")
    if not os.path.isfile(os.path.join(webui, "app_index.html")) or not (
        os.path.isdir(fonts_dir) and any(f.endswith(".woff2") for f in os.listdir(fonts_dir))
    ):
        print("webui/ template or fonts missing — serving the stock textual-serve page")
        return {}

    statics = os.path.join(tmp_dir, "static")
    shutil.copytree(os.path.join(os.path.dirname(textual_serve.__file__), "static"), statics)
    for name in os.listdir(fonts_dir):
        if name.endswith(".woff2"):
            shutil.copy(os.path.join(fonts_dir, name), os.path.join(statics, "fonts", name))
    return {"statics_path": statics, "templates_path": webui}


def _serve_webgui(port: int | None) -> None:
    """Serve the unchanged TUI in the browser; one app subprocess per tab."""
    import shlex
    import tempfile

    Server = _require_textual_serve()
    with tempfile.TemporaryDirectory(prefix="qwowl35-webui-") as tmp_dir:
        Server(
            shlex.join(_child_command([])),
            port=port or 8000,
            title="qwowl35",
            **_webui_overrides(tmp_dir),
        ).serve()


def _serve_gui(port: int | None) -> None:
    """--ui webgui in a subprocess, wrapped in a native desktop window."""
    import socket
    import subprocess
    import time

    _require_textual_serve()  # fail here with the install hint, not in the child
    if port is None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    url = f"http://localhost:{port}"

    server = subprocess.Popen(_child_command(["--ui", "webgui", "--ui-port", str(port)]))
    try:
        deadline = time.monotonic() + 30.0
        while True:
            if server.poll() is not None:
                sys.exit(f"web UI server exited (code {server.returncode}) before {url} came up")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    sys.exit(f"web UI server did not open {url} within 30s")
                time.sleep(0.25)

        try:
            import webview  # pywebview — optional native window chrome
        except ImportError:
            import webbrowser

            print(
                f"pywebview not installed (pip install pywebview) — opening {url} "
                "in the default browser instead; Ctrl+C quits"
            )
            webbrowser.open(url)
            server.wait()
        else:
            webview.create_window("qwowl35", url, width=1200, height=800)
            webview.start()  # returns when the window is closed
    except KeyboardInterrupt:
        pass
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(5)
            except subprocess.TimeoutExpired:
                server.kill()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="qwowl35", description="Minimal qw35 coding agent TUI")
    parser.add_argument("--base-url", help="qw35-server base URL (default http://127.0.0.1:8080)")
    parser.add_argument(
        "--ui",
        choices=["tui", "gui", "webgui"],
        default="tui",
        help="render target: tui = this terminal (default); webgui = the same UI "
        "served in the browser (optional textual-serve package); gui = webgui "
        "wrapped in a native desktop window (pywebview when installed, else the "
        "default browser)",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        help="port for --ui webgui/gui (default: 8000 for webgui, a free port for gui)",
    )
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

    if parsed.ui == "webgui":
        _serve_webgui(parsed.ui_port)
        return
    if parsed.ui == "gui":
        _serve_gui(parsed.ui_port)
        return

    from app import QwowlApp
    from config import load_config

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
