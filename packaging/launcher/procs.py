"""Child-process management for the qw35.app launcher.

Two long-lived children, each leading its own session so closing the window
can reap the whole tree (textual-serve spawns one agent instance per tab —
killing just its PID would orphan them; same pattern as qwowl35 `_serve_gui`):

  bin/qw35 -m <model> --reranker-model <reranker> --port P1
  <self> --qw35-dispatch agent-serve --ui-port P2 --base-url http://127.0.0.1:P1
"""

from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import time

import httpx

import runtime_paths


class ProcError(Exception):
    pass


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ManagedProc:
    """Popen wrapper: own session, log file, killpg TERM->KILL teardown.

    Each child also gets a watchdog sidecar that kills the child's process
    group if this (launcher) process dies without running its cleanup — in
    the frozen app the main thread sits inside the native Cocoa event loop,
    where Python signal handlers never fire, so SIGTERM/SIGKILL on the
    launcher would otherwise orphan the whole tree.
    """

    def __init__(self, name: str, cmd: list[str], *, cwd: str | None = None):
        self.name = name
        log_dir = runtime_paths.log_dir()
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{name}.log")
        self._log = open(self.log_path, "ab", buffering=0)
        self._log.write(f"\n--- {name}: {shlex.join(cmd)}\n".encode())
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._log,
            stderr=self._log,
            stdin=subprocess.DEVNULL,
            cwd=cwd,
            start_new_session=True,
        )
        watchdog = (
            f"while kill -0 {os.getpid()} 2>/dev/null; do sleep 2; done; "
            f"kill -TERM -{self.proc.pid} 2>/dev/null; sleep 5; "
            f"kill -KILL -{self.proc.pid} 2>/dev/null"
        )
        self._watchdog = subprocess.Popen(
            ["/bin/sh", "-c", watchdog],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    def alive(self) -> bool:
        return self.proc.poll() is None

    def log_tail(self, lines: int = 30) -> str:
        try:
            with open(self.log_path, "rb") as fh:
                return b"\n".join(fh.read().splitlines()[-lines:]).decode(errors="replace")
        except OSError:
            return "(no log)"

    def _killpg(self, sig: int) -> None:
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            pass

    def terminate(self) -> None:
        # The watchdog first: killing the child before the watchdog would
        # leave the watchdog looping until the launcher exits, and the
        # child's pgid could be reused by then.
        self._watchdog.kill()
        self._watchdog.wait()
        self._killpg(signal.SIGTERM)
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self._killpg(signal.SIGKILL)
        self._log.close()


def start_server(model: str, reranker: str | None, port: int) -> ManagedProc:
    cmd = [runtime_paths.server_binary(), "-m", model, "--port", str(port)]
    if reranker is not None:
        cmd += ["--reranker-model", reranker]
    return ManagedProc("server", cmd, cwd=runtime_paths.gguf_dir())


def wait_server_ready(proc: ManagedProc, port: int, *, timeout: float = 180.0, emit=None) -> None:
    """Poll /health until decoder_ready — the server binds before the model
    engine is usable, so a 200 alone is not readiness."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while True:
        if not proc.alive():
            raise ProcError(
                f"The qw35 server exited during startup (code {proc.proc.returncode}).\n"
                f"Log: {proc.log_path}\n\n{proc.log_tail()}"
            )
        try:
            data = httpx.get(url, timeout=2.0).json()
            if data.get("decoder_ready"):
                return
        except (httpx.HTTPError, ValueError):
            pass
        if time.monotonic() > deadline:
            raise ProcError(
                f"The qw35 server did not become ready within {int(timeout)}s.\n"
                f"Log: {proc.log_path}"
            )
        time.sleep(0.5)


def start_agent_serve(ui_port: int, base_url: str) -> ManagedProc:
    cmd = [
        *runtime_paths.self_invoke_argv(),
        "--qw35-dispatch", "agent-serve",
        "--ui-port", str(ui_port),
        "--base-url", base_url,
    ]
    # Finder launches apps with cwd "/"; agent sessions must not operate there.
    return ManagedProc("agent", cmd, cwd=os.path.expanduser("~"))


def wait_port(proc: ManagedProc, port: int, *, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if not proc.alive():
            raise ProcError(
                f"The agent UI server exited during startup (code {proc.proc.returncode}).\n"
                f"Log: {proc.log_path}\n\n{proc.log_tail()}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            if time.monotonic() > deadline:
                raise ProcError(
                    f"The agent UI server did not open port {port} within {int(timeout)}s.\n"
                    f"Log: {proc.log_path}"
                )
            time.sleep(0.25)


# --- dispatch: agent-serve ---------------------------------------------------
# Adapted from qwowl35.__main__._serve_webgui/_webui_overrides, which cannot be
# imported here: its _child_command re-invokes `sys.executable __main__.py`,
# which is meaningless once frozen. The child command below re-enters this
# launcher via the agent-tui dispatch instead.


def _webui_overrides(tmp_dir: str) -> dict:
    import shutil

    import textual_serve

    webui = os.path.join(runtime_paths.qwowl_pkg_dir(), "webui")
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


def run_agent_serve(rest: list[str]) -> None:
    import argparse
    import tempfile

    from textual_serve.server import Server

    parser = argparse.ArgumentParser(prog="qw35 --qw35-dispatch agent-serve")
    parser.add_argument("--ui-port", type=int, required=True)
    parser.add_argument("--base-url", required=True)
    ns = parser.parse_args(rest)

    child_cmd = [
        *runtime_paths.self_invoke_argv(),
        "--qw35-dispatch", "agent-tui",
        "--base-url", ns.base_url,
    ]
    with tempfile.TemporaryDirectory(prefix="qwowl35-webui-") as tmp_dir:
        Server(
            shlex.join(child_cmd),
            port=ns.ui_port,
            title="qwowl35",
            **_webui_overrides(tmp_dir),
        ).serve()
