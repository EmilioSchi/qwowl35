"""pywebview js_api bridge: the setup page's window into the launcher.

JS -> Python: `window.pywebview.api.<method>()` (each call runs on its own
thread, so the long-running ones below spawn worker threads and return
immediately). Python -> JS: `window.evaluate_js("qw35OnEvent({...})")`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading

import downloads
import procs
import runtime_paths


class SetupApi:
    def __init__(self):
        self._window = None
        self._downloader = downloads.Downloader(runtime_paths.gguf_dir(), self._emit)
        self._server: procs.ManagedProc | None = None
        self._agent: procs.ManagedProc | None = None
        self._lock = threading.Lock()  # guards the flags below, never held across waits
        self._launch_started = False
        self._closing = False

    def bind_window(self, window) -> None:
        self._window = window

    def _emit(self, event: dict) -> None:
        window = self._window
        if window is None:
            return
        try:
            window.evaluate_js(f"qw35OnEvent({json.dumps(event)})")
        except Exception:
            pass  # window already gone; the event has nowhere to land

    # --- JS-callable surface -------------------------------------------------

    def get_state(self) -> dict:
        gguf = runtime_paths.gguf_dir()
        os.makedirs(gguf, exist_ok=True)
        models = []
        for spec in downloads.MODELS:
            final = downloads.model_path(gguf, spec)
            part = downloads.part_path(gguf, spec)
            models.append(
                {
                    "name": spec.name,
                    "label": spec.label,
                    "present": os.path.exists(final),
                    "part_size": os.path.getsize(part) if os.path.exists(part) else 0,
                    "fallback_size": spec.fallback_size,
                }
            )
        return {
            "models": models,
            "disk_free": shutil.disk_usage(gguf).free,
            "gguf_dir": gguf,
        }

    def probe_models(self) -> dict:
        return {spec.name: downloads.probe(spec) for spec in downloads.MODELS}

    def start_downloads(self) -> None:
        threading.Thread(target=self._download_flow, daemon=True).start()

    def cancel_downloads(self) -> None:
        self._downloader.cancel_event.set()

    def launch(self) -> None:
        with self._lock:
            if self._launch_started or self._closing:
                return
            self._launch_started = True
        threading.Thread(target=self._launch_flow, daemon=True).start()

    def reveal_models(self) -> None:
        subprocess.run(["open", "-R", runtime_paths.gguf_dir()], check=False)

    def quit(self) -> None:
        if self._window is not None:
            self._window.destroy()

    # --- worker flows --------------------------------------------------------

    def _download_flow(self) -> None:
        self._emit({"type": "phase", "value": "downloading"})
        try:
            completed = self._downloader.run()
        except downloads.DownloadError as exc:
            self._emit({"type": "error", "message": str(exc), "retriable": exc.retriable})
            return
        except Exception as exc:  # anything else still needs to reach the user
            self._emit({"type": "error", "message": f"Unexpected error: {exc}", "retriable": True})
            return
        if completed:
            self._emit({"type": "phase", "value": "downloaded"})
        else:
            self._emit({"type": "phase", "value": "cancelled"})

    def _start_child(self, factory):
        """Spawn a child unless the window is closing; reap it if the close
        raced the spawn."""
        with self._lock:
            if self._closing:
                return None
        proc = factory()
        with self._lock:
            if self._closing:
                proc.terminate()
                return None
        return proc

    def _launch_flow(self) -> None:
        try:
            gguf = runtime_paths.gguf_dir()
            model = downloads.model_path(gguf, downloads.MODELS[0])
            reranker = downloads.model_path(gguf, downloads.MODELS[1])
            if not os.path.exists(reranker):
                reranker = None  # optional; the server runs without /v1/rerank
            server_port = procs.free_port()
            ui_port = procs.free_port()

            self._emit({"type": "phase", "value": "engine-starting"})
            self._server = self._start_child(
                lambda: procs.start_server(model, reranker, server_port)
            )
            if self._server is None:
                return
            procs.wait_server_ready(self._server, server_port)

            self._emit({"type": "phase", "value": "agent-starting"})
            base_url = f"http://127.0.0.1:{server_port}"
            self._agent = self._start_child(
                lambda: procs.start_agent_serve(ui_port, base_url)
            )
            if self._agent is None:
                return
            procs.wait_port(self._agent, ui_port)
        except procs.ProcError as exc:
            self._emit({"type": "error", "message": str(exc), "retriable": False})
            return
        except Exception as exc:
            self._emit({"type": "error", "message": f"Unexpected error: {exc}", "retriable": False})
            return
        self._emit({"type": "phase", "value": "ready"})
        if self._window is not None:
            self._window.load_url(f"http://localhost:{ui_port}")

    def shutdown(self) -> None:
        with self._lock:
            if self._closing:
                return
            self._closing = True
        self._downloader.cancel_event.set()
        for proc in (self._agent, self._server):
            if proc is not None:
                proc.terminate()
