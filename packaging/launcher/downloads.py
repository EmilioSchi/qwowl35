"""Model downloads for the qw35.app launcher.

Mirrors download_model.sh conventions: same filenames, partials as
`<name>.part` next to the final file, byte-range resume, atomic rename on
completion. Progress is pushed to the GUI through a callback, throttled to
~4 Hz.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass

import httpx

DISK_MARGIN = 2 * 1024**3  # keep this much free beyond the downloads
CHUNK = 1024 * 1024
PROGRESS_INTERVAL = 0.25
SPEED_WINDOW = 5.0


@dataclass(frozen=True)
class ModelSpec:
    name: str  # final filename in the gguf dir
    label: str  # human-readable description for the GUI
    url: str
    fallback_size: int  # shown when the HEAD request fails
    # Optional models are skipped (with a visible notice) when the host
    # refuses anonymous downloads or the download fails, instead of blocking
    # the app. The reranker is optional: without it the server serves no
    # /v1/rerank and the agent's web-result rerank falls back to BM25.
    optional: bool = False


MODELS = [
    ModelSpec(
        name="Qwowl3.5-9B.gguf",
        label="Qwowl3.5-9B — language model",
        url="https://huggingface.co/EmilioSchi/Qwowl3.5-9B-GGUF/resolve/main/Qwowl3.5-9B.gguf",
        fallback_size=5_170_914_624,
    ),
    ModelSpec(
        name="qwen3-reranker-0.6b-q8_0.gguf",
        label="Qwen3-Reranker-0.6B — web-result reranker",
        url="https://huggingface.co/gpustack/qwen3-reranker-0.6b-GGUF/resolve/main/qwen3-reranker-0.6b-q8_0.gguf",
        fallback_size=639_153_184,
        optional=True,
    ),
]


class DownloadError(Exception):
    """User-facing download failure. `retriable` drives the GUI's Retry."""

    def __init__(self, message: str, *, retriable: bool = True):
        super().__init__(message)
        self.retriable = retriable


def model_path(gguf_dir: str, spec: ModelSpec) -> str:
    return os.path.join(gguf_dir, spec.name)


def part_path(gguf_dir: str, spec: ModelSpec) -> str:
    return model_path(gguf_dir, spec) + ".part"


def fetch_size(spec: ModelSpec, timeout: float = 15.0) -> int | None:
    """Content-Length via HEAD (follows the HF resolve/ redirect to the CDN)."""
    try:
        resp = httpx.head(spec.url, follow_redirects=True, timeout=timeout)
        resp.raise_for_status()
        return int(resp.headers["content-length"])
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def probe(spec: ModelSpec, timeout: float = 15.0) -> dict:
    """HEAD probe: {'size': int, 'available': bool}.

    `available` is False when the host refuses anonymous access (401/403) —
    for optional models the GUI then shows a skip notice instead of a
    download row. Network errors leave availability at True (the download
    itself will surface them with a Retry).
    """
    try:
        resp = httpx.head(spec.url, follow_redirects=True, timeout=timeout)
        if resp.status_code in (401, 403, 404):
            return {"size": spec.fallback_size, "available": False}
        resp.raise_for_status()
        return {"size": int(resp.headers["content-length"]), "available": True}
    except (httpx.HTTPError, KeyError, ValueError):
        return {"size": spec.fallback_size, "available": True}


def check_disk(gguf_dir: str, remaining_bytes: int) -> None:
    free = shutil.disk_usage(gguf_dir).free
    if free < remaining_bytes + DISK_MARGIN:
        need = _human(remaining_bytes + DISK_MARGIN)
        have = _human(free)
        raise DownloadError(
            f"Not enough disk space: the downloads need about {need} free "
            f"(including a safety margin) but only {have} is available.",
            retriable=False,
        )


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def _friendly_http_error(spec: ModelSpec, exc: Exception) -> DownloadError:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403):
            return DownloadError(
                f"Hugging Face refused the download of {spec.name} "
                f"(HTTP {code}). The file may have moved or become gated.\n{spec.url}"
            )
        return DownloadError(f"Download of {spec.name} failed with HTTP {code}.\n{spec.url}")
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return DownloadError(
            "No network connection — check that this Mac is online, then Retry."
        )
    if isinstance(exc, httpx.HTTPError):
        return DownloadError(f"The download of {spec.name} was interrupted ({exc}). Retry resumes it.")
    if isinstance(exc, OSError) and exc.errno == 28:  # ENOSPC
        return DownloadError("The disk filled up during the download.", retriable=False)
    return DownloadError(f"Download of {spec.name} failed: {exc}")


class Downloader:
    """Sequential downloader for the missing models with resume + progress."""

    def __init__(self, gguf_dir: str, emit):
        """`emit(event: dict)` delivers progress/errors to the GUI."""
        self.gguf_dir = gguf_dir
        self.emit = emit
        self.cancel_event = threading.Event()

    def missing(self) -> list[ModelSpec]:
        return [s for s in MODELS if not os.path.exists(model_path(self.gguf_dir, s))]

    def run(self) -> bool:
        """Download every missing model. True on success, False on cancel.

        Raises DownloadError for failures the GUI should show.
        """
        self.cancel_event.clear()
        os.makedirs(self.gguf_dir, exist_ok=True)
        probes = {s.name: probe(s) for s in self.missing()}
        todo = []
        for spec in self.missing():
            if spec.optional and not probes[spec.name]["available"]:
                self.emit({"type": "file-skipped", "file": spec.name})
            else:
                todo.append(spec)
        remaining = sum(
            max(0, probes[s.name]["size"] - self._part_size(s)) for s in todo
        )
        check_disk(self.gguf_dir, remaining)
        for spec in todo:
            try:
                if not self._download_one(spec, probes[spec.name]["size"]):
                    return False
            except DownloadError:
                if not spec.optional:
                    raise
                self.emit({"type": "file-skipped", "file": spec.name})
        return True

    def _part_size(self, spec: ModelSpec) -> int:
        try:
            return os.path.getsize(part_path(self.gguf_dir, spec))
        except OSError:
            return 0

    def _download_one(self, spec: ModelSpec, expected_total: int) -> bool:
        final = model_path(self.gguf_dir, spec)
        part = part_path(self.gguf_dir, spec)
        offset = self._part_size(spec)

        if offset and offset == expected_total:
            os.replace(part, final)
            self.emit({"type": "file-done", "file": spec.name})
            return True

        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with httpx.stream(
                "GET", spec.url, headers=headers, follow_redirects=True, timeout=30.0
            ) as resp:
                if offset and resp.status_code == 200:
                    # Server ignored the Range header: restart from zero.
                    offset = 0
                resp.raise_for_status()
                content_length = resp.headers.get("content-length")
                total = offset + int(content_length) if content_length else expected_total
                mode = "ab" if offset else "wb"
                received = offset
                window: deque[tuple[float, int]] = deque()
                last_emit = 0.0
                with open(part, mode) as fh:
                    if not offset:
                        fh.truncate(0)
                    for chunk in resp.iter_bytes(CHUNK):
                        if self.cancel_event.is_set():
                            return False
                        fh.write(chunk)
                        received += len(chunk)
                        now = time.monotonic()
                        window.append((now, len(chunk)))
                        while window and now - window[0][0] > SPEED_WINDOW:
                            window.popleft()
                        if now - last_emit >= PROGRESS_INTERVAL:
                            last_emit = now
                            span = now - window[0][0] if len(window) > 1 else 0.0
                            speed = sum(n for _, n in window) / span if span > 0 else 0.0
                            self.emit(
                                {
                                    "type": "progress",
                                    "file": spec.name,
                                    "received": received,
                                    "total": total,
                                    "speed_bps": int(speed),
                                    "eta_s": int((total - received) / speed) if speed > 0 else None,
                                }
                            )
        except (httpx.HTTPError, OSError) as exc:
            raise _friendly_http_error(spec, exc) from exc

        got = self._part_size(spec)
        if got != total:
            raise DownloadError(
                f"{spec.name} ended early ({_human(got)} of {_human(total)}). Retry resumes it."
            )
        os.replace(part, final)
        self.emit({"type": "file-done", "file": spec.name})
        return True
