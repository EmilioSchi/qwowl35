"""Serve command transport hooks, mirroring hashline's ``commands/serve.rs``."""

from __future__ import annotations

from pathlib import Path


def default_daemon_socket() -> Path:
    return Path.home() / ".hashline" / "hashline.sock"


def handle_http(*args, **kwargs) -> None:
    raise NotImplementedError("HTTP daemon transport is replaced by qwowl35 tool calling")


def handle_unix(*args, **kwargs) -> None:
    raise NotImplementedError("Unix daemon transport is replaced by qwowl35 tool calling")


def run(*args, **kwargs) -> None:
    raise NotImplementedError("serve transport is not used by qwowl35")
