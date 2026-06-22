"""CLI entry helpers, mirroring hashline's ``main.rs`` transport names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SharedFileGuard:
    path: Path


@dataclass
class SharedFileWriter:
    path: Path


def command_to_tool_name(command) -> str:
    return str(command).lower().replace("commands.", "hashline_")


def hashline_home_dir() -> Path:
    return Path.home() / ".hashline"


def default_log_path() -> Path:
    return hashline_home_dir() / "audit.jsonl"


def get_socket_env() -> str | None:
    import os

    return os.environ.get("HASHLINE_SOCKET")


def get_url_env() -> str | None:
    import os

    return os.environ.get("HASHLINE_URL")


def resolve_log_path(path: str | None = None) -> Path:
    return Path(path) if path else default_log_path()


def serialize_command_args(command) -> dict:
    return getattr(command, "__dict__", {"command": command})


def route_via_http(*args, **kwargs):
    raise NotImplementedError("HTTP routing is replaced by qwowl35 tool calling")


def route_via_socket(*args, **kwargs):
    raise NotImplementedError("socket routing is replaced by qwowl35 tool calling")


def init_tracing(*args, **kwargs) -> None:
    return None


def tracing_filter(*args, **kwargs) -> str:
    return "info"


def run(command=None):
    from .orchestration import run as run_command

    return run_command(command) if command is not None else None


def main() -> None:
    return None
