"""Command context helpers, mirroring hashline's ``context.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class OutputMode(Enum):
    Pretty = "pretty"
    Json = "json"
    Ndjson = "ndjson"


@dataclass
class CommandContext:
    output_mode: OutputMode = OutputMode.Pretty
    json_pretty: bool = False
    modified_doc: Any | None = None

    @classmethod
    def new(cls, stdout=None, stderr=None, output_mode: OutputMode = OutputMode.Pretty) -> "CommandContext":
        return cls(output_mode=output_mode)

    def with_json_pretty(self, pretty: bool) -> "CommandContext":
        self.json_pretty = pretty
        return self

    def with_modified_doc(self, doc) -> "CommandContext":
        self.modified_doc = doc
        return self


def output_mode_for(command) -> OutputMode:
    if getattr(command, "ndjson", False):
        return OutputMode.Ndjson
    if getattr(command, "json", False):
        return OutputMode.Json
    return OutputMode.Pretty


def json_pretty_for(command) -> bool:
    return bool(getattr(command, "pretty", False))


def flag_mode(json: bool = False, ndjson: bool = False) -> OutputMode:
    if ndjson:
        return OutputMode.Ndjson
    if json:
        return OutputMode.Json
    return OutputMode.Pretty


def format_mode(output_mode: OutputMode) -> str:
    return output_mode.value


def json_pretty_flag(pretty: bool) -> bool:
    return pretty
