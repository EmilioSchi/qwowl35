"""Command dataclasses, mirroring hashline's ``cli.rs``."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


def default_context() -> int:
    return 5


class MoveDirection(Enum):
    After = "after"
    Before = "before"


@dataclass
class ReadCmd:
    file: Path
    anchor: list[str] = field(default_factory=list)
    context: int = 5
    json: bool = False
    pretty: bool = False
    ndjson: bool = False
    no_cache: bool = False
    compact: bool = False


@dataclass
class EditCmd:
    file: Path
    anchor: str
    content: str


@dataclass
class InsertCmd(EditCmd):
    before: bool = False


@dataclass
class DeleteCmd:
    file: Path
    anchor: str


@dataclass
class IndexCmd:
    file: Path
    compact: bool = False


@dataclass
class GrepCmd:
    file: Path
    pattern: str


@dataclass
class AnnotateCmd:
    file: Path
    query: str


@dataclass
class VerifyCmd:
    file: Path
    anchors: list[str]


@dataclass
class StatsCmd:
    file: Path


@dataclass
class PatchCmd:
    file: Path
    patch: str


@dataclass
class BatchCmd:
    file: Path
    edits: list


@dataclass
class DiffApplyCmd:
    file: Path
    diff: str


@dataclass
class ReplaceCmd:
    file: Path
    old: str
    new: str


@dataclass
class MoveCmd:
    file: Path
    anchor: str
    direction: MoveDirection
    target: str


@dataclass
class SwapCmd:
    file: Path
    anchor_a: str
    anchor_b: str


@dataclass
class IndentCmd:
    file: Path
    anchor: str
    amount: str


@dataclass
class FindBlockCmd:
    file: Path
    line: int


@dataclass
class DoctorCmd:
    file: Path


@dataclass
class ServeCmd:
    pass


@dataclass
class McpCmd:
    pass


@dataclass
class Cli:
    command: object


class Commands(Enum):
    Read = "read"
    Edit = "edit"
    Insert = "insert"
    Delete = "delete"
    Index = "index"
    Grep = "grep"
    Annotate = "annotate"
    Verify = "verify"
    Stats = "stats"
    Patch = "patch"
    Batch = "batch"
    ApplyDiff = "apply_diff"
    Replace = "replace"
    Move = "move"
    Swap = "swap"
    Indent = "indent"
    FindBlock = "find_block"
    Doctor = "doctor"
    Serve = "serve"
    Mcp = "mcp"


def deserialize_edits(value) -> list:
    return value if isinstance(value, list) else json.loads(value)


def parse_edits_json(s: str) -> list:
    return json.loads(s)
