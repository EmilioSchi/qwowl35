"""Replace command, mirroring hashline's ``commands/replace.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from ..commands.common import atomic_write


@dataclass(frozen=True)
class ReplaceReceipt:
    count: int
    file: str


@dataclass
class ReplaceCmd:
    file: Path
    old: str
    new: str
    regex: bool = False


def count_occurrences(content: str, old: str, regex: bool = False) -> int:
    return len(re.findall(old, content)) if regex else content.count(old)


def replace_plain(content: str, old: str, new: str) -> tuple[str, int]:
    return content.replace(old, new), content.count(old)


def replace_regex(content: str, old: str, new: str) -> tuple[str, int]:
    return re.subn(old, new, content)


def replace_full(path: str | Path, old: str, new: str, regex: bool = False) -> ReplaceReceipt:
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")
    updated, count = replace_regex(content, old, new) if regex else replace_plain(content, old, new)
    atomic_write(file_path, updated)
    return ReplaceReceipt(count, str(file_path))


def replace_streaming(path: str | Path, old: str, new: str) -> ReplaceReceipt:
    return replace_full(path, old, new, False)


def stream_replace_text(path: str | Path, old: str, new: str) -> ReplaceReceipt:
    return replace_streaming(path, old, new)


def run(cmd: ReplaceCmd) -> str:
    receipt = replace_full(cmd.file, cmd.old, cmd.new, cmd.regex)
    return f"Replaced {receipt.count} occurrence(s).\n"


def run_dry_run(cmd: ReplaceCmd) -> str:
    content = Path(cmd.file).read_text(encoding="utf-8")
    count = count_occurrences(content, cmd.old, cmd.regex)
    return f"Would replace {count} occurrence(s).\nNo file was written.\n"
