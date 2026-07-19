"""Tool-call argument parsing: streamed JSON/XML recovery, arg compaction,
and the tool-name sets that drive per-family rendering."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from client import parse_xml_tool_args, recover_json_string_field_object, recover_xml_parameter

if TYPE_CHECKING:
    from widgets.chat.tool_block import ToolBlock


_FILE_CHANGE_TOOLS = {
    "replace",
    "edit",  # pre-rename transcripts; live "edit" is the freestyle delegator
    "insert",
    "delete",
}
_FILE_READ_TOOLS = {"read_file", "beginTransaction"}  # old name: pre-rename transcripts
_FILE_VIEW_TOOLS = _FILE_READ_TOOLS | _FILE_CHANGE_TOOLS


# The shell tool's names (trained wire name + legacy alias): both get the
# full terminal treatment — `$`-prompt command box, bash syntax highlighting,
# and the type-out reveal animation.
_SHELL_TOOL_NAMES = frozenset({"bash", "run_shell_command"})


def _parse_args(buffer: str) -> dict[str, Any]:
    """Parse streamed tool-call JSON, tolerating partial wrappers."""
    if not buffer.strip():
        return {}
    try:
        parsed = json.loads(buffer)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = buffer.find("{")
        end = buffer.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(buffer[start:end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                pass
    parsed_xml = parse_xml_tool_args(buffer)
    if parsed_xml is not None:
        return parsed_xml
    return {}


def _compact_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    parts: list[str] = []
    for key in (
        "file", "path", "url", "pattern", "query", "question", "meaning",
        "symbol", "id", "position", "mode", "from", "to",
    ):
        value = args.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value!r}")
    if not parts and "command" in args:
        parts.append(f"command={args['command']!r}")
    return " ".join(parts)


def _recover_string_arg(buffer: str, field: str) -> str | None:
    """One string argument of a streaming call, recovered from the partial XML
    (or final JSON) the same way the bash command preview is — so long text
    args (an edit's `content`, an explorer resume's `summary`) type out live
    instead of hiding until their parameter closes."""
    args = _parse_args(buffer)
    value = args.get(field)
    if isinstance(value, str):
        return value
    recovered = recover_json_string_field_object(buffer, field)
    if recovered is not None:
        return recovered
    return recover_xml_parameter(buffer, field, partial=True)


def _closed_string_arg(buffer: str, field: str) -> str | None:
    """A string arg only once its value is COMPLETE (full JSON parse or a
    closed XML parameter) — never a half-streamed prefix like "12-1" of
    "12-18". Deliberately no brace-slice fallback: that path can parse a
    partial JSON buffer and hand back a truncated value."""
    try:
        parsed = json.loads(buffer)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        value = parsed.get(field)
        return value if isinstance(value, str) else None
    return recover_xml_parameter(buffer, field, partial=False)


def _spawn_chip(block: ToolBlock) -> str | None:
    """The sub-agent a call spawns ("Explorer"/"Editor"), or None.

    `explore` always spawns the Explorer. `edit` is two different tools that
    share a name: the freestyle delegator (filename/line_ranges/instructions —
    spawns the Editor) and the editor's own hashline tool (file/content) —
    discriminate by arg shape, never by name, or the editor's edits would be
    mis-carded as spawns.
    """
    if block.tool_name == "explore":
        return "Explorer"
    if block.tool_name == "edit":
        if _recover_string_arg(block.args_buf, "content") is not None:
            return None  # hashline editor tool
        args = _parse_args(block.args_buf)
        if (
            "instructions" in args
            or "line_ranges" in args
            or "filename" in args
            or _recover_string_arg(block.args_buf, "instructions") is not None
        ):
            return "Editor"
    return None


def _command_from_args(buffer: str) -> str:
    args = _parse_args(buffer)
    command = args.get("command")
    if isinstance(command, str):
        return command
    recovered = recover_json_string_field_object(buffer, "command")
    if recovered is not None:
        return recovered
    recovered_xml = recover_xml_parameter(buffer, "command", partial=True)
    if recovered_xml is not None:
        return recovered_xml
    match = re.search(r'"command"\s*:\s*"((?:\\.|[^"\\])*)', buffer, re.DOTALL)
    if not match:
        return buffer.strip()
    try:
        return json.loads(f'"{match.group(1)}"')
    except Exception:
        return match.group(1)


def _path_from_args(args: dict[str, Any]) -> str:
    path = args.get("file") or args.get("path") or args.get("file_path")
    return path if isinstance(path, str) else ""
