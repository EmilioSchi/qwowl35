"""OpenAI-style tool calling adapter for anchored file commands."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xxhash

from .commands.common import atomic_write_document
from .commands.delete import DeleteCmd, run as run_delete
from .commands.edit import EditCmd, run as run_edit
from .commands.insert import InsertCmd, run as run_insert
from .commands.read import ReadCmd, run as run_read
from .anchor import looks_like_range_anchor
from .document import Document
from .error import HashlineError
from .mutation import delete_adjacent_duplicates
from .output import line_view, unified_diff

try:  # syntax warnings are best-effort; never let them break the file tools.
    from ...syntax.checker import (
        check_file_structured,
        language_for_path,
        syntax_report,
    )
except Exception:  # pragma: no cover - defensive fallback

    def syntax_report(path, source):  # type: ignore[misc]
        return ""

    def check_file_structured(path, source):  # type: ignore[misc]
        return []

    def language_for_path(path):  # type: ignore[misc]
        return None


MAX_READ_CHARS = 24_000

# F1: a full (anchorless) re-read of a file already fully read this session is
# suppressed when its byte size moved less than this fraction since that read.
# Anchored reads and the internal ``_force`` flag always bypass. Compile-time
# tunable per the project's no-env-vars convention.
REDUNDANT_READ_SIZE_DELTA = 0.40

# F3: which consecutive-identical-line pairs the post-mutation pass auto-deletes.
# "all_exact" removes EVERY consecutive identical pair, including blank/whitespace
# and bracket-only lines (this WILL collapse PEP8 double-blank lines and repeated
# ``}``/``);`` — see ``_dup_should_drop`` to switch to a smarter predicate).
DUP_POLICY = "all_exact"

# F2: how many error rows the syntax block lists in full before summarising.
MAX_SHOWN_SYNTAX_ANCHORS = 5

# Multi-char lines the "smart" duplicate policy spares because they legitimately
# repeat (stacked block closers). Single-char closers like ``}`` are already
# spared by the length check. Only consulted when DUP_POLICY != "all_exact".
_TRIVIAL_DUP_EXEMPT = {"})", "});", "));", "],", "];", "})));", ")));"}

# Zero-width, collision-proof marker the file tools prepend to a result that
# SUCCEEDED but should reach the model flagged as needing attention (syntax
# errors present). The agent loop strips it and maps it to ``is_error=True``.
# Carried in-band so the ``str`` return contract is unchanged.
TOOL_ATTENTION_MARKER = "\x00qw35:attention\x00"


def mark_attention(text: str) -> str:
    """Prefix ``text`` with the attention sentinel (idempotent)."""
    if text.startswith(TOOL_ATTENTION_MARKER):
        return text
    return TOOL_ATTENTION_MARKER + text


def _dup_should_drop(prev: str, curr: str) -> bool:
    """Whether the adjacent identical pair (``prev`` == ``curr``) is auto-deleted.

    Policy is :data:`DUP_POLICY`. "all_exact" drops every consecutive identical
    line, including blanks and bracket-only lines (the user's explicit choice). To
    soften this later, add a branch here — this is the single decision point, e.g.
    ``stripped = curr.strip(); return bool(stripped) and len(stripped) > 1 and
    stripped not in {"}", ");", "})", "],"}``.
    """
    if prev != curr:
        return False
    if DUP_POLICY == "all_exact":
        return True
    # "smart": spare lines that legitimately repeat — blanks/whitespace, single
    # characters (``}``, ``)``), and common stacked closers — and only collapse
    # substantive duplicate content lines.
    stripped = curr.strip()
    if len(stripped) <= 1:
        return False
    if stripped in _TRIVIAL_DUP_EXEMPT:
        return False
    return True


@dataclass(frozen=True)
class ReadRecord:
    """What the model holds after a full read, for the F1 re-read gate."""

    fingerprint: str  # xxh3-64 hexdigest of full content at the last full read
    byte_len: int  # len(content.encode("utf-8")) at that read


class HashlineTools:
    """Model-facing tool-calling adapter around anchored file commands."""

    def __init__(self) -> None:
        self._last_file: str | None = None
        self._read_headered: set[str] = set()
        # file → (fingerprint, byte_len) captured when its anchors were last shown
        # by a FULL read. The fingerprint lets the post-write auto-read tell whether
        # the model still holds CURRENT anchors (suppress) or the file has since
        # changed — by this write, an external editor, or another process — and
        # needs fresh ones (re-show). The byte length powers the F1 re-read gate.
        self._read_records: dict[str, ReadRecord] = {}
        # file → fingerprint already suppressed once this session. Lets a redundant
        # full re-read self-clear: ask again while still unchanged and it is served
        # (a prompt-free escape hatch that also prevents a read loop).
        self._suppressed_once: dict[str, str] = {}

    def schemas(self) -> list[dict]:
        file_schema = {"type": "string", "description": "Path to the file."}
        range_anchor_schema = {
            "type": "string",
            "description": (
                "Qualified anchor copied from read, such as '12:af'; ranges use "
                "'12:af..18:9c'. A range is inclusive of both endpoints: '12:af..18:9c' "
                "covers lines 12 through 18, including 12 and 18."
            ),
        }
        single_anchor_schema = {
            "type": "string",
            "description": "Single qualified anchor copied from read, such as '12:af'. Use position before/after for placement.",
        }
        content_schema = {"type": "string", "description": "Literal replacement or inserted content."}
        return [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read current line anchors before editing.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file": file_schema,
                            "anchor": {
                                "type": "string",
                                "description": "Optional anchor or range for snippet output.",
                            },
                            "context": {
                                "type": "integer",
                                "description": "Context lines around anchors. Defaults to 5.",
                            },
                        },
                        "required": ["file"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit",
                    "description": "Replace one anchored line or range only in an existing non-empty file. This tool cannot create files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file": file_schema,
                            "anchor": range_anchor_schema,
                            "content": content_schema,
                        },
                        "required": ["file", "anchor", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "insert",
                    "description": "Insert before or after one anchor only in an existing non-empty file. This tool cannot create files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file": file_schema,
                            "anchor": single_anchor_schema,
                            "content": content_schema,
                            "position": {
                                "type": "string",
                                "enum": ["after", "before"],
                                "description": "Placement relative to anchor. Defaults to after.",
                            },
                        },
                        "required": ["file", "anchor", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete",
                    "description": "Delete one anchored line or range only from an existing non-empty file. This tool cannot create files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file": file_schema,
                            "anchor": range_anchor_schema,
                        },
                        "required": ["file", "anchor"],
                    },
                },
            },
        ]

    def execute(self, name: str, args: dict[str, Any]) -> str:
        if args.get("_invalid_json") is True:
            message = (
                f"Error: Your {name} call's arguments were not a valid JSON object. "
                "Resend exactly one JSON object."
            )
            detail = args.get("_json_error")
            if isinstance(detail, str) and detail:
                message += f" Details: {detail}."
            return message
        try:
            if name == "read":
                return self.read(args)
            if name == "edit":
                return self.edit(args)
            if name == "insert":
                return self.insert(args)
            if name == "delete":
                return self.delete(args)
            return f"Error: unknown tool {name!r}."
        except FileNotFoundError:
            file = args.get("file") or args.get("path") or self._last_file or ""
            return f"Error: file not found: {file}"
        except HashlineError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Error running {name}: {exc}"

    def read(self, args: dict[str, Any]) -> str:
        file = self._file(args)
        anchors = self._anchors(args)
        # F1: a full (anchorless) re-read of an already-read, barely-changed file is
        # suppressed. Anchored reads are targeted lookups and always run; the
        # internal ``_force`` flag (used by the post-write auto-read) bypasses too.
        if not anchors and not bool(args.get("_force")):
            suppressed = self._maybe_suppress_full_read(file)
            if suppressed is not None:
                return suppressed
        context = int(args.get("context") if args.get("context") is not None else 5)
        output = self._cap(run_read(ReadCmd(file=Path(file), anchor=anchors, context=max(0, context))))
        if not output.strip():
            return output
        # Read the full file once (not the possibly anchored/truncated snippet) so
        # warnings use absolute line numbers and the fingerprint covers the whole
        # file. Only a FULL read records the gate baseline / current anchors — an
        # anchored snippet does not give the model whole-file anchors.
        text = self._read_text(file)
        warnings, attention = self._syntax_section(file, text)
        if not anchors:
            self._read_records[file] = ReadRecord(
                fingerprint=self._content_fingerprint(text),
                byte_len=len(text.encode("utf-8")),
            )
            self._suppressed_once.pop(file, None)
        headered = file in self._read_headered
        self._read_headered.add(file)
        if headered:
            body = self._with_warnings(output, warnings)
        else:
            header = f"{file} (anchors: each line is '<line>:<hash>|<content>'):"
            body = self._with_warnings(f"{header}\n{output}", warnings)
        return mark_attention(body) if attention else body

    def _maybe_suppress_full_read(self, file: str) -> str | None:
        """Informational message for a redundant full re-read, or ``None`` to run it.

        Suppresses only when the file was already fully read this session AND its
        byte size moved < :data:`REDUNDANT_READ_SIZE_DELTA` since. A first read, a
        missing/empty/unreadable file, or a size swing past the gate all return
        ``None``. Self-clears: asking again while still unchanged serves the read.
        """
        if file not in self._read_headered:
            return None
        record = self._read_records.get(file)
        if record is None:
            return None
        text = self._read_text(file)
        if not text:
            return None  # deleted/emptied/unreadable → let the real read surface it
        new_bytes = len(text.encode("utf-8"))
        delta = abs(new_bytes - record.byte_len) / max(record.byte_len, 1)
        if delta >= REDUNDANT_READ_SIZE_DELTA:
            return None  # changed enough that fresh full anchors are worth it
        fingerprint = self._content_fingerprint(text)
        if self._suppressed_once.get(file) == fingerprint:
            return None  # already suppressed this exact state once → serve it now
        self._suppressed_once[file] = fingerprint
        if fingerprint == record.fingerprint:
            return (
                f"Skipped re-reading {file}: it is unchanged since your last read this "
                "session — you still hold its current anchors. Edit with them, read an "
                "anchor for a specific region, or read again to override."
            )
        return (
            f"Skipped re-reading {file}: it changed only ~{delta * 100:.0f}% by size "
            "since your last full read this session. Most anchors still hold, but those "
            "near edits may be stale — read with an anchor for the changed region, or "
            "read again to override."
        )

    def has_current_anchors(self, file: str) -> bool:
        """Whether the model already holds this file's CURRENT line anchors.

        True only when the file was read (anchors shown) AND its content is
        unchanged since — so a file the model read and that was then modified (by a
        bash write, an external editor, or another process) returns False and earns
        a fresh post-write auto-read. Used to suppress redundant auto-reads.
        """
        record = self._read_records.get(file)
        if record is None:
            return False
        return record.fingerprint == self._content_fingerprint(self._read_text(file))

    @staticmethod
    def _content_fingerprint(text: str) -> str:
        return xxhash.xxh3_64_hexdigest(text.encode("utf-8"))

    def _syntax_section(self, file: str, text: str) -> tuple[str, bool]:
        """Trailing syntax block and an attention flag.

        When the file has syntax errors, returns a block that lists each error with
        a ready ``edit anchor: <line>:<hash>|<content>`` so the model can fix that
        exact row, plus ``True`` (the result should be flagged is_error). When clean
        or unknown, returns the existing OK/"" confirmation and ``False``. Reuses
        :func:`check_file_structured` and ``line_view``. Never raises.
        """
        try:
            if not text:
                return "", False
            errors = check_file_structured(file, text)
            if not errors:
                return self._syntax_warnings(file, text), False
            label = language_for_path(file) or "syntax"
            doc = Document.load(Path(file))
            n = len(doc.lines)
            shown = errors[:MAX_SHOWN_SYNTAX_ANCHORS]
            lines = [
                f"Syntax check ({label}) — {len(errors)} issue(s); fix each line "
                "below with edit, then it is re-checked:"
            ]
            for line_no, _col, message in shown:
                lines.append(f"- {message}")
                if 1 <= line_no <= n:
                    lines.append(f"  edit anchor: {line_view(line_no, doc.lines[line_no - 1])}")
            extra = len(errors) - len(shown)
            if extra > 0:
                lines.append(f"- … and {extra} more")
            return "\n".join(lines), True
        except Exception:  # noqa: BLE001 - syntax warnings are best-effort
            return "", False

    def _autodelete_dups(self, file: str) -> list[int]:
        """Remove adjacent exactly-identical lines from ``file`` per :data:`DUP_POLICY`.

        Returns the removed 1-based line numbers (pre-removal numbering), or ``[]``.
        Runs only after a mutation (never on read). Best effort: never raises.
        """
        try:
            doc = Document.load(Path(file))
            removed = delete_adjacent_duplicates(doc, _dup_should_drop)
            if removed:
                atomic_write_document(Path(file), doc)
            return removed
        except Exception:  # noqa: BLE001 - best-effort cleanup must not break the edit
            return []

    def edit(self, args: dict[str, Any]) -> str:
        file = self._file(args)
        anchor = self._anchor(args, required=not self._has_start_query(args))
        content = self._content(args)
        before = self._read_text(file)
        raw = run_edit(
            EditCmd(
                file=Path(file),
                anchor=anchor,
                content=content,
                start_query=self._optional_string(args, "start_query"),
                end_query=self._optional_string(args, "end_query"),
            )
        )
        return self._present_mutation(file, raw, before)

    def insert(self, args: dict[str, Any]) -> str:
        file = self._file(args)
        anchor = self._anchor(args, required=not self._has_start_query(args))
        if anchor and looks_like_range_anchor(anchor):
            left, right = anchor.split("..", 1)
            raise HashlineError(
                "insert requires one line:hash anchor, not a range. "
                f"Use anchor {left!r} with position='after', or anchor {right!r} with position='before'."
            )
        content = self._content(args)
        position = str(args.get("position") or "after").lower()
        if position not in {"before", "after"}:
            raise HashlineError("'position' must be 'before' or 'after'")
        before = self._read_text(file)
        cmd = InsertCmd(
            file=Path(file),
            anchor=anchor,
            content=content,
            before=position == "before",
            start_query=self._optional_string(args, "start_query"),
            end_query=self._optional_string(args, "end_query"),
        )
        return self._present_mutation(file, run_insert(cmd), before)

    def delete(self, args: dict[str, Any]) -> str:
        file = self._file(args)
        anchor = self._anchor(args, required=not self._has_start_query(args))
        before = self._read_text(file)
        raw = run_delete(
            DeleteCmd(
                file=Path(file),
                anchor=anchor,
                start_query=self._optional_string(args, "start_query"),
                end_query=self._optional_string(args, "end_query"),
            )
        )
        return self._present_mutation(file, raw, before)

    def _present_mutation(self, file: str, raw: str, before: str) -> str:
        """Enrich a command result with a unified diff and a refreshed anchor
        snippet so the TUI can render the patch and syntax-highlighted code.

        ``raw`` is the command's own output; its first line is the success
        message. The diff is computed from the file's before/after text, and the
        anchored snippet is rebuilt from the post-edit file so it is uniform
        across edit/insert/delete (the ported commands do not all emit one).
        """
        success = raw.rstrip("\n").split("\n", 1)[0]
        # F3: after the command's own write, auto-delete adjacent identical lines.
        # Done before reading ``after`` so the diff, changed-window snippet, and
        # refreshed anchors all reflect the cleanup. Reads never reach here.
        removed = self._autodelete_dups(file)
        after = self._read_text(file)
        if before != after:
            # The file changed (this edit and/or the dedup), so the model's
            # full-read baseline is stale: drop it so a follow-up full read returns
            # the current file instead of being suppressed by the F1 gate.
            self._read_records.pop(file, None)
            self._suppressed_once.pop(file, None)
        diff = unified_diff(file, before, after)
        # A mutation that leaves the file byte-identical is a no-op: the command
        # still emits "Edited lines X-Y", but nothing changed and there is no diff
        # or changed-window snippet to show. Without an explicit note the model
        # reads the bare success line as a real edit and is left confused when the
        # file "still has issues". Surface the no-op loudly instead. (``removed`` is
        # always empty here: a dedup would have changed the file, so before != after.)
        if before == after:
            # Nothing changed, so there is no diff, no snippet, and no point
            # re-running the syntax check (its status is unchanged too): keep the
            # note short.
            note = (
                f"No changes were made: the content was byte-identical to the "
                f"targeted lines, so {file} is unchanged. Send different content if "
                "you meant to change it."
            )
            return f"{success}\n{note}"
        parts = [success]
        if removed:
            parts.append(
                f"Removed {len(removed)} adjacent duplicate line(s): "
                f"{', '.join(str(n) for n in removed)}."
            )
        if diff and diff != "(no text changes)":
            parts.append("Diff:")
            parts.append(diff)
        window = self._changed_window(before, after)
        if window is not None:
            lo, hi = window
            snippet = self._anchored_snippet(file, lo, hi)
            if snippet:
                parts.append(
                    f"Current {file} (anchors, lines {lo}-{hi}: "
                    "each line is '<line>:<hash>|<content>'):"
                )
                parts.extend(snippet)
        # Report every syntax error now present in the post-mutation file so an
        # edit/insert/delete that broke (or left) the file malformed is visible.
        warnings, attention = self._syntax_section(file, after)
        body = self._with_warnings("\n".join(parts), warnings)
        return mark_attention(body) if attention else body

    def _syntax_warnings(self, file: str, text: str) -> str:
        """Syntax-status block for ``text`` (as ``file``): errors, a clean-parse
        confirmation, or ``""`` when the language is unknown/unchecked."""
        try:
            if not text:
                return ""
            return syntax_report(file, text)
        except Exception:  # noqa: BLE001 - syntax warnings are best-effort
            return ""

    @staticmethod
    def _with_warnings(body: str, warnings: str) -> str:
        if not warnings:
            return body
        return f"{body}\n\n{warnings}"

    def _changed_window(self, before: str, after: str, context: int = 2) -> tuple[int, int] | None:
        """1-based inclusive line range in ``after`` that changed, plus context."""
        after_lines = after.splitlines()
        total = len(after_lines)
        if total == 0:
            return None
        matcher = difflib.SequenceMatcher(a=before.splitlines(), b=after_lines, autojunk=False)
        firsts: list[int] = []
        lasts: list[int] = []
        for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                firsts.append(j1)
                lasts.append(j2)
        if not firsts:
            return None
        lo = max(0, min(firsts) - context)
        hi = min(total, max(lasts) + context)
        if hi <= lo:
            hi = min(total, lo + 1)
        return lo + 1, hi

    def _anchored_snippet(self, file: str, lo: int, hi: int) -> list[str]:
        doc = Document.load(Path(file))
        if not doc.lines:
            return []
        hi = min(hi, len(doc.lines))
        return [line_view(i, doc.lines[i - 1]) for i in range(lo, hi + 1)]

    def _read_text(self, file: str) -> str:
        try:
            return Path(file).read_text(encoding="utf-8")
        except OSError:
            return ""

    def _file(self, args: dict[str, Any]) -> str:
        file = args.get("file") or args.get("path") or self._last_file
        if not isinstance(file, str) or not file:
            raise HashlineError("'file' is required")
        self._last_file = file
        return file

    def _anchor(self, args: dict[str, Any], required: bool = True) -> str:
        anchor = args.get("anchor")
        if not isinstance(anchor, str) or not anchor:
            if not required:
                return ""
            raise HashlineError(
                "'anchor' is required; file mutations cannot create files or target empty files. "
                "Use bash to create the file, then read for anchors before editing."
            )
        return anchor

    def _anchors(self, args: dict[str, Any]) -> list[str]:
        raw = args.get("anchor")
        if raw is None or raw == "":
            return []
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
            return raw
        raise HashlineError("'anchor' must be a string or string list")

    def _content(self, args: dict[str, Any]) -> str:
        content = args.get("content")
        if content is None:
            content = args.get("text")
        if not isinstance(content, str):
            raise HashlineError("'content' is required")
        return content

    def _optional_string(self, args: dict[str, Any], key: str) -> str | None:
        value = args.get(key)
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise HashlineError(f"'{key}' must be a string")
        return value

    def _has_start_query(self, args: dict[str, Any]) -> bool:
        value = args.get("start_query")
        return isinstance(value, str) and value != ""

    def _cap(self, text: str) -> str:
        if len(text) <= MAX_READ_CHARS:
            return text.rstrip("\n")
        return text[:MAX_READ_CHARS].rstrip("\n") + "\n... (truncated; read an anchor with context for the rest)"
