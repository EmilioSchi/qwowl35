"""OpenAI-style tool calling adapter for anchored file commands."""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xxhash

from .commands.common import atomic_write_document
from .commands.delete import DeleteCmd, DeleteSummary, run as run_delete
from .commands.edit import EditCmd, EditSummary, run as run_edit
from .commands.insert import InsertCmd, run as run_insert
from .commands.read import ReadCmd, run as run_read
from .anchor import (
    looks_like_range_anchor,
    parse_anchor,
    parse_range,
    resolve,
    resolve_range,
)
from .document import Document
from .error import HashlineError
from .mutation import (
    delete_adjacent_duplicates,
    delete_line,
    delete_range,
    insert_line,
    replace_line,
    replace_range,
    split_content_lines,
)
from .output import line_view, render_lines, unified_diff

try:  # validation is best-effort; never let it break the file tools.
    from ...syntax.validate import Validation, validate_file
except Exception:  # pragma: no cover - defensive fallback

    class Validation:  # type: ignore[no-redef]
        errors: list = []
        warnings: list = []
        label = "syntax"
        checked = False

        def report(self):
            return ""

    def validate_file(path, source):  # type: ignore[misc]
        return Validation()


try:  # diagnostics presentation (section grammar + per-agent dedup); same guard.
    from ...diagnostics import (
        ALL_UNCHANGED,
        DiagnosticsMemory,
        clean_validation_report,
        join_section,
        unchanged_note,
    )
except Exception:  # pragma: no cover - defensive fallback
    DiagnosticsMemory = None  # type: ignore[assignment]
    clean_validation_report = None  # type: ignore[assignment]
    ALL_UNCHANGED = "all unchanged and already reported above"

    def join_section(body: str, section: str) -> str:  # type: ignore[misc]
        return f"{body}\n\n{section}" if body and section else body or section

    def unchanged_note(count: int, noun: str = "issue") -> str:  # type: ignore[misc]
        return f"- {count} unchanged {noun}(s) already reported above (not repeated)"


MAX_READ_CHARS = 24_000

# F3: which consecutive-identical-line pairs the post-mutation pass auto-deletes.
# "smart" (default) spares lines that legitimately repeat — blank/whitespace lines,
# single-character lines (incl. the brackets ``{`` ``}`` ``[`` ``]`` ``(`` ``)``),
# and common stacked closers (``_TRIVIAL_DUP_EXEMPT``) — and only collapses
# substantive duplicate content lines. "all_exact" is the alternative branch: it
# removes EVERY consecutive identical pair, including blanks and bracket-only lines
# (WILL collapse PEP8 double-blank lines and repeated ``}``/``);``). See
# ``_dup_should_drop`` for the single decision point.
DUP_POLICY = "smart"

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

    Policy is :data:`DUP_POLICY` and this is the single decision point. "smart"
    (the shipped default) spares lines that legitimately repeat — blanks/whitespace,
    single-character lines (incl. brackets ``{}[]()``), and common stacked closers —
    and only collapses substantive duplicate content. "all_exact" remains selectable
    and drops every consecutive identical line, blanks and bracket-only lines
    included.
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


@dataclass
class _OpPlan:
    """One resolved edit within a coalesced batch, ready to apply against the
    ORIGINAL document. All indices are 0-based into that original doc, captured
    BEFORE any mutation runs, so applying the batch bottom-to-top never
    invalidates a not-yet-applied op's anchor. See :meth:`HashlineTools.execute_batch`.
    """

    index: int  # emission order (position in the batch's tool_calls)
    name: str  # "replace" | "insert" | "delete"
    mutation: tuple  # tagged args for _apply_op, e.g. ("replace_line", idx, content)
    success_line: str  # terse per-op acknowledgment, in ORIGINAL line numbers
    primary_line: int  # 1-based original line the op targeted (for the batch summary)
    kind: str  # "consuming" (replace/delete) | "insert"
    pos: int  # primary apply-order key: start index, or the insert position
    tie_rank: int  # 0 consuming, 1 insert — consuming applies first at a tie
    span: tuple[int, int]  # inclusive original indices a consuming op occupies
    anchor_idx: int  # the resolved anchor index (insert: the line it anchors to)
    boundary: int  # insert only: the gap index the new line lands at


class HashlineTools:
    """Model-facing tool-calling adapter around anchored file commands."""

    def __init__(self) -> None:
        self._last_file: str | None = None
        self._read_headered: set[str] = set()
        # file → (fingerprint, byte_len) captured when its anchors were last shown
        # by a FULL read. The fingerprint lets the post-write auto-read tell whether
        # the model still holds CURRENT anchors (suppress) or the file has since
        # changed — by this write, an external editor, or another process — and
        # needs fresh ones (re-show).
        self._read_records: dict[str, ReadRecord] = {}
        # Which diagnostic rows the CURRENT agent instance has already been shown
        # (per-agent, NOT per-session like the read records above): the
        # orchestrator repoints this at each running agent's own store, so a
        # fresh editor spawn sees a broken file's diagnostics once in full while
        # repeat validations within one agent only report what changed.
        self.diag_memory = DiagnosticsMemory() if DiagnosticsMemory is not None else None

    def schemas(self) -> list[dict]:
        file_schema = {"type": "string", "description": "Path to the file."}
        range_id_schema = {
            "type": "string",
            "description": (
                "Line id copied from read_file, such as '12af'; ranges use "
                "'12af..189c'. A range is inclusive of both endpoints: '12af..189c' "
                "covers lines 12 through 18, including 12 and 18."
            ),
        }
        single_id_schema = {
            "type": "string",
            "description": "Single line id copied from read_file, such as '12af'. Use position before/after for placement.",
        }
        content_schema = {"type": "string", "description": "Literal replacement or inserted content."}
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Open a file and read its line ids: each row is "
                        "'<line><hash>|<content>'."
                        "If the file is large the result is truncated; page through it "
                        "with 'offset' and 'limit'."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": (
                                    "The absolute path to the file to read (e.g., "
                                    "'/home/user/project/file.txt'). Relative paths are not "
                                    "supported. You must provide an absolute path."
                                ),
                            },
                            "offset": {
                                "type": "integer",
                                "description": (
                                    "Optional: For text files, the 0-based line number to "
                                    "start reading from. Requires 'limit' to be set. Use for "
                                    "paginating through large files."
                                ),
                            },
                            "limit": {
                                "type": "integer",
                                "description": (
                                    "Optional: For text files, maximum number of lines to "
                                    "read. Use with 'offset' to paginate through large "
                                    "files. If omitted, reads the entire file (if feasible, "
                                    "up to a default limit)."
                                ),
                            },
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "replace",
                    "description": "Replace one line or range by id only in an existing non-empty file. This tool cannot create files. Prefer this over delete+insert when changing existing code — editing in place keeps surrounding ids stable.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file": file_schema,
                            "id": range_id_schema,
                            "content": content_schema,
                        },
                        "required": ["file", "id", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "insert",
                    "description": "Insert before or after one line id only in an existing file. This tool cannot create files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file": file_schema,
                            "id": single_id_schema,
                            "content": content_schema,
                            "position": {
                                "type": "string",
                                "enum": ["after", "before"],
                                "description": "Placement relative to the id. Defaults to after.",
                            },
                        },
                        "required": ["file", "id", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete",
                    "description": "Delete one line or range by id only from an existing non-empty file. This tool cannot create files. Use this only to remove code outright; to change a line, use replace instead of delete+insert.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file": file_schema,
                            "id": range_id_schema,
                        },
                        "required": ["file", "id"],
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
            if name == "read_file":
                return self.begin_transaction(args)
            if name == "replace":
                return self.edit(args)
            if name == "insert":
                return self.insert(args)
            if name == "delete":
                return self.delete(args)
            return f"Error: unknown tool {name!r}."
        except FileNotFoundError:
            file = args.get("file_path") or args.get("file") or self._last_file or ""
            return f"Error: file not found: {file}"
        except HashlineError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Error running {name}: {exc}"

    def execute_batch(self, ops: list[tuple[str, dict[str, Any]]]) -> list[str]:
        """Apply a group of edit calls (replace/insert/delete) that arrived
        together in ONE assistant turn against the SAME file as a single unit:
        one atomic write, one diff, one syntax/LSP pass — instead of one of each
        per call. Returns one result string per input op, aligned by index.

        All ids in a parallel batch come from the same pre-turn snapshot, so
        every op is resolved against the ORIGINAL document (loaded once) and the
        applicable ops are applied bottom-to-top (descending position): a
        mutation at position p only shifts indices >= p, so no not-yet-applied
        op's anchor is invalidated. This is strictly more faithful than the
        sequential path, where op #2 must fuzzy-relocate against op #1's write.

        Overlap policy is greedy, emission-order-wins: the first op of a
        conflict applies; a later op overlapping it gets a clear per-op error
        (so two identical replaces apply once, and duplicate inserts collapse to
        one copy). Per-op resolution failures (stale anchor, bad range) return a
        plain ``Error:`` string on that op, matching the single-call path
        (is_error stays False). The combined diff + refreshed-ids snippet + one
        syntax block ride on the LAST applicable op's result via
        :meth:`_present_mutation`; earlier applicable ops get a terse success
        line in the file's ORIGINAL line numbers. Never raises.
        """
        n = len(ops)
        results: list[str | None] = [None] * n
        # The runner groups by absolute path, so every op names the same file;
        # pick the first usable one and resolve everything against its document.
        file = next(
            (
                a.get("file")
                for _name, a in ops
                if isinstance(a, dict) and isinstance(a.get("file"), str) and a.get("file")
            ),
            None,
        )
        if file is None:  # defensive: the runner shouldn't group file-less edits
            return [self.execute(name, a) for name, a in ops]
        self._last_file = file
        try:
            self._require_transaction(file, "edit")
            before = self._read_text(file)
            doc = Document.load(Path(file))
        except FileNotFoundError:
            return [f"Error: file not found: {file}"] * n
        except HashlineError as exc:
            return [f"Error: {exc}"] * n

        if before == "":
            # An empty file has no anchors to batch against: route each op
            # through the single-op path, where the first insert/replace
            # populates it (any id accepted). Later ops then see the filled file.
            return [self.execute(name, a) for name, a in ops]

        index = doc.build_index()  # ONE index over the ORIGINAL doc
        plans: list[_OpPlan | None] = [None] * n
        for k, (name, a) in enumerate(ops):
            if not isinstance(a, dict) or a.get("_invalid_json") is True:
                results[k] = (
                    f"Error: Your {name} call's arguments were not a valid JSON "
                    "object. Resend exactly one JSON object."
                )
                continue
            try:
                plans[k] = self._plan_op(k, name, a, doc, index)
            except HashlineError as exc:
                results[k] = f"Error: {exc}"
            except Exception as exc:  # noqa: BLE001 - report, never crash the batch
                results[k] = f"Error running {name}: {exc}"

        # Overlap resolution in emission order: the first op of a conflict wins.
        kept: list[_OpPlan] = []
        for k in range(n):
            plan = plans[k]
            if plan is None:
                continue
            clash = self._conflict(plan, kept)
            if clash is not None:
                results[k] = f"Error: {self._conflict_message(clash)}"
            else:
                kept.append(plan)
        if not kept:
            return [r if r is not None else "Error: not applied." for r in results]

        # Apply bottom-to-top: highest position first so lower positions stay
        # valid; at a shared position a consuming op (replace/delete) applies
        # before an insert, so the insert lands relative to the final content.
        for plan in sorted(kept, key=lambda p: (-p.pos, p.tie_rank)):
            self._apply_op(doc, plan)
        atomic_write_document(Path(file), doc)  # ONE write

        last_index = kept[-1].index  # last APPLICABLE op, in emission order
        touched = ", ".join(str(x) for x in sorted({p.primary_line for p in kept}))
        summary = (
            f"Applied {len(kept)} grouped edit{'s' if len(kept) != 1 else ''} to "
            f"{file} (lines {touched})."
        )
        for plan in kept:
            if plan.index == last_index:
                # ONE dedup + ONE diff (original before -> final) + ONE window +
                # ONE syntax/LSP pass, reusing the single-call presentation.
                results[plan.index] = self._present_mutation(file, summary + "\n", before)
            else:
                results[plan.index] = plan.success_line
        return [r if r is not None else "Error: not applied." for r in results]

    def _plan_op(
        self, k: int, name: str, args: dict[str, Any], doc: Document, index
    ) -> _OpPlan:
        """Resolve one batch op against the ORIGINAL doc into an :class:`_OpPlan`.

        Mirrors the resolve+summary branching of the single-call command
        ``run()`` bodies (commands/edit.py, insert.py, delete.py) but captures
        indices without mutating or writing, so the whole group can be applied
        together afterwards. Raises HashlineError on a stale/invalid anchor.
        """
        if name == "replace":
            anchor = self._id(args, required=True)
            content = self._content(args)
            if looks_like_range_anchor(anchor):
                start, end = resolve_range(parse_range(anchor), doc, index)
                after = split_content_lines(content)
                success = EditSummary.range(start.line_no, end.line_no, after).success_message()
                return _OpPlan(
                    index=k, name=name,
                    mutation=("replace_range", start.index, end.index, content),
                    success_line=success, primary_line=start.line_no, kind="consuming",
                    pos=start.index, tie_rank=0, span=(start.index, end.index),
                    anchor_idx=start.index, boundary=start.index,
                )
            resolved = resolve(parse_anchor(anchor), doc, index)
            if "\n" in content or "\r" in content:
                after = split_content_lines(content)
                success = EditSummary.range(resolved.line_no, resolved.line_no, after).success_message()
                mutation = ("replace_range", resolved.index, resolved.index, content)
            else:
                success = EditSummary.single(resolved.line_no).success_message()
                mutation = ("replace_line", resolved.index, content)
            return _OpPlan(
                index=k, name=name, mutation=mutation, success_line=success,
                primary_line=resolved.line_no, kind="consuming", pos=resolved.index,
                tie_rank=0, span=(resolved.index, resolved.index),
                anchor_idx=resolved.index, boundary=resolved.index,
            )
        if name == "insert":
            anchor = self._id(args, required=True)
            if looks_like_range_anchor(anchor):
                left, right = anchor.split("..", 1)
                raise HashlineError(
                    "insert requires one line id, not a range. "
                    f"Use id {left!r} with position='after', or id {right!r} with position='before'."
                )
            content = self._content(args)
            position = str(args.get("position") or "after").lower()
            if position not in {"before", "after"}:
                raise HashlineError("'position' must be 'before' or 'after'")
            resolved = resolve(parse_anchor(anchor), doc, index)
            insert_at = resolved.index if position == "before" else resolved.index + 1
            return _OpPlan(
                index=k, name=name, mutation=("insert_line", insert_at, content),
                success_line=f"Inserted line {insert_at + 1}.", primary_line=insert_at + 1,
                kind="insert", pos=insert_at, tie_rank=1, span=(insert_at, insert_at),
                anchor_idx=resolved.index, boundary=insert_at,
            )
        if name == "delete":
            anchor = self._id(args, required=True)
            if looks_like_range_anchor(anchor):
                start, end = resolve_range(parse_range(anchor), doc, index)
                success = DeleteSummary.range(start.line_no, end.line_no).success_message()
                return _OpPlan(
                    index=k, name=name, mutation=("delete_range", start.index, end.index),
                    success_line=success, primary_line=start.line_no, kind="consuming",
                    pos=start.index, tie_rank=0, span=(start.index, end.index),
                    anchor_idx=start.index, boundary=start.index,
                )
            resolved = resolve(parse_anchor(anchor), doc, index)
            success = DeleteSummary.single(resolved.line_no).success_message()
            return _OpPlan(
                index=k, name=name, mutation=("delete_line", resolved.index),
                success_line=success, primary_line=resolved.line_no, kind="consuming",
                pos=resolved.index, tie_rank=0, span=(resolved.index, resolved.index),
                anchor_idx=resolved.index, boundary=resolved.index,
            )
        raise HashlineError(f"unknown edit op {name!r}")

    @staticmethod
    def _apply_op(doc: Document, plan: _OpPlan) -> None:
        """Run one planned mutation on the in-memory doc (indices pre-resolved)."""
        m = plan.mutation
        if m[0] == "replace_line":
            replace_line(doc, m[1], m[2])
        elif m[0] == "replace_range":
            replace_range(doc, m[1], m[2], m[3])
        elif m[0] == "insert_line":
            insert_line(doc, m[1], m[2])
        elif m[0] == "delete_line":
            delete_line(doc, m[1])
        elif m[0] == "delete_range":
            delete_range(doc, m[1], m[2])

    @staticmethod
    def _conflict(plan: _OpPlan, kept: list[_OpPlan]) -> _OpPlan | None:
        """The already-kept op ``plan`` collides with, or None.

        Consuming vs consuming: their inclusive spans intersect. Insert vs
        consuming: the insert's anchor line is inside the consumed span (its
        anchor is being replaced/deleted). Insert vs insert: same landing gap.
        """
        for other in kept:
            if plan.kind == "consuming" and other.kind == "consuming":
                if max(plan.span[0], other.span[0]) <= min(plan.span[1], other.span[1]):
                    return other
            elif plan.kind == "insert" and other.kind == "consuming":
                if other.span[0] <= plan.anchor_idx <= other.span[1]:
                    return other
            elif plan.kind == "consuming" and other.kind == "insert":
                if plan.span[0] <= other.anchor_idx <= plan.span[1]:
                    return other
            elif plan.kind == "insert" and other.kind == "insert":
                if plan.boundary == other.boundary:
                    return other
        return None

    @staticmethod
    def _conflict_message(other: _OpPlan) -> str:
        lo, hi = other.span
        where = f"line {lo + 1}" if lo == hi else f"lines {lo + 1}-{hi + 1}"
        return (
            f"overlaps another edit in this turn targeting {where}, which was "
            "applied first. Re-target this against the refreshed ids in the "
            "grouped result, or combine them into one range edit."
        )

    def begin_transaction(self, args: dict[str, Any]) -> str:
        """Open a file: return its whole-file line ids.

        Wire name ``read_file`` (qwen's trained name; previously
        ``beginTransaction``, before that ``read``). Takes an absolute
        ``file_path`` and returns the full set of line ids, or an explicit
        ``offset``/``limit`` window of them.
        """
        file = self._file_path(args)
        key = self._key(file)
        offset = self._window_arg(args.get("offset"))
        limit = self._window_arg(args.get("limit"))
        if offset or limit:
            return self._paged_read(file, key, offset, limit or None)
        output = self._cap(run_read(ReadCmd(file=Path(file), anchor=[], context=0)))
        if not output.strip():
            return output
        # Read the full file so warnings use absolute line numbers and the
        # fingerprint covers the whole file for has_current_anchors.
        text = self._read_text(file)
        warnings, attention = self._syntax_section(file, text)
        self._read_records[key] = ReadRecord(
            fingerprint=self._content_fingerprint(text),
            byte_len=len(text.encode("utf-8")),
        )
        headered = key in self._read_headered
        self._read_headered.add(key)
        if headered:
            body = join_section(output, warnings)
        else:
            header = f"{file} (ids: each line is '<line><hash>|<content>'):"
            body = join_section(f"{header}\n{output}", warnings)
        return mark_attention(body) if attention else body

    @staticmethod
    def _window_arg(value: Any) -> int:
        """Lenient positive-int coercion for offset/limit: ints, floats, and
        numeric strings (XML parameters can reach us as strings); anything
        else — or a non-positive value — means "not set" (0). The schema says
        offset "requires 'limit'", but the runtime is deliberately lenient,
        matching run_inspect_file: an offset alone reads to the end (byte cap
        permitting) instead of burning a round-trip on an error."""
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
        return 0

    def _paged_read(self, file: str, key: str, offset: int, limit: int | None) -> str:
        """An explicit offset/limit window of line ids (0-based offset; the
        rendered ids keep their true 1-based line numbers).

        Satisfies the mutation gate — the ids returned are genuine — but never
        records the F1 full-read baseline: a paged read must not later
        suppress a full read with "you still hold its current line ids", which
        would be false. It never consults the suppress gate either (an
        explicit page always serves). Diagnostics run on the FULL text, never
        the window — the language server reads the file from disk.
        """
        doc = Document.load(Path(file))
        total = len(doc.lines)
        if offset >= total and total > 0:
            return f"Error: offset {offset} is past the end of the file ({total} lines)."
        end = min(offset + limit, total) if limit is not None else total
        body = self._cap(render_lines(doc, range(offset, end)))
        if not body.strip():
            return body
        text = self._read_text(file)
        warnings, attention = self._syntax_section(file, text)
        headered = key in self._read_headered
        self._read_headered.add(key)
        parts: list[str] = []
        if not headered:
            parts.append(f"{file} (ids: each line is '<line><hash>|<content>'):")
        if offset > 0 or end < total:
            # 1-based inclusive line numbers, matching read-file.ts.
            parts.append(f"Showing lines {offset + 1}-{end} of {total} total lines.")
        parts.append(body)
        out = join_section("\n".join(parts), warnings)
        return mark_attention(out) if attention else out

    def has_current_anchors(self, file: str) -> bool:
        """Whether the model already holds this file's CURRENT line anchors.

        True only when the file was read (anchors shown) AND its content is
        unchanged since — so a file the model read and that was then modified (by a
        bash write, an external editor, or another process) returns False and earns
        a fresh post-write auto-read. Used to suppress redundant auto-reads.
        """
        record = self._read_records.get(self._key(file))
        if record is None:
            return False
        return record.fingerprint == self._content_fingerprint(self._read_text(file))

    @staticmethod
    def _content_fingerprint(text: str) -> str:
        return xxhash.xxh3_64_hexdigest(text.encode("utf-8"))

    def _syntax_section(self, file: str, text: str) -> tuple[str, bool]:
        """Trailing validation block and an attention flag.

        Runs the LSP-first validation router (:func:`validate_file`; tree-sitter
        fallback). When the file has errors, returns a block that lists each NEW
        one with a ready ``replace id: <line><hash>|<content>`` so the model can fix
        that exact row, plus ``True`` (the result should be flagged is_error).
        Rows this agent instance was already shown (``diag_memory``) are
        summarised instead of repeated — the headline count always reflects the
        file's CURRENT state, so a still-broken file keeps its flag even when
        every row is old news, and a suppressed row whose line content later
        changes re-shows with a fresh id (the memory keys on content, so a
        stale ``replace id`` is never left as the model's only anchor). When clean
        or unknown, returns the OK/"" confirmation and ``False`` — built from
        the same :class:`Validation` (a second check would repeat the LSP
        round-trip). LSP warnings ride along informationally and never flip the
        flag. Never raises.
        """
        try:
            if not text:
                return "", False
            v = validate_file(file, text)
            sifted = (
                self.diag_memory.sift(file, v, text)
                if self.diag_memory is not None
                else None
            )
            if not v.errors:
                return self._clean_report(v, sifted), False
            if sifted is not None and sifted.all_prior:
                return (
                    f"Syntax check ({v.label}) — {len(v.errors)} issue(s), "
                    f"{ALL_UNCHANGED}; fix them with replace, then they are re-checked.",
                    True,
                )
            if sifted is None:
                new_errors, new_warnings = list(v.errors), list(v.warnings)
                prior_errors = prior_warnings = 0
            else:
                new_errors, new_warnings = sifted.errors, sifted.warnings
                prior_errors, prior_warnings = sifted.prior_errors, sifted.prior_warnings
            doc = Document.load(Path(file))
            n = len(doc.lines)
            shown = new_errors[:MAX_SHOWN_SYNTAX_ANCHORS]
            lines = [
                f"Syntax check ({v.label}) — {len(v.errors)} issue(s); fix each line "
                "below with replace, then it is re-checked:"
            ]
            for line_no, _col, message in shown:
                lines.append(f"- {message}")
                if 1 <= line_no <= n:
                    lines.append(f"  replace id: {line_view(line_no, doc.lines[line_no - 1])}")
            extra = len(new_errors) - len(shown)
            if extra > 0:
                lines.append(f"- … and {extra} more")
            if prior_errors:
                lines.append(unchanged_note(prior_errors))
            warn_shown = new_warnings[:MAX_SHOWN_SYNTAX_ANCHORS]
            if warn_shown or prior_warnings:
                lines.append(f"Warnings (not blocking) — {len(v.warnings)}:")
                lines.extend(f"- {message}" for _line, _col, message in warn_shown)
                warn_extra = len(new_warnings) - len(warn_shown)
                if warn_extra > 0:
                    lines.append(f"- … and {warn_extra} more")
                if prior_warnings:
                    lines.append(unchanged_note(prior_warnings, "warning"))
            if sifted is not None:
                sifted.mark_rendered(shown, warn_shown)
            return "\n".join(lines), True
        except Exception:  # noqa: BLE001 - syntax warnings are best-effort
            return "", False

    def _clean_report(self, v: Validation, sifted) -> str:
        """The clean-file confirmation, with riding warnings deduped per agent."""
        if clean_validation_report is None:
            return v.report()
        return clean_validation_report(v, sifted, MAX_SHOWN_SYNTAX_ANCHORS)

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

    def _require_transaction(self, file: str, action: str) -> None:
        """Deny a mutation on a file never opened with read_file this session.

        The model only holds line ids for files it has opened; without a prior
        read_file any id it sends is a guess.

        Exception: an existing but EMPTY (0-byte) file has no line ids to hold,
        so requiring a prior read is pointless — allow (and record) a mutation
        that populates it. The editor has no bash, so this is its only way to
        fill an empty stub. A NONEXISTENT file still falls through to the guard:
        create it with bash first.
        """
        key = self._key(file)
        if key in self._read_headered:
            return
        if os.path.exists(file) and self._read_text(file) == "":
            self._read_headered.add(key)
            return
        raise HashlineError(
            f"{action} denied: {file} was not opened with read_file in this "
            "session, so you do not hold its line ids. Call the read_file "
            f"tool on {file} first to read its current '<line><hash>' ids, then "
            "retry using an id copied from that output."
        )

    def _populate_empty(self, file: str, content: str, before: str) -> str:
        """Write ``content`` as the initial lines of an existing EMPTY file.

        A 0-line file has no anchors to address, so replace/insert on it are
        treated as "set the content": whatever id the model sent is irrelevant.
        The file must already exist — creating brand-new files still goes
        through bash. Presented like any other mutation (diff + fresh ids +
        syntax check) so the caller can keep editing with the returned ids.
        """
        doc = Document.load(Path(file))
        insert_line(doc, 0, content)
        # A 0-byte file loads with trailing_newline=False; a populated source
        # file should end with a newline like any other.
        doc.trailing_newline = True
        atomic_write_document(Path(file), doc)
        return self._present_mutation(file, "Added the file's initial content.\n", before)

    def edit(self, args: dict[str, Any]) -> str:
        file = self._file(args)
        self._require_transaction(file, "replace")
        before = self._read_text(file)
        if before == "" and os.path.exists(file):
            # Nothing to replace in an empty file: treat it as "set the content".
            return self._populate_empty(file, self._content(args), before)
        anchor = self._id(args, required=not self._has_start_query(args))
        content = self._content(args)
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
        self._require_transaction(file, "insert")
        before = self._read_text(file)
        if before == "" and os.path.exists(file):
            # Empty file: nothing to anchor to, so any id is accepted and the
            # content becomes the file's first lines.
            return self._populate_empty(file, self._content(args), before)
        anchor = self._id(args, required=not self._has_start_query(args))
        if anchor and looks_like_range_anchor(anchor):
            left, right = anchor.split("..", 1)
            raise HashlineError(
                "insert requires one line id, not a range. "
                f"Use id {left!r} with position='after', or id {right!r} with position='before'."
            )
        content = self._content(args)
        position = str(args.get("position") or "after").lower()
        if position not in {"before", "after"}:
            raise HashlineError("'position' must be 'before' or 'after'")
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
        self._require_transaction(file, "delete")
        anchor = self._id(args, required=not self._has_start_query(args))
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
            # full-read baseline is stale: drop it so has_current_anchors returns
            # False and the post-write auto-read surfaces fresh ids.
            self._read_records.pop(self._key(file), None)
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
                    f"Current {file} (ids, lines {lo}-{hi}: "
                    "each line is '<line><hash>|<content>'):"
                )
                parts.extend(snippet)
        # Report the post-mutation validation state as a canonical trailing
        # section (tools/diagnostics grammar): the mutation body above and the
        # diagnostics below stay structurally separate, so the TUI and any
        # other consumer can carve them apart without guessing.
        warnings, attention = self._syntax_section(file, after)
        body = join_section("\n".join(parts), warnings)
        return mark_attention(body) if attention else body

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
        file = args.get("file") or self._last_file
        if not isinstance(file, str) or not file:
            raise HashlineError("'file' is required")
        self._last_file = file
        return file

    def _file_path(self, args: dict[str, Any]) -> str:
        """The read tool's target: an absolute ``file_path`` (qwen's trained
        schema; ``file`` accepted as a fallback). Internal callers pass
        ``_force``, which also waives the absolute-path requirement."""
        file = args.get("file_path") or args.get("file") or self._last_file
        if not isinstance(file, str) or not file:
            raise HashlineError("'file_path' is required")
        if not os.path.isabs(file) and not bool(args.get("_force")):
            raise HashlineError(f"File path must be absolute: {file}")
        self._last_file = file
        return file

    @staticmethod
    def _key(path: str) -> str:
        """Canonical per-file key for the session dicts (read gate, F1
        fingerprints): absolute-ized but NOT symlink-resolved — resolving would
        diverge from the paths LSP, grep, and the TUI display. Lets read_file's
        absolute file_path and a mutation's relative ``file`` land on the same
        record."""
        return os.path.abspath(path)

    def _id(self, args: dict[str, Any], required: bool = True) -> str:
        value = args.get("id")
        if not isinstance(value, str) or not value:
            if not required:
                return ""
            raise HashlineError(
                "'id' is required; file mutations cannot create files or target empty files. "
                "Use bash to create the file, then read_file for line ids before editing."
            )
        return value

    def _content(self, args: dict[str, Any]) -> str:
        content = args.get("content")
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
        return text[:MAX_READ_CHARS].rstrip("\n") + "\n... (truncated: file too large to show all ids; page through the rest with read_file offset/limit)"
