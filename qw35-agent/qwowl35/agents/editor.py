"""The editor: a standalone sub-agent that applies delegated file changes.

Spawned by the freestyle `edit` tool call. All `edit` calls in a turn continue
ONE persistent editor conversation — like the planner and the executor — each
appending a slim task message, so the scratch-session checkpoint stack
prefills only the new directive. Runs on the scratch GPU session with its own
strict prompt (running its divergent prompt on the main session would clobber
the pipeline's KV rows and checkpoints). Toolset: hashline `replace`/`insert`/`delete` plus read-only
lookups — `read_file` (re-open or page a file when ids go stale or lines fall
outside the slice), `lsp`, and the single-file `grep_search` (the editor's
basic id-row variant, not the explorer's tree-walking grep). The target file
is opened for it (line ids supplied up front); there is no bash.
"""

from __future__ import annotations

import re

from .base import SESSION_SCRATCH, AgentSpec

SPEC = AgentSpec(
    name="editor",
    session=SESSION_SCRATCH,
    allowed_tools=frozenset({"read_file", "replace", "insert", "delete", "lsp", "grep_search"}),
    mascot="edit",
)

# How many context lines around a requested range the editor gets to see.
RANGE_MARGIN = 25
# Files at or under this many lines are shown whole regardless of ranges.
SMALL_FILE_LINES = 400

EDITOR_SYSTEM_PROMPT = """\
You are the qwowl35 editor. You receive edit tasks one at a time: each names
its File:, shows that file's current content (with line ids), and gives
instructions for a change. Make every edit the instructions require — as many
lines or places as it takes to implement them correctly and leave the file
working — but nothing beyond them: do not make unrelated changes, and never
touch another file. Use the editing tools, then reply with a one-paragraph
summary of what you changed (no tool call) to finish. A later task may target a
different file; always edit the file named by the CURRENT task, and never reuse
line ids from an earlier task's content.

Line ids look like 12af|content — line number 12, hash af. Address lines by id.
When a change touches several lines, issue those edits together in one turn:
they are applied as a group and the diff and syntax check are shown once.
- replace: replace one line (id: 12af) or an inclusive range (id: 12af..18bc)
  with new content. This is the default for CHANGING code — editing in place
  keeps surrounding ids stable; prefer it over delete+insert.
- insert: add lines before/after a single id (position defaults to after).
- delete: remove a line or range by id. Use it only to take code OUT; to change
  a line, replace it rather than deleting and re-inserting.
- lsp: look up a symbol before touching it — operation goToDefinition |
  findReferences | hover | documentSymbol, with filePath and 1-based
  line/character. Read-only; it does not count as a change. Results in your
  file show line ids (12af|content) you can pass directly to replace/delete.
- grep_search: regex-search ONE file (pattern + path, case-insensitive) when
  the lines you need are outside the slice you were shown. Every match comes
  back as an id row (12af|content) you can pass directly to replace/delete.
  Read-only; it does not count as a change.
- read_file: read the file (absolute file_path) when your ids are stale or
  you need lines outside the slice you were shown; page big files with offset
  (0-based) and limit. Read-only; it does not count as a change.
Content is literal: match the file's exact indentation. If a result reports a
stale anchor, re-read the refreshed ids in that result and retry. If a result
ends with a Syntax check block, the file no longer parses — fix the reported
line(s) before finishing.
You may receive a Background section (plan excerpt, the delegating agent's
recent activity and reasoning). It is context only — use it to understand the
change, but implement only what the Instructions ask; do not take on unrelated
work or edit other files.

Do not emit JSON inside <tool_call>. Use nested XML; do not put arguments as XML attributes.
Each call has one <function=tool_name> element and child <parameter=name>value</parameter> elements."""


def parse_line_ranges(
    line_ranges: str,
    total_lines: int,
    margin: int = RANGE_MARGIN,
    whole_file_under: int = SMALL_FILE_LINES,
) -> list[tuple[int, int]]:
    """Parse `\"12-18, 40\"` (1-based, inclusive) into sorted merged spans.

    `\"all\"`, unparseable input, or a small file select the whole file —
    showing too much beats slicing away the lines the change needs. The Spawn
    card passes ``margin=0, whole_file_under=0`` to visualize only the ranges
    the call actually named.
    """
    text = (line_ranges or "").strip().lower()
    if not text or text == "all" or total_lines <= whole_file_under:
        return [(1, total_lines)] if total_lines else []
    spans: list[tuple[int, int]] = []
    for part in re.split(r"[,;]", text):
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if start > end:
            start, end = end, start
        start = max(1, start - margin)
        end = min(total_lines, end + margin)
        if start <= total_lines:
            spans.append((start, end))
    if not spans:
        return [(1, total_lines)] if total_lines else []
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def slice_annotated_body(annotated: str, spans: list[tuple[int, int]]) -> str:
    """Cut an id-annotated file body down to the requested spans.

    The annotated body is one `<line><hash>|content` row per file line, in
    order; omitted stretches are marked so the editor knows lines exist there.
    """
    lines = annotated.splitlines()
    if not spans or (len(spans) == 1 and spans[0] == (1, len(lines))):
        return annotated
    pieces: list[str] = []
    previous_end = 0
    for start, end in spans:
        if start > previous_end + 1:
            pieces.append(f"... ({start - previous_end - 1} lines not shown) ...")
        pieces.extend(lines[start - 1 : end])
        previous_end = end
    if previous_end < len(lines):
        pieces.append(f"... ({len(lines) - previous_end} lines not shown) ...")
    return "\n".join(pieces)


def build_editor_user_message(
    filename: str, instructions: str, annotated_slice: str, background: str = ""
) -> dict:
    """The editor's single opening user message.

    ``background`` (see :mod:`agents.spawn_context`) precedes the task so the
    editor reads context first and lands on its actual instructions last; the
    ``---`` rule keeps the two visually separate.
    """
    task = (
        f"File: {filename}\n"
        f"Instructions: {instructions.strip()}\n\n"
        f"Current content (line ids):\n{annotated_slice}"
    )
    if background:
        task = f"{background}\n\n---\n\n{task}"
    return {"role": "user", "content": task}


def build_editor_continuation_message(
    filename: str, instructions: str, annotated_slice: str, background: str = ""
) -> dict:
    """A later edit on THE editor conversation — same file or not: slim, the
    system prompt and earlier tasks already live in the inherited context, so
    only the new task (with the target file's fresh content) lands here.
    Mirrors :func:`freestyle.build_continuation_message`.
    """
    task = (
        f"Previous edit complete. New edit task:\n"
        f"File: {filename}\n"
        f"Instructions: {instructions.strip()}\n\n"
        f"Current content (line ids):\n{annotated_slice}"
    )
    if background:
        task = f"{background}\n\n---\n\n{task}"
    return {"role": "user", "content": task}
