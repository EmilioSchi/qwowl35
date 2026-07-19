"""Explorer file-listing cards: ls / glob / grep result parsers and their
guide-lined tree renderers."""

from __future__ import annotations

import os
import re

from rich.text import Text

import theme
from config import TOOL_PREVIEW_LINES


# --- explorer tool cards (list_directory / glob / grep_search / inspect_file) -
# Each explorer tool renders as a self-titled card (its own header row carries
# the subject + counts), so none of them get the badge `_tool_title` row — it
# would just repeat the header. Empty results (empty dir, no glob/grep match)
# parse to zero-entry tuples and render as the same card shape; every parser
# returns None on any other mismatch (errors, format drift) → plain result
# box fallback.

# `Listed {total} item(s) in {path}:` — the run_ls success header.
_LS_HEADER = re.compile(r"^Listed (\d+) item\(s\) in (.+):$")
# run_ls's empty-directory message (a single line, no `---` listing).
_LS_EMPTY = re.compile(r"^Directory (.+) is empty\.$")
# `({N} more entries not shown)` — run_ls's own >MAX_ENTRIES truncation tail.
_LS_TAIL = re.compile(r"^\((\d+) more entries not shown\)$")
# run_glob's success header and >MAX_FILES truncation tail.
_GLOB_HEADER = re.compile(
    r"^Found (\d+) file\(s\) matching \"(.*)\" within (.+), "
    r"sorted by modification time \(newest first\):$"
)
_GLOB_TAIL = re.compile(r"^\(Results truncated: (\d+) more files? matched\)$")
# run_glob's no-match message (single line, no trailing period).
_GLOB_EMPTY = re.compile(r'^No files found matching pattern "(.*)" within (.+)$')
# run_grep's success header, `L{n}: {line}` match rows, and the per-file
# elision rows compress_grep may add.
_GREP_HEADER = re.compile(r"^Found (\d+) match(?:es)? for pattern \"(.*?)\" (.+):$")
_GREP_LINE = re.compile(r"^L(\d+): (.*)$")
# run_grep's no-match message; the scope group absorbs any `(filter: …)` tail.
_GREP_EMPTY = re.compile(r'^No matches found for pattern "(.*?)" (.+)\.$')


# Nerd Font glyphs for the ls tree — the one deliberate exception to the
# module's "no fancy glyphs" rule: the xplr-style listing is opt-in eye candy
# and degrades to tofu boxes (not broken layout) without a patched font.
_LS_DIR_ICON = ""   # folder
_LS_FILE_ICON = ""  # generic file
# Exact file names first, then lowercased extensions. Values are
# (glyph, palette token); the token is resolved at render time so the tree
# follows live theme switches like every other theme.* consumer.
_LS_NAME_ICONS: dict[str, tuple[str, str]] = {
    "license": ("", "ERROR_SOFT"),
    "copying": ("", "ERROR_SOFT"),
    "makefile": ("", "SUCCESS_SOFT"),
    "dockerfile": ("", "SUCCESS_SOFT"),
    ".gitignore": ("", "FG_DIM"),
    ".gitattributes": ("", "FG_DIM"),
    ".gitmodules": ("", "FG_DIM"),
}
_LS_EXT_ICONS: dict[str, tuple[str, str]] = {
    # docs
    ".md": ("", "WARNING"),
    ".rst": ("", "WARNING"),
    ".txt": ("", "FG_BRIGHT"),
    # code
    ".py": ("", "ACCENT"),
    ".rs": ("", "ACCENT"),
    ".js": ("", "ACCENT"),
    ".ts": ("", "ACCENT"),
    ".c": ("", "ACCENT"),
    ".h": ("", "ACCENT"),
    ".m": ("", "ACCENT"),
    ".metal": ("", "ACCENT"),
    ".sh": ("", "ACCENT"),
    # config / data
    ".toml": ("", "SUCCESS_SOFT"),
    ".ini": ("", "SUCCESS_SOFT"),
    ".cfg": ("", "SUCCESS_SOFT"),
    ".json": ("", "SUCCESS_SOFT"),
    ".yaml": ("", "SUCCESS_SOFT"),
    ".yml": ("", "SUCCESS_SOFT"),
    ".lock": ("", "ERROR_SOFT"),
    # images
    ".png": ("", "FG_MUTED"),
    ".jpg": ("", "FG_MUTED"),
    ".jpeg": ("", "FG_MUTED"),
    ".gif": ("", "FG_MUTED"),
    ".svg": ("", "FG_MUTED"),
    ".ico": ("", "FG_MUTED"),
}


def _parse_ls_result(result: str) -> tuple[str, int, list[tuple[str, bool]], int] | None:
    """Parse a `list_directory` success into (path, total, entries, hidden)
    where entries are (name, is_dir) rows and hidden is run_ls's own
    truncation count. An empty directory parses to zero entries; anything
    else (errors, format drift) returns None so the caller falls back to
    the plain result box."""
    empty = _LS_EMPTY.match(result.strip())
    if empty is not None:
        return empty.group(1), 0, [], 0
    lines = result.splitlines()
    if len(lines) < 3 or lines[1] != "---":
        return None
    header = _LS_HEADER.match(lines[0])
    if header is None:
        return None
    total, path = int(header.group(1)), header.group(2)
    body = lines[2:]
    hidden = 0
    if len(body) >= 2 and body[-2] == "":
        tail = _LS_TAIL.match(body[-1])
        if tail is None:
            return None
        hidden = int(tail.group(1))
        body = body[:-2]
    entries: list[tuple[str, bool]] = []
    for line in body:
        if not line:
            return None
        if line.startswith("[DIR] "):
            entries.append((line[len("[DIR] "):], True))
        else:
            entries.append((line, False))
    if not entries:
        return None
    return path, total, entries, hidden


def _ls_icon(name: str, is_dir: bool) -> tuple[str, str]:
    """The (glyph, color) pair for one ls entry; colors read live theme.*."""
    if is_dir:
        return _LS_DIR_ICON, theme.ACCENT
    glyph, token = _LS_NAME_ICONS.get(name.lower()) or _LS_EXT_ICONS.get(
        os.path.splitext(name)[1].lower(), (_LS_FILE_ICON, "FG_BRIGHT")
    )
    return glyph, getattr(theme, token)


def _ls_tree_text(
    path: str,
    total: int,
    entries: list[tuple[str, bool]],
    hidden_server: int,
    expanded: bool,
) -> Text:
    """The xplr-style directory card: `path (total)` header, then one
    guide-lined row per entry — dirs first (run_ls already sorts them),
    folder icons + accent names with a trailing `/`, per-filetype file icons.
    An empty directory shows the ever-present `.` and `..` rows instead, so
    the card keeps the populated-listing shape."""
    tree = Text()
    tree.append(path, style=f"bold {theme.FG_BRIGHT}")
    tree.append(f" ({total})", style=theme.FG_GHOST)
    if not entries:
        entries = [(".", True), ("..", True)]
    shown = entries
    if not expanded and len(entries) > TOOL_PREVIEW_LINES:
        shown = entries[:TOOL_PREVIEW_LINES]
    trimmed = len(entries) - len(shown)
    for index, (name, is_dir) in enumerate(shown):
        last = index == len(shown) - 1 and not trimmed and not hidden_server
        tree.append("\n")
        tree.append(" └─ " if last else " ├─ ", style=theme.FG_GHOST)
        glyph, color = _ls_icon(name, is_dir)
        tree.append(f"{glyph} ", style=color)
        if is_dir:
            tree.append(f"{name}/", style=f"bold {theme.ACCENT}")
        else:
            tree.append(name, style=color)
    if trimmed:
        tree.append(f"\n... {trimmed} more lines (Ctrl+o to expand)", style="dim")
    if hidden_server:
        tree.append(f"\n({hidden_server} more entries not shown)", style="dim")
    return tree


def _parse_glob_result(result: str) -> tuple[str, str, int, list[str], int] | None:
    """Parse a `glob` success into (pattern, base, total, paths, hidden).
    A no-match result parses to zero paths."""
    empty = _GLOB_EMPTY.match(result.strip())
    if empty is not None:
        return empty.group(1), empty.group(2), 0, [], 0
    lines = result.splitlines()
    if len(lines) < 3 or lines[1] != "---":
        return None
    header = _GLOB_HEADER.match(lines[0])
    if header is None:
        return None
    total, pattern, base = int(header.group(1)), header.group(2), header.group(3)
    body = lines[2:]
    hidden = 0
    if len(body) >= 2 and body[-2] == "":
        tail = _GLOB_TAIL.match(body[-1])
        if tail is None:
            return None
        hidden = int(tail.group(1))
        body = body[:-2]
    if not body or any(not line for line in body):
        return None
    return pattern, base, total, body, hidden


def _glob_tree_text(
    pattern: str,
    base: str,
    total: int,
    paths: list[str],
    hidden_server: int,
    expanded: bool,
) -> Text:
    """The glob card: `pattern (total)  within base` header, then one
    guide-lined row per matched file — the base prefix is shown once in the
    header and stripped from every row; the remaining directory part stays
    dim, per-filetype icon + colored name."""
    tree = Text()
    tree.append(pattern, style=f"bold {theme.FG_BRIGHT}")
    tree.append(f" ({total})", style=theme.FG_GHOST)
    tree.append(f"  within {base}", style=theme.FG_FAINT)
    if not paths:
        tree.append("\n")
        tree.append(" └─ ", style=theme.FG_GHOST)
        tree.append("no matches", style="dim")
        return tree
    shown = paths
    if not expanded and len(paths) > TOOL_PREVIEW_LINES:
        shown = paths[:TOOL_PREVIEW_LINES]
    trimmed = len(paths) - len(shown)
    prefix = base.rstrip("/") + "/"
    for index, path in enumerate(shown):
        last = index == len(shown) - 1 and not trimmed and not hidden_server
        tree.append("\n")
        tree.append(" └─ " if last else " ├─ ", style=theme.FG_GHOST)
        rel = path[len(prefix):] if path.startswith(prefix) else path
        head, _, name = rel.rpartition("/")
        glyph, color = _ls_icon(name, False)
        tree.append(f"{glyph} ", style=color)
        if head:
            tree.append(f"{head}/", style=theme.FG_GHOST)
        tree.append(name, style=color)
    if trimmed:
        tree.append(f"\n... {trimmed} more lines (Ctrl+o to expand)", style="dim")
    if hidden_server:
        tree.append(f"\n({hidden_server} more files matched)", style="dim")
    return tree


def _parse_grep_result(
    result: str,
) -> tuple[str, str, int, list[tuple[str, list[tuple[str, str]]]], list[str]] | None:
    """Parse a `grep_search` success into (pattern, scope, total, groups, notes)
    where groups are (file, [(line_number, text), ...]) — an elision row from
    compress_grep keeps an empty line_number — and notes are the trailing
    truncation / compression sentences. A no-match result parses to zero
    groups."""
    empty = _GREP_EMPTY.match(result.strip())
    if empty is not None:
        return empty.group(1), empty.group(2), 0, [], []
    lines = result.splitlines()
    if len(lines) < 3 or lines[1] != "---":
        return None
    header = _GREP_HEADER.match(lines[0])
    if header is None:
        return None
    total, pattern, scope = int(header.group(1)), header.group(2), header.group(3)
    groups: list[tuple[str, list[tuple[str, str]]]] = []
    notes: list[str] = []
    for index in range(2, len(lines)):
        line = lines[index]
        if line == "---":
            continue
        if not line:
            # A blank line ends the listing; what follows is footer notes
            # (truncation sentence, compression marker).
            notes = [note for note in lines[index + 1:] if note]
            break
        if line.startswith("File: "):
            groups.append((line[len("File: "):], []))
            continue
        match = _GREP_LINE.match(line)
        if match is not None and groups:
            groups[-1][1].append((match.group(1), match.group(2)))
            continue
        if line.lstrip().startswith("…") and groups:
            groups[-1][1].append(("", line.strip()))
            continue
        return None
    if not groups:
        return None
    return pattern, scope, total, groups, notes


def _grep_tree_text(
    pattern: str,
    scope: str,
    total: int,
    groups: list[tuple[str, list[tuple[str, str]]]],
    notes: list[str],
    expanded: bool,
) -> Text:
    """The grep card: `pattern (N matches)` header, then a guide-lined tree of
    files with their matching lines nested under a continuation guide."""
    tree = Text()
    tree.append(pattern, style=f"bold {theme.FG_BRIGHT}")
    match_term = "match" if total == 1 else "matches"
    tree.append(f" ({total} {match_term})", style=theme.FG_GHOST)
    tree.append(f"  {scope}", style=theme.FG_FAINT)
    if not groups:
        tree.append("\n")
        tree.append(" └─ ", style=theme.FG_GHOST)
        tree.append("no matches", style="dim")
        return tree
    budget = None if expanded else TOOL_PREVIEW_LINES
    total_rows = sum(1 + len(hits) for _, hits in groups)
    rows_shown = 0
    hidden = 0
    for group_index, (file, hits) in enumerate(groups):
        if budget is not None and rows_shown >= budget:
            hidden = total_rows - rows_shown
            break
        last_group = group_index == len(groups) - 1
        tree.append("\n")
        tree.append(" └─ " if last_group else " ├─ ", style=theme.FG_GHOST)
        glyph, color = _ls_icon(file, False)
        tree.append(f"{glyph} ", style=color)
        tree.append(file, style=f"bold {theme.ACCENT}")
        rows_shown += 1
        stem = "    " if last_group else " │  "
        for line_number, content in hits:
            if budget is not None and rows_shown >= budget:
                hidden = total_rows - rows_shown
                break
            tree.append("\n")
            tree.append(stem, style=theme.FG_GHOST)
            if line_number:
                tree.append(f"L{line_number} ", style=theme.ACCENT)
                tree.append(content, style=theme.FG_BRIGHT)
            else:
                tree.append(content, style="dim")
            rows_shown += 1
        if hidden:
            break
    if hidden:
        tree.append(f"\n... {hidden} more lines (Ctrl+o to expand)", style="dim")
    for note in notes:
        tree.append(f"\n{note}", style="dim")
    return tree
