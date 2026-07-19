"""Tests for code-view renderers: anchored file views, diffs, inspect
windows, and streaming edit content."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from widgets.chat import ChatView, ToolBlock  # noqa: E402

from chat_test_helpers import _ansi, _fg_triplet, _plain, assert_true  # noqa: E402


def _bg_triplet(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"48;2;{int(h[0:2], 16)};{int(h[2:4], 16)};{int(h[4:6], 16)}"


def _joined(text: str) -> str:
    """Collapse whitespace so phrase checks survive the console's line wrapping."""
    return " ".join(text.split())


def test_read_result_keeps_anchors() -> None:
    block = ToolBlock("beginTransaction")
    block.args_buf = '{"file":"example.py"}'
    block.full_result = (
        "example.py (ids: each line is '<line><hash>|<content>'):\n"
        "1af|def f():\n"
        "230|    return 1\n"
        "3b2|abc  = 2"
    )

    text = _plain(ChatView()._render_tool_result(block))

    assert_true("Read  example.py" in text, "read badge includes path")
    assert_true("1af" in text, "anchor shown")
    assert_true("def f():" in text and "    return 1" in text, "code content shown")
    assert_true("abc  = 2" in text, "plain source with two spaces is not parsed as a hidden anchor")


def test_edit_result_shows_diff_before_refreshed_labels() -> None:
    block = ToolBlock("edit")
    block.args_buf = '{"path":"example.py","anchor":"2:30"}'
    block.full_result = (
        "Edited line 2 in example.py.\n"
        "Diff:\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return 1\n"
        "+    return 2\n"
        "Current example.py (ids, lines 1-2: each line is '<line><hash>|<content>'):\n"
        "1af|def f():\n"
        "292|    return 2"
    )

    text = _plain(ChatView()._render_tool_result(block))

    assert_true("Diff" in text, "diff label shown")
    # Marker gets its own column + a separator space so +/- content aligns with
    # context rows: "-" + " " + original "    return 1".
    assert_true("-     return 1" in text, "removed line shown")
    assert_true("+     return 2" in text, "added line shown")
    assert_true("Current example.py" in text and "292" in text, "refreshed anchors shown")


def test_attention_flagged_edit_still_shows_colored_diff() -> None:
    """An edit that succeeded but tripped the attention marker (e.g. it left a
    syntax warning) is surfaced with is_error=True — it must still get the
    colored diff view, not degrade to the plain error-text fallback."""
    block = ToolBlock("edit")
    block.args_buf = '{"filename":"example.py","line_ranges":"1-2","instructions":"x"}'
    block.is_error = True
    block.full_result = (
        "Editor result for example.py (1 edit applied):\n"
        "Done.\n\n"
        "Diff:\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return 1\n"
        "+    return 2\n"
    )

    text = _plain(ChatView()._render_tool_result(block))

    assert_true("Diff" in text, "diff label shown")
    assert_true("-     return 1" in text, "removed line shown")
    assert_true("+     return 2" in text, "added line shown")
    assert_true("Result" not in text, "does not fall back to the plain error renderer")
    # A block that never streamed (restored session shape) has no captured
    # slice — the card renders exactly as before, no numbered gutter rows.
    assert_true(block.spawn_snippet is None, "no snippet without a live capture")


def test_hard_edit_error_without_diff_falls_back_to_plain_result() -> None:
    """A genuine failure (no diff was ever produced) must keep the plain,
    error-colored fallback instead of trying to render a structured view."""
    block = ToolBlock("edit")
    block.args_buf = '{"file":"example.py","anchor":"2:30"}'
    block.is_error = True
    block.full_result = "Error: hash ab not found in example.py"

    text = _plain(ChatView()._render_tool_result(block))

    assert_true("Result" in text, "plain error fallback shown")
    assert_true("Error: hash ab not found in example.py" in text, "error message shown")


def test_read_shows_syntax_status() -> None:
    ok = ToolBlock("beginTransaction")
    ok.args_buf = '{"file":"ok.py"}'
    ok.full_result = (
        "ok.py (ids: each line is '<line><hash>|<content>'):\n"
        "17b|def g():\n2a7|    return 1\n\n"
        "Syntax check (python): OK — no syntax errors."
    )
    assert_true("Syntax check (python): OK" in _plain(ChatView()._render_tool_result(ok)),
                "read of a clean file shows the OK status")

    bad = ToolBlock("beginTransaction")
    bad.args_buf = '{"file":"bad.py"}'
    bad.full_result = (
        "bad.py (ids: each line is '<line><hash>|<content>'):\n"
        "1cc|def oops(\n\n"
        "Syntax check (python) — 1 issue(s):\n- line 1, col 1: unexpected 'def oops('"
    )
    assert_true("issue(s)" in _plain(ChatView()._render_tool_result(bad)),
                "read of a broken file shows the issue list")


def test_attention_flagged_read_still_shows_anchored_view() -> None:
    """The editor sub-agent's read: the file carries a syntax warning →
    is_error=True, and the anchors header is suppressed (a repeat read after
    the orchestrator pre-opened the file). With no diff, this used to skip the
    file-view branch and drop to the plain 'Result' box. It must render as the
    anchored code card, with its path recovered from the `file_path` arg for
    both the title and the synthesized header."""
    import re

    block = ToolBlock("read_file")
    block.args_buf = '{"file_path":"bad.py"}'
    block.is_error = True
    block.full_result = (
        "1cc|def oops(\n\n"
        "Syntax check (python) — 1 issue(s):\n- line 1, col 1: unexpected 'def oops('"
    )

    text = _plain(ChatView()._render_tool_result(block))

    assert_true("Read  bad.py" in text, f"title recovers the file_path arg: {text}")
    assert_true("bad.py (ids:" in text and "file (ids:" not in text,
                f"synthesized header names the file, not 'file': {text}")
    assert_true(re.search(r"1cc  def oops\(", text), f"anchored code row shown: {text}")
    assert_true("issue(s)" in text, "syntax-check note still surfaced")
    assert_true("Result" not in text, "does not fall back to the plain error box")


def test_anchored_code_degrades_when_highlighting_raises() -> None:
    """When Rich/Pygments highlighting throws, `_render_anchored_code` must
    degrade to plain text, not crash. The exception fallback used to unpack the
    parsed 3-tuples into two names and raise ValueError itself — taking down
    both the read_file card and the editor spawn card that share this helper."""
    from unittest.mock import patch

    from widgets.chat.renderers.code import _render_anchored_code

    rows = ["1af|def f():", "230|    return 1"]
    with patch("rich.syntax.Syntax.guess_lexer", side_effect=RuntimeError("boom")):
        text = _plain(_render_anchored_code("example.py", rows))

    assert_true("def f():" in text and "    return 1" in text,
                f"content still rendered via the fallback: {text}")
    assert_true("1af" in text and "230" in text, "anchors still shown")


def test_path_from_args_reads_file_path() -> None:
    """read_file's schema uses `file_path`; the render path must recover it so
    the view gets a filename for lexer guessing and the synthesized header."""
    from widgets.chat.tool_args import _path_from_args

    assert_true(_path_from_args({"file_path": "/a/b.py"}) == "/a/b.py",
                "file_path is recovered as the render path")
    assert_true(_path_from_args({"file": "/x.py", "file_path": "/y.py"}) == "/x.py",
                "existing file/path keys still win over file_path")


def test_edit_result_without_snippet_keeps_syntax_out_of_diff() -> None:
    """Regression: with no `Current <path> (ids, …)` snippet after the diff,
    the trailing Syntax check block used to be swallowed into the diff text
    and drawn as red removal rows with diff gutter numbers."""
    import re

    import theme
    from widgets.chat.renderers.code import _split_diff_section

    body = (
        "Edited line 2 in example.py.\n"
        "Diff:\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return 1\n"
        "+    return 2\n"
        "\n"
        "Syntax check (python, lsp) — 2 issue(s); fix each line below with edit, "
        "then it is re-checked:\n"
        "- line 5, col 1: [no-member] Instance of 'A' has no 'b' member"
    )

    # Belt-and-suspenders: the raw diff scan itself must stop at the section.
    _intro, diff, _after = _split_diff_section(body)
    assert_true("Syntax check" not in diff, f"diff scan stops at the section: {diff!r}")

    block = ToolBlock("edit")
    block.args_buf = '{"file":"example.py","id":"230"}'
    block.full_result = body
    rendered = ChatView()._render_tool_result(block)
    text = _plain(rendered)
    assert_true("-     return 1" in text, "real removal row still shown")
    assert_true("Syntax check (python, lsp)" in text, "syntax block still shown")
    assert_true("- line 5, col 1:" in text, "issue bullet still shown")
    assert_true(
        re.search(r"\d+ +- line 5", text) is None,
        f"no diff gutter number before the issue bullet: {text}",
    )

    status_rows = [
        row for row in _ansi(rendered, width=200).splitlines()
        if "Syntax check" in row or "no-member" in row
    ]
    assert_true(len(status_rows) == 2, f"both section rows found in ansi: {len(status_rows)}")
    for row in status_rows:
        assert_true(
            _bg_triplet(theme.DIFF_REMOVE_BG) not in row,
            f"section row not on the diff-remove background: {row!r}",
        )
        assert_true(
            _fg_triplet(theme.ERROR_SOFT) in row,
            f"section row uses the syntax-status colour: {row!r}",
        )


def test_edit_result_with_snippet_keeps_syntax_status_outside_code_block() -> None:
    # The already-correct shape (Current snippet present) keeps every piece —
    # diff rows, refreshed anchors — and the syntax block renders through the
    # syntax-status route, never inside the code block.
    import theme

    block = ToolBlock("edit")
    block.args_buf = '{"file":"example.py","id":"230"}'
    block.full_result = (
        "Edited line 2 in example.py.\n"
        "Diff:\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return 1\n"
        "+    return 2\n"
        "Current example.py (ids, lines 1-2: each line is '<line><hash>|<content>'):\n"
        "1af|def f():\n"
        "292|    return 2\n"
        "\n"
        "Syntax check (python) — 1 issue(s); fix each line below with edit, "
        "then it is re-checked:\n"
        "- line 5, col 1: [no-member] whatever"
    )
    rendered = ChatView()._render_tool_result(block)
    text = _plain(rendered)
    assert_true("-     return 1" in text and "+     return 2" in text, "diff rows kept")
    assert_true("Current example.py" in text and "292" in text, "refreshed anchors kept")
    assert_true(
        "Syntax check (python)" in text and "- line 5, col 1:" in text, "syntax block shown"
    )
    status_rows = [
        row for row in _ansi(rendered, width=200).splitlines() if "Syntax check" in row
    ]
    assert_true(len(status_rows) == 1, f"syntax header found in ansi: {len(status_rows)}")
    assert_true(
        _bg_triplet(theme.CODE_BG) not in status_rows[0],
        f"syntax status not painted inside the code block: {status_rows[0]!r}",
    )
    assert_true(
        _fg_triplet(theme.ERROR_SOFT) in status_rows[0],
        f"syntax-status colour used: {status_rows[0]!r}",
    )


def test_all_unchanged_single_line_section_renders_as_status() -> None:
    # The dedup layer can collapse a whole section to one line (all rows were
    # already reported) or keep a summary bullet; both must ride the
    # syntax-status route for edit results and read results alike.
    import re

    import theme

    dedup = (
        "Syntax check (python, lsp) — 9 issue(s), all unchanged and already "
        "reported above; fix each line with edit, then it is re-checked."
    )

    edit = ToolBlock("edit")
    edit.args_buf = '{"file":"example.py","id":"230"}'
    edit.full_result = (
        "Edited line 2 in example.py.\n"
        "Diff:\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return 1\n"
        "+    return 2\n"
        "\n" + dedup
    )
    text = _plain(ChatView()._render_tool_result(edit))
    assert_true(
        "all unchanged and already reported above" in _joined(text),
        "edit: dedup line shown",
    )
    assert_true(
        re.search(r"\d+ +Syntax check", text) is None,
        f"edit: no diff gutter number before the dedup line: {text}",
    )

    read = ToolBlock("beginTransaction")
    read.args_buf = '{"file":"example.py"}'
    read.full_result = (
        "example.py (ids: each line is '<line><hash>|<content>'):\n"
        "1af|def f():\n"
        "230|    return 1\n"
        "\n" + dedup
    )
    rendered = ChatView()._render_tool_result(read)
    text = _plain(rendered)
    assert_true(
        "all unchanged and already reported above" in _joined(text),
        "read: dedup line shown",
    )
    status_rows = [
        row for row in _ansi(rendered, width=200).splitlines() if "Syntax check" in row
    ]
    assert_true(len(status_rows) == 1, f"read: dedup line found in ansi: {len(status_rows)}")
    assert_true(
        _bg_triplet(theme.CODE_BG) not in status_rows[0],
        f"read: dedup line not painted inside the code block: {status_rows[0]!r}",
    )
    assert_true(
        _fg_triplet(theme.ERROR_SOFT) in status_rows[0],
        f"read: syntax-status colour used: {status_rows[0]!r}",
    )

    # A section that keeps new rows but summarises the repeats with a bullet.
    bullet = ToolBlock("beginTransaction")
    bullet.args_buf = '{"file":"example.py"}'
    bullet.full_result = (
        "example.py (ids: each line is '<line><hash>|<content>'):\n"
        "1af|def f():\n"
        "230|    return 1\n"
        "\n"
        "Syntax check (python, lsp) — 3 issue(s); fix each line below with edit, "
        "then it is re-checked:\n"
        "- line 2, col 5: [undefined-name] nope\n"
        "- 2 unchanged issue(s) already reported above (not repeated)"
    )
    text = _plain(ChatView()._render_tool_result(bullet))
    assert_true(
        "- 2 unchanged issue(s) already reported above (not repeated)" in text,
        "unchanged-bullet shown",
    )
    assert_true(
        re.search(r"\d+ +- 2 unchanged", text) is None,
        f"unchanged-bullet carries no code gutter number: {text}",
    )


def test_edit_call_streams_content_live() -> None:
    # A streaming edit call must type out the code being written (recovered
    # from the partial XML like the bash command), not freeze on `file=...`.
    from widgets.chat.tool_args import _recover_string_arg

    partial_xml = (
        "\n<function=edit>\n<parameter=file>\ncalc.py\n</parameter>\n"
        "<parameter=id>\n14d\n</parameter>\n<parameter=content>\ndef add(a, b):\n    ret"
    )
    assert_true(
        _recover_string_arg(partial_xml, "content") == "def add(a, b):\n    ret",
        f"partial content recovered: {_recover_string_arg(partial_xml, 'content')!r}",
    )
    view = ChatView()
    block = ToolBlock("edit")
    block.args_buf = partial_xml
    target = view._call_target(block)
    assert_true(target == "def add(a, b):\n    ret", f"reveal target is the content: {target!r}")
    block.reveal = len(target)
    text = _plain(view._render_tool_call(block))
    assert_true("def add(a, b):" in text, f"code streams in the call box: {text}")
    assert_true("<parameter" not in text and "file=" not in text, f"no raw XML/arg noise: {text}")

    # Finalized JSON args keep the same target (no shape flip at the end).
    block.args_buf = '{"file":"calc.py","id":"14d","content":"def add(a, b):\\n    return a + b"}'
    assert_true(
        view._call_target(block) == "def add(a, b):\n    return a + b",
        "finalized JSON content is the target",
    )

    # Reveal slices the highlighted code: a short budget hides the second line.
    from widgets.chat.renderers.code import _streaming_code_text

    sliced = _plain(_streaming_code_text("calc.py", "def add(a, b):\n    return a + b", reveal=14))
    assert_true("def add(a, b)" in sliced and "return" not in sliced,
                f"reveal budget trims unrevealed lines: {sliced}")

    # A delete call has no content: the compact detail stays the target.
    block = ToolBlock("delete")
    block.args_buf = '{"file":"calc.py","id":"14d"}'
    assert_true("file='calc.py'" in view._call_target(block), "delete falls back to detail")


def _spawn_edit_block(view: ChatView, filename: str, line_ranges: str) -> ToolBlock:
    """A delegator edit mid-stream: filename + line_ranges closed, the long
    instructions parameter still open — the moment the capture fires."""
    block = ToolBlock("edit")
    block.args_buf = (
        f"\n<function=edit>\n<parameter=filename>\n{filename}\n</parameter>\n"
        f"<parameter=line_ranges>\n{line_ranges}\n</parameter>\n"
        "<parameter=instructions>\nRework the helper"
    )
    block.reveal = len(view._call_target(block))
    return block


def test_spawn_snippet_capture_slices_and_caps() -> None:
    # The Spawn card shows the named ranges only (no margin, no small-file
    # widening — a focus view), as hashline id rows (the Editor's dialect),
    # with the 20-line collapsed cap and the Ctrl+o hint.
    import re
    import tempfile

    view = ChatView()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "big.py")
        Path(path).write_text("".join(f"line_{i} = {i}\n" for i in range(1, 301)))
        block = _spawn_edit_block(view, path, "100-150")
        text = _plain(view._render_tool_call(block))

    snippet = block.spawn_snippet
    assert_true(snippet is not None and snippet.spans == [(100, 150)],
                f"exactly the named range, no widening: {snippet and snippet.spans}")
    assert_true(re.search(r"100[0-9a-f]{2}  line_100 = 100", text),
                f"hashline id rows start at 100: {text}")
    assert_true(re.search(r"119[0-9a-f]{2}  line_119 = 119", text),
                "collapsed cap keeps 20 code lines")
    assert_true("line_120" not in text, f"line 21 hidden while collapsed: {text}")
    assert_true("... 31 more lines (Ctrl+o to expand)" in text, f"cap hint: {text}")
    assert_true("Rework the helper" in text, f"instructions still type out below: {text}")

    block.expanded = True
    text = _plain(view._render_tool_call(block))
    assert_true(re.search(r"150[0-9a-f]{2}  line_150 = 150", text),
                f"expanded shows the whole slice: {text}")
    assert_true("more lines" not in text, "no hint when expanded")

    block.expanded = False
    view.tools_expanded = True
    text = _plain(view._render_tool_call(block))
    assert_true(re.search(r"150[0-9a-f]{2}  line_150 = 150", text),
                "view-wide Ctrl+o expands streaming cards too")


def test_spawn_snippet_multi_span() -> None:
    import re
    import tempfile

    view = ChatView()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "big.py")
        Path(path).write_text("".join(f"line_{i} = {i}\n" for i in range(1, 301)))
        block = _spawn_edit_block(view, path, "10-12, 250-255")
        text = _plain(view._render_tool_call(block))

        assert_true(block.spawn_snippet.spans == [(10, 12), (250, 255)],
                    f"exactly the two named spans: {block.spawn_snippet.spans}")
        assert_true(re.search(r"10[0-9a-f]{2}  line_10 = 10", text), f"first span shown: {text}")
        assert_true(re.search(r"250[0-9a-f]{2}  line_250 = 250", text), "second span shown")
        assert_true("... (237 lines not shown) ..." in text,
                    f"gap separator between spans: {text}")
        assert_true("more lines (Ctrl+o" not in text, "9 rows fit under the cap: no hint")

        # A wide first span spends the whole collapsed budget before span two:
        # no dangling separator, and the hint counts hidden rows of BOTH spans.
        block = _spawn_edit_block(view, path, "1-30, 250-255")
        text = _plain(view._render_tool_call(block))
    assert_true(re.search(r"20[0-9a-f]{2}  line_20 = 20", text), "budget caps inside span one")
    assert_true("line_21 " not in text, f"line 21 hidden: {text}")
    assert_true("... 16 more lines (Ctrl+o to expand)" in text,
                f"hidden count spans both ranges: {text}")
    assert_true("lines not shown" not in text, "no dangling separator after the cut")


def test_spawn_snippet_all_and_exact_small_ranges() -> None:
    import tempfile

    view = ChatView()
    with tempfile.TemporaryDirectory() as tmp:
        big = str(Path(tmp) / "big.py")
        Path(big).write_text("".join(f"line_{i} = {i}\n" for i in range(1, 301)))
        block = _spawn_edit_block(view, big, "all")
        text = _plain(view._render_tool_call(block))
        assert_true(block.spawn_snippet.spans == [(1, 300)], "all selects the whole file")
        assert_true("... 280 more lines (Ctrl+o to expand)" in text,
                    f"whole file still capped collapsed: {text}")

        small = str(Path(tmp) / "small.py")
        Path(small).write_text("".join(f"s_{i} = {i}\n" for i in range(1, 51)))
        block = _spawn_edit_block(view, small, "40-45")
        _plain(view._render_tool_call(block))
        assert_true(block.spawn_snippet.spans == [(40, 45)],
                    "small files keep the named range too — the card is a focus view")


def test_spawn_snippet_waits_for_closed_ranges_and_freezes_pre_edit() -> None:
    import re
    import tempfile

    view = ChatView()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "small.py")
        Path(path).write_text("".join(f"s_{i} = {i}\n" for i in range(1, 51)))

        # line_ranges still streaming ("12-1" of "12-18"): no capture, and the
        # one-shot flag stays clear so a later tick retries.
        block = ToolBlock("edit")
        block.args_buf = (
            f"\n<function=edit>\n<parameter=filename>\n{path}\n</parameter>\n"
            "<parameter=line_ranges>\n12-1"
        )
        _plain(view._render_tool_call(block))
        assert_true(block.spawn_snippet is None, "no snippet from a half-streamed range")
        assert_true(not block.spawn_snippet_tried, "capture not burned on partial args")

        # The parameter closes: captured on the next repaint.
        block.args_buf = (
            f"\n<function=edit>\n<parameter=filename>\n{path}\n</parameter>\n"
            "<parameter=line_ranges>\n12-18\n</parameter>\n"
            "<parameter=instructions>\nRework"
        )
        block.reveal = len(view._call_target(block))
        _plain(view._render_tool_call(block))
        assert_true(block.spawn_snippet is not None, "snippet captured once the range closes")

        # The editor rewrites the file; the result card must keep the frozen
        # pre-edit slice (never re-read post-edit disk) above the diff.
        Path(path).write_text("changed = True\n")
        block.stream_done = True
        block.full_result = (
            f"Editor result for {path} (1 edit applied):\nDone.\n\n"
            "Diff:\n"
            f"--- a/{path}\n+++ b/{path}\n"
            "@@ -12,1 +12,1 @@\n"
            "-s_12 = 12\n"
            "+changed = True\n"
        )
        text = _plain(view._render_tool_result(block))
        assert_true(re.search(r"12[0-9a-f]{2}  s_12 = 12", text),
                    f"pre-edit slice frozen in the result card: {text}")
        assert_true("Diff" in text, "editor diff still rendered beneath the card")

        # Unreadable file: one attempt, no snippet, no crash.
        missing = _spawn_edit_block(view, str(Path(tmp) / "nope.py"), "1-2")
        text = _plain(view._render_tool_call(missing))
        assert_true(missing.spawn_snippet is None and missing.spawn_snippet_tried,
                    "missing file burns the single attempt without a snippet")
        assert_true("Spawn" in text, f"card still renders without a slice: {text}")


def test_inspect_result_renders_numbered_code() -> None:
    from widgets.chat.renderers.code import _split_inspect_result

    paged = "Showing lines 5-6 of 40 total lines.\n\n---\n\ndef f():\n    return 1"
    start, total, content, marker = _split_inspect_result(paged)
    assert_true((start, total, content, marker) == (5, 40, ["def f():", "    return 1"], None),
                f"paged window split: {(start, total, content, marker)}")
    whole = _split_inspect_result("x = 1\ny = 2\n[compressed: comments pruned]")
    assert_true(whole == (1, None, ["x = 1", "y = 2"], "[compressed: comments pruned]"),
                f"whole file + marker split: {whole}")

    block = ToolBlock("inspect_file")
    block.args_buf = '{"file_path": "/repo/app.py"}'
    block.full_result = paged
    text = _plain(ChatView()._render_tool_result(block))
    assert_true("/repo/app.py (lines 5-6 of 40)" in text, f"inspect header: {text}")
    assert_true("5  def f():" in text, f"numbered content row: {text}")
    assert_true("6      return 1" in text, "line numbers continue from the window start")
    assert_true("Result" not in text, "generic Result label replaced")

    err = ToolBlock("inspect_file")
    err.args_buf = '{"file_path": "/nope"}'
    err.full_result = "Error: File not found: /nope"
    err.is_error = True
    err_text = _plain(ChatView()._render_tool_result(err))
    assert_true("Result" in err_text and "Error: File not found" in err_text,
                "error keeps the plain fallback box")


def test_inspect_result_carves_lsp_section_from_code() -> None:
    # inspect_file attaches `LSP diagnostics (…)` as a trailing section; it
    # must never render as numbered code rows, and the line count in the
    # header must describe only the file content.
    import re

    import theme

    block = ToolBlock("inspect_file")
    block.args_buf = '{"file_path": "/repo/app.py"}'
    block.full_result = (
        "x = 1\ny = 2\n"
        "\n"
        "LSP diagnostics (python, pylsp) — 1 error(s), 1 warning(s):\n"
        "- line 1, col 1: [undefined-name] nope\n"
        "- 1 unchanged warning(s) already reported above (not repeated)"
    )
    rendered = ChatView()._render_tool_result(block)
    text = _plain(rendered)
    assert_true(
        "/repo/app.py (2 lines)" in text,
        f"header counts only file lines, not the section: {text}",
    )
    assert_true("2  y = 2" in text, "numbered content kept")
    assert_true("LSP diagnostics (python, pylsp)" in text, "section shown")
    assert_true(
        "- 1 unchanged warning(s) already reported above (not repeated)" in text,
        "dedup bullet shown",
    )
    assert_true(
        re.search(r"\d+ +LSP diagnostics", text) is None,
        f"section is not a numbered code row: {text}",
    )
    status_rows = [
        row for row in _ansi(rendered, width=200).splitlines() if "LSP diagnostics" in row
    ]
    assert_true(len(status_rows) == 1, f"section header found in ansi: {len(status_rows)}")
    assert_true(
        _bg_triplet(theme.CODE_BG) not in status_rows[0],
        f"section not painted on the code background: {status_rows[0]!r}",
    )
    assert_true(
        _fg_triplet(theme.ERROR_SOFT) in status_rows[0],
        f"syntax-status colour used: {status_rows[0]!r}",
    )

    # Compressed reads attach the section AFTER the marker; carving it back
    # off restores the marker as the last body line, so its dim row renders
    # and a single-line all-unchanged section still gets the status route.
    packed = ToolBlock("inspect_file")
    packed.args_buf = '{"file_path": "/repo/app.py"}'
    packed.full_result = (
        "x = 1\n"
        "[compressed: 10 of 20 chars elided; re-call with compress:false for the full output]\n"
        "\n"
        "LSP diagnostics (python, pylsp) — 0 error(s), 9 warning(s): "
        "all unchanged and already reported above."
    )
    text = _plain(ChatView()._render_tool_result(packed))
    assert_true("[compressed: 10 of 20 chars elided" in text, "marker row kept")
    assert_true(
        "all unchanged and already reported above." in _joined(text),
        "single-line section shown",
    )
    assert_true(
        re.search(r"\d+ +\[compressed", text) is None
        and re.search(r"\d+ +LSP diagnostics", text) is None,
        f"neither marker nor section numbered as code: {text}",
    )


def test_syntax_status_renders_structured_section() -> None:
    # The section renderer is structured per line, not one flat paint: header
    # keeps the status colour, `edit id:`/`line N:` echoes become anchor-style
    # code rows, summaries dim, warnings wear the warning tone.
    import theme
    from widgets.chat.renderers.code import _render_syntax_status

    section = (
        "Syntax check (javascript, lsp) — 2 issue(s); fix each line below with "
        "edit, then it is re-checked:\n"
        "- line 116, col 3: [no-undef] 'birdElement' is not defined (eslint)\n"
        "  edit id: 11685|  birdElement.style.left = `${x}px`;\n"
        "- line 3, col 1: [semi] missing semicolon (eslint)\n"
        "  line 3: const speed = 5\n"
        "- … and 1 more\n"
        "Warnings (not blocking) — 2:\n"
        "- line 2, col 1: [camelcase] identifier 'bird_x' (eslint)\n"
        "- 1 unchanged warning(s) already reported above (not repeated)"
    )
    rendered = _render_syntax_status(section, "bird.js")
    text = _plain(rendered)
    assert_true("edit id: 11685|  birdElement" not in text, "raw echo pipe replaced")
    assert_true(
        "edit id: 11685    birdElement.style.left" in text,
        f"hashline echo wears the gutter form: {text}",
    )
    assert_true("  line 3: const speed" not in text, "raw plain echo prefix replaced")
    assert_true("line 3  const speed = 5" in text, f"plain echo wears the gutter form: {text}")

    # Echo-row content is syntax-highlighted (ANSI codes interleave inside it),
    # so the styled checks anchor on each row's single-span gutter label; hit
    # counters prove every check actually fired.
    soft = _fg_triplet(theme.ERROR_SOFT)
    hits = {"header": 0, "bullet": 0, "hash_echo": 0, "plain_echo": 0, "dim": 0, "warn": 0}
    for row in _ansi(rendered, width=200).splitlines():
        if "Syntax check (javascript" in row:
            hits["header"] += 1
            assert_true(soft in row, f"header wears the status colour: {row!r}")
        if "no-undef" in row:
            hits["bullet"] += 1
            assert_true(soft in row, f"issue bullet wears the status colour: {row!r}")
        if "edit id: 11685" in row:
            hits["hash_echo"] += 1
            assert_true(
                _bg_triplet(theme.CODE_BG) in row and _fg_triplet(theme.ACCENT) in row,
                f"hashline echo: accent gutter on the code bg: {row!r}",
            )
        if "line 3  " in row:
            hits["plain_echo"] += 1
            assert_true(
                _bg_triplet(theme.CODE_BG) in row and _fg_triplet(theme.ACCENT) in row,
                f"plain echo: accent gutter on the code bg: {row!r}",
            )
        if "and 1 more" in row or "not repeated" in row:
            hits["dim"] += 1
            assert_true(soft not in row and "\x1b[2m" in row, f"summary bullet dim: {row!r}")
        if "Warnings (not blocking)" in row or "camelcase" in row:
            hits["warn"] += 1
            assert_true(
                _fg_triplet(theme.WARNING) in row, f"warning tone used: {row!r}"
            )
    assert_true(
        hits == {"header": 1, "bullet": 1, "hash_echo": 1, "plain_echo": 1, "dim": 2, "warn": 2},
        f"every styled check fired: {hits}",
    )

    # No path: never crash, echoes still get the gutter treatment (plain tone).
    bare = _plain(_render_syntax_status(section))
    assert_true("line 3  const speed = 5" in bare, f"no-path fallback still structured: {bare}")

    # The classic single-line shapes keep their colours.
    ok = _render_syntax_status("Syntax check (python): OK — no syntax errors.")
    ok_ansi = _ansi(ok, width=200)
    assert_true(_fg_triplet(theme.SUCCESS) in ok_ansi, "OK line stays status green")
    dedup = _render_syntax_status(
        "Syntax check (python, lsp) — 9 issue(s), all unchanged and already reported above."
    )
    assert_true(_fg_triplet(theme.ERROR_SOFT) in _ansi(dedup, width=200),
                "all-unchanged one-liner keeps the soft error header colour")


def main() -> None:
    test_read_result_keeps_anchors()
    test_edit_result_shows_diff_before_refreshed_labels()
    test_attention_flagged_edit_still_shows_colored_diff()
    test_hard_edit_error_without_diff_falls_back_to_plain_result()
    test_read_shows_syntax_status()
    test_edit_result_without_snippet_keeps_syntax_out_of_diff()
    test_edit_result_with_snippet_keeps_syntax_status_outside_code_block()
    test_all_unchanged_single_line_section_renders_as_status()
    test_edit_call_streams_content_live()
    test_spawn_snippet_capture_slices_and_caps()
    test_spawn_snippet_multi_span()
    test_spawn_snippet_all_and_exact_small_ranges()
    test_spawn_snippet_waits_for_closed_ranges_and_freezes_pre_edit()
    test_inspect_result_renders_numbered_code()
    test_inspect_result_carves_lsp_section_from_code()
    test_syntax_status_renders_structured_section()
    print("code renderer tests passed")


if __name__ == "__main__":
    main()
