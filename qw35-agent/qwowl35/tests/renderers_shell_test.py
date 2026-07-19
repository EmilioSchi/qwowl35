"""Tests for shell-call rendering: command/output rows, the type-out
reveal, advisories, and the mini terminal window chrome."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from widgets.chat import ChatView, ToolBlock  # noqa: E402
from widgets.chat.primitives import _FullWidthLines, _line_with_bg  # noqa: E402
from widgets.chat.renderers.shell import _advisory_segments, _command_rows  # noqa: E402
from widgets.chat.renderers.shell import _split_bash_advisories  # noqa: E402
from widgets.chat.terminal_chrome import _head_capped_rows, _prompt_text  # noqa: E402
from widgets.chat.terminal_chrome import _term_bg, _window_title_row  # noqa: E402
from widgets.status_bar import rough_token_count  # noqa: E402

import theme  # noqa: E402

from chat_test_helpers import (  # noqa: E402
    _ansi,
    _fg_triplet,
    _plain,
    _shell_result_block,
    _textual_content_height,
    assert_true,
)


def test_bash_result_height_includes_output_row() -> None:
    # The concrete symptom: `$ date` and its output were measured as 2 rows, so
    # Textual cropped the output line to a blank. Badge + command + output = 3.
    block = ToolBlock("bash")
    block.args_buf = '{"command":"date"}'
    block.full_result = "Wed Jul  1 21:11:27 CEST 2026\n"
    height = _textual_content_height(ChatView()._render_tool_result(block))
    assert_true(height >= 3, f"result measured tall enough to show output (got {height})")


def test_bash_result_keeps_command_visible() -> None:
    block = ToolBlock("bash")
    block.args_buf = '{"command":"python3 -m py_compile qwowl35/widgets/chat_log.py"}'
    block.full_result = "ok\n"
    # Pin the prompt identity: the real host/cwd vary by machine and could
    # push the command line past the test console's width, wrapping it.
    block.prompt_host = "meg"
    block.prompt_path = "~"

    text = _plain(ChatView()._render_tool_result(block))

    # The "Sh" badge became a mini terminal window: title bar with a stable
    # per-window id + `- □ x` controls, a PS1 prompt line, a copy/time footer.
    assert_true(f"terminal #{block.term_hash}" in text, "window title names the terminal")
    for control in (" - ", " □ ", " x "):
        assert_true(control in text, f"title bar control {control.strip()!r} shown")
    assert_true("qwowl@" in text, "PS1 prompt shown")
    assert_true("$ python3 -m py_compile qwowl35/widgets/chat_log.py" in text, "command shown")
    # Output sits in the same mini-terminal box as the command, distinguished by
    # color only — no bold "Output" label anymore.
    assert_true("ok" in text, "output shown")
    assert_true("copy" in text, "footer copy button shown")
    assert_true(block.started_at in text, "footer shows the run time")


def test_bash_call_recovers_xml_command_for_terminal_preview() -> None:
    block = ToolBlock("bash")
    block.args_buf = (
        "<tool_call>\n"
        "<function=bash>\n"
        '<parameter=command>printf "ok"</parameter>\n'
        "</function>\n"
        "</tool_call>"
    )
    block.reveal = 11

    text = _plain(ChatView()._render_tool_call(block))

    assert_true('$ printf "ok"' in text, "xml command shown as shell")
    assert_true("<parameter=command>" not in text, "raw xml hidden")


def test_bash_call_uses_hidden_bash_syntax_pass() -> None:
    block = ToolBlock("bash")
    block.args_buf = '{"command":"printf \\"ok\\" && echo done"}'
    block.reveal = len('printf "ok" && echo done')

    ansi = _ansi(ChatView()._render_tool_call(block), width=80)

    # The body sits on the derived terminal background (BG_BASE pushed darker).
    h = _term_bg().lstrip("#")
    bg = f"48;2;{int(h[0:2], 16)};{int(h[2:4], 16)};{int(h[4:6], 16)}"
    assert_true(f"38;2;230;219;116;{bg}m" in ansi, "bash string syntax color emitted")
    assert_true(f"{bg}m" in ansi, "terminal body background emitted")
    assert_true("\x1b[1m" not in ansi, "bash terminal text is not bold")


_HEREDOC_CMD = "cat <<EOF > f.txt\nhello world\nEOF"


def _row_ansi(rows, width: int = 80) -> str:
    return _ansi(_FullWidthLines(rows, wrap=True), width=width)


def _revealed_command(rows) -> str:
    """Visible command text with the leading ``$ ``/``> `` prompt stripped."""
    return "\n".join(row.text.plain[2:] for row in rows)


def test_bash_reveal_plain_tracks_prefix() -> None:
    # The visible command at reveal=k must equal command[:k] (newlines counted),
    # so the type-out reveals exactly the streamed prefix and nothing more. A
    # reveal landing exactly on a newline shows no trailing empty row (the next
    # line hasn't started), matching plain ``splitlines`` behaviour.
    for k in range(1, len(_HEREDOC_CMD) + 1):
        rows = _command_rows(_HEREDOC_CMD, cursor=False, reveal=k)
        expected = _HEREDOC_CMD[:k].split("\n")
        if expected and expected[-1] == "":
            expected = expected[:-1]
        assert_true(
            _revealed_command(rows) == "\n".join(expected),
            f"reveal={k} shows exactly the prefix",
        )


def test_bash_reveal_highlights_full_command_not_prefix() -> None:
    # Heredoc body colour only exists once the lexer sees the *whole* command.
    # Cutting mid-body and lexing that prefix alone (the old bug) would leave it
    # default-foreground; highlighting the full command keeps the string colour.
    k = _HEREDOC_CMD.index("world")  # cut inside the heredoc body
    partial = _row_ansi(_command_rows(_HEREDOC_CMD, cursor=False, reveal=k))
    prefix_lexed = _row_ansi(_command_rows(_HEREDOC_CMD[:k], cursor=False))

    assert_true("230;219;116" in partial, "heredoc body keeps string colour mid-reveal")
    assert_true("230;219;116" not in prefix_lexed, "prefix-only lexing loses the colour (old bug)")


def test_bash_reveal_colours_are_stable_across_steps() -> None:
    # Same characters must carry the same colour at every reveal step — no
    # frame-to-frame flicker as the type-out crosses token boundaries.
    body = _HEREDOC_CMD.index("hello")
    early = _row_ansi(_command_rows(_HEREDOC_CMD, cursor=False, reveal=body + 3))
    later = _row_ansi(_command_rows(_HEREDOC_CMD, cursor=False, reveal=body + 5))
    full = _row_ansi(_command_rows(_HEREDOC_CMD, cursor=False, reveal=None))

    # The "$ cat <<EOF > f.txt" first row renders identically regardless of how
    # far the reveal has progressed.
    first_line = lambda ansi: ansi.split("\n", 1)[0]
    assert_true(first_line(early) == first_line(full), "revealed first row matches final colours")
    assert_true(first_line(later) == first_line(full), "still matches one step later")


def _long_heredoc_write(n_lines: int = 100) -> str:
    body = "\n".join(f"CONST_{i} = compute({i}, base={i * 7})" for i in range(n_lines))
    return f"cat > big.py <<'EOF'\n{body}\nEOF"


def _reveal_steps(target: str):
    """Replays the _tick reveal schedule, yielding each reveal length."""
    from widgets.chat.chat_view import _REVEAL_MAX_STEP, _REVEAL_MIN_STEP

    reveal = 0
    while reveal < len(target):
        step = min(
            _REVEAL_MAX_STEP,
            max(_REVEAL_MIN_STEP, (len(target) - reveal) // 8),
        )
        reveal = min(len(target), reveal + step)
        yield reveal


def test_long_multiline_write_types_out_line_by_line() -> None:
    # Regression: a very long `cat <<EOF` write used to dump ~14 lines in the
    # first frame because the adaptive reveal step had no upper bound. It must
    # now reveal at most a couple of new lines per frame so it visibly animates.
    cmd = _long_heredoc_write(100)
    prev = 0
    max_new = 0
    for reveal in _reveal_steps(cmd):
        rows = _command_rows(cmd, cursor=False, reveal=reveal)
        max_new = max(max_new, len(rows) - prev)
        prev = len(rows)
    assert_true(prev == len(cmd.splitlines()), "every line revealed by the end")
    assert_true(max_new <= 3, f"types out gradually, not dumped ({max_new} lines/frame)")


def test_long_multiline_write_reveal_matches_prefix() -> None:
    # Each animation frame shows exactly the streamed prefix (newline-counted),
    # for a long multiline command across the whole reveal schedule.
    cmd = _long_heredoc_write(60)
    for reveal in _reveal_steps(cmd):
        rows = _command_rows(cmd, cursor=False, reveal=reveal)
        visible = "\n".join(row.text.plain[2:] for row in rows)
        expected = cmd[:reveal].split("\n")
        if expected and expected[-1] == "":
            expected = expected[:-1]
        assert_true(visible == "\n".join(expected), f"prefix matches at reveal={reveal}")


def test_long_write_full_render_shows_all_lines() -> None:
    # The approval modal renders the command with no reveal budget (full); every
    # line of a long write must be present.
    cmd = _long_heredoc_write(50)
    rows = _command_rows(cmd, cursor=False, reveal=None)
    assert_true(len(rows) == len(cmd.splitlines()), "all lines rendered at full reveal")
    text = _plain(_FullWidthLines(rows, wrap=True))
    assert_true("CONST_0 = compute(0" in text, "first body line shown")
    assert_true("CONST_49 = compute(49" in text, "last body line shown")


def test_bash_result_recovers_malformed_quoted_command() -> None:
    block = ToolBlock("bash")
    block.args_buf = '{"command":"printf "ok""}'
    block.full_result = "ok"

    text = _plain(ChatView()._render_tool_result(block))

    assert_true('$ printf "ok"' in text, "recovered command shown")


def test_bash_result_trims_dangling_recovered_quote() -> None:
    block = ToolBlock("bash")
    block.args_buf = '{"command":"touch cal.py && echo ""}'
    block.full_result = "\n"

    text = _plain(ChatView()._render_tool_result(block))

    assert_true("$ touch cal.py && echo" in text, "trimmed command shown")
    assert_true('$ touch cal.py && echo "' not in text, "dangling quote hidden")


def test_bash_result_recovers_command_equals_typo() -> None:
    block = ToolBlock("bash")
    block.args_buf = '{"command="python3 cal.py"}}'
    block.full_result = "ok"

    text = _plain(ChatView()._render_tool_result(block))

    assert_true("$ python3 cal.py" in text, "equals typo command shown")
    assert_true('{"command=' not in text, "raw malformed json hidden")


_AUTOREAD_RESULT = (
    "(no output)\n\n"
    "You just wrote `greet.py`. Its current line ids are below — edit it in "
    "place with edit; no separate beginTransaction needed:\n"
    "greet.py (ids: each line is '<line><hash>|<content>'):\n"
    "17b|def greet():\n"
    "2a7|    return 1\n\n"
    "Syntax check (python): OK — no syntax errors."
)


def test_split_bash_advisories() -> None:
    out, adv = _split_bash_advisories(_AUTOREAD_RESULT)
    assert_true(out == "(no output)", f"command output isolated: {out!r}")
    assert_true(adv.startswith("You just wrote `greet.py`"), "advisory starts at the marker")
    # A plain command result with no appended block has no advisory region.
    plain_out, plain_adv = _split_bash_advisories("hello\nworld\n")
    assert_true(plain_out == "hello\nworld\n" and plain_adv == "", "no advisory when none appended")


def test_advisory_segments_classifies() -> None:
    _, adv = _split_bash_advisories(_AUTOREAD_RESULT)
    kinds = [kind for kind, _ in _advisory_segments(adv)]
    assert_true(kinds == ["autoread", "syntax"], f"classified segments: {kinds}")


def test_bash_post_write_anchors_render_as_file_view() -> None:
    block = ToolBlock("bash")
    block.args_buf = '{"command":"cat > greet.py <<EOF\\nx\\nEOF"}'
    block.full_result = _AUTOREAD_RESULT

    text = _plain(ChatView()._render_tool_result(block))

    assert_true("(no output)" in text, "command output still shown in the box")
    assert_true("Model also received" in text, "advisory preview is labelled")
    assert_true("You just wrote `greet.py`" in text, "auto-read intro shown")
    # The anchors render with the read file view (anchor gutter + code), so the
    # raw 'line:hash|content' text is gone.
    assert_true("17b" in text and "def greet():" in text, "anchors rendered as code")
    assert_true("17b|def greet():" not in text, "raw anchor pipe replaced by file view")
    assert_true("Syntax check (python): OK" in text, "clean-file confirmation shown")


def test_render_is_display_only_and_non_mutating() -> None:
    # The renderer must add its decoration (file-view gutter, "Model also received"
    # label, badges) WITHOUT touching the source string the model receives.
    block = ToolBlock("bash")
    block.args_buf = '{"command":"cat > app.py <<EOF\\nx\\nEOF"}'
    block.full_result = _AUTOREAD_RESULT
    source = block.full_result

    text = _plain(ChatView()._render_tool_result(block))

    assert_true(block.full_result == source, "render did not mutate full_result")
    # Decoration is present in the view but absent from the source the model sees.
    assert_true("Model also received" in text and "Model also received" not in source,
                "preview label is render-only")
    assert_true("qwowl@" in text and "qwowl@" not in source, "terminal chrome is render-only")
    assert_true("17b|def greet():" in source, "source keeps the raw pipe anchors")
    assert_true("17b|def greet():" not in text, "render replaces the pipe with the file-view gutter")
    assert_true("17b  def greet():" in text, "rendered anchors use the gutter form")

    # Same separation for a plain read result.
    rf = ToolBlock("beginTransaction")
    rf.args_buf = '{"file":"app.py"}'
    rf.full_result = (
        "app.py (ids: each line is '<line><hash>|<content>'):\n"
        "17b|def f():\n2a7|    return 1"
    )
    rf_source = rf.full_result
    rf_text = _plain(ChatView()._render_tool_result(rf))
    assert_true(rf.full_result == rf_source, "read render did not mutate full_result")
    assert_true("Read" in rf_text and "Read" not in rf_source, "read badge is render-only")
    assert_true("17b|def f():" in rf_source and "17b|def f():" not in rf_text,
                "read source keeps the pipe; render uses the gutter")


def _bg_triplet(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"48;2;{int(h[0:2], 16)};{int(h[2:4], 16)};{int(h[4:6], 16)}"


_POST_WRITE_REPORT = (
    "(no output)\n\n"
    "You just wrote `calc.py` — it has problems. Syntax check (python, lsp) — 3 issue(s). "
    "Fix ONLY these lines with the `edit` tool (filename, line ranges, instructions); "
    "do NOT rewrite the file through bash:\n"
    "- line 1, col 8: [undefined-name] undefined name 'nope' (pyflakes)\n"
    "  line 1: value = nope()\n"
    "- … and 1 more\n"
    "- 2 unchanged issue(s) already reported above (not repeated)\n"
    "Warnings (not blocking) — 1:\n"
    "- line 2, col 1: [unused-import] os (pylint)"
)


def _report_block(body: str) -> ToolBlock:
    block = ToolBlock("bash")
    block.args_buf = '{"command":"cat > calc.py <<EOF\\nx\\nEOF"}'
    block.full_result = body
    return block


def test_advisory_classifies_post_write_report_shapes() -> None:
    # The plain post-write report carries its Syntax check headline on the
    # intro line — that separates it from the hashline auto-read intro, which
    # never does. Clean `Wrote …` confirmations get their own lane.
    _, adv = _split_bash_advisories(_POST_WRITE_REPORT)
    kinds = [kind for kind, _ in _advisory_segments(adv)]
    assert_true(kinds == ["report"], f"problem report classified: {kinds}")

    one_liner = (
        "You just wrote `calc.py` — it still has problems. Syntax check "
        "(python, lsp) — 9 issue(s), all unchanged and already reported above."
    )
    kinds = [kind for kind, _ in _advisory_segments(one_liner)]
    assert_true(kinds == ["report"], f"all-unchanged one-liner classified: {kinds}")

    multi = (
        "You just wrote `a.py` — it has problems. Syntax check (python) — 1 issue(s):\n"
        "- line 1, col 1: unexpected 'def oops(' (tree-sitter)\n"
        "  line 1: def oops(\n\n"
        "Wrote `b.py` (12 lines). Syntax check (python, lsp): OK — no syntax errors.\n"
        "Warnings (not blocking) — 1:\n"
        "- line 2, col 1: [unused-import] os (pylint)\n\n"
        "(+1 more written file(s) not shown.)"
    )
    kinds = [kind for kind, _ in _advisory_segments(multi)]
    assert_true(kinds == ["report", "written", "note"], f"multi-file classified: {kinds}")


def test_post_write_report_renders_pretty() -> None:
    # The screenshot bug: the whole report painted as one flat red text dump.
    # Now: muted intro + status-coloured headline, soft-red bullets, dim
    # summary bullets, warning-toned warnings, and the echoed file rows as
    # gutter-labelled, syntax-highlighted code rows.
    block = _report_block(_POST_WRITE_REPORT)
    rendered = ChatView()._render_tool_result(block)
    text = _plain(rendered)
    assert_true("Model also received" in text, "advisory preview labelled")
    assert_true("You just wrote `calc.py`" in text, "intro shown")
    assert_true("- line 1, col 8:" in text, "issue bullet shown")
    assert_true("  line 1: value = nope()" not in text, "raw echo prefix replaced")
    assert_true("line 1  value = nope()" in text, f"echo row wears the gutter form: {text}")

    # Echo-row content is syntax-highlighted (ANSI codes interleave inside it),
    # so the echo check anchors on the row's single-span gutter label; hit
    # counters prove every styled check actually fired.
    ansi = _ansi(rendered, width=400)
    muted, soft = _fg_triplet(theme.FG_MUTED), _fg_triplet(theme.ERROR_SOFT)
    hits = {"header": 0, "bullet": 0, "echo": 0, "dim": 0, "warn": 0}
    for row in ansi.splitlines():
        if "You just wrote" in row:
            hits["header"] += 1
            assert_true(muted in row and soft in row, f"header wears both tones: {row!r}")
            assert_true(
                row.find(muted) < row.find("Syntax check"),
                "intro muted before the status-coloured headline",
            )
            # The tone is an ACCENT on the count, never a red wall: the soft
            # span opens at the count fragment, and the long instruction
            # clause after it is back to FG_DIM (the most recent colour
            # before "Fix ONLY" must be dim, not soft).
            assert_true(
                row.find("Syntax check") < row.find(soft) < row.find("issue(s)"),
                f"soft tone opens at the count: {row!r}",
            )
            fix_at = row.find("Fix ONLY")
            assert_true(fix_at > 0, f"instruction clause present: {row!r}")
            dim_fg = _fg_triplet(theme.FG_DIM)
            assert_true(
                row.rfind(dim_fg, 0, fix_at) > row.rfind(soft, 0, fix_at),
                f"instruction clause dim, not red: {row!r}",
            )
        if "col 8" in row:
            hits["bullet"] += 1
            assert_true(soft in row, f"issue bullet wears the status colour: {row!r}")
        if "line 1  " in row:
            hits["echo"] += 1
            assert_true(_bg_triplet(theme.CODE_BG) in row, f"echo row on the code bg: {row!r}")
            assert_true(_fg_triplet(theme.ACCENT) in row, f"echo gutter wears the accent: {row!r}")
        if "and 1 more" in row or "not repeated" in row:
            hits["dim"] += 1
            assert_true(soft not in row and "\x1b[2m" in row, f"summary bullet dim: {row!r}")
        if "Warnings (not blocking)" in row or "unused-import" in row:
            hits["warn"] += 1
            assert_true(_fg_triplet(theme.WARNING) in row, f"warning tone used: {row!r}")
    assert_true(
        hits == {"header": 1, "bullet": 1, "echo": 1, "dim": 2, "warn": 2},
        f"every styled check fired: {hits}",
    )


def test_post_write_report_respects_preview_cap() -> None:
    issue_rows: list[str] = []
    for i in range(1, 35):
        issue_rows.append(f"- line {i}, col 1: [undefined-name] undefined name 'x{i}' (pyflakes)")
        issue_rows.append(f"  line {i}: x{i} = y{i}")
    body = (
        "(no output)\n\n"
        "You just wrote `big.py` — it has problems. Syntax check (python) — 34 issue(s):\n"
        + "\n".join(issue_rows)
    )
    block = _report_block(body)
    collapsed = _plain(ChatView()._render_tool_result(block))
    assert_true("- line 3, col 1:" in collapsed, "head of the report shown collapsed")
    assert_true("- line 34, col 1:" not in collapsed, "tail capped collapsed")
    assert_true("48 more lines (Ctrl+o to expand)" in collapsed,
                f"expand hint counts the hidden rows: {collapsed}")

    block.expanded = True
    expanded = _plain(ChatView()._render_tool_result(block))
    assert_true("- line 34, col 1:" in expanded, "expanded shows the full report")
    assert_true("more lines (Ctrl+o to expand)" not in expanded, "no hint when expanded")


def test_post_write_all_unchanged_one_liner_keeps_status_header() -> None:
    body = (
        "(no output)\n\n"
        "You just wrote `calc.py` — it still has problems. Syntax check "
        "(python, lsp) — 9 issue(s), all unchanged and already reported above."
    )
    block = _report_block(body)
    rendered = ChatView()._render_tool_result(block)
    joined = " ".join(_plain(rendered).split())
    assert_true("all unchanged and already reported above." in joined, "one-liner shown")
    ansi_rows = [row for row in _ansi(rendered, width=400).splitlines() if "still has problems" in row]
    assert_true(len(ansi_rows) == 1, "one-liner found in ansi")
    assert_true(
        _fg_triplet(theme.FG_MUTED) in ansi_rows[0]
        and _fg_triplet(theme.ERROR_SOFT) in ansi_rows[0],
        f"muted intro + status-coloured remainder: {ansi_rows[0]!r}",
    )


def test_clean_write_confirmation_routes_status_style() -> None:
    # A clean confirmation only reaches the advisory lane behind a recognised
    # block (multi-file write); its OK portion must wear the status green, not
    # the flat dim note style.
    body = (
        "(no output)\n\n"
        "You just wrote `a.py` — it has problems. Syntax check (python) — 1 issue(s):\n"
        "- line 1, col 1: unexpected 'def oops(' (tree-sitter)\n\n"
        "Wrote `b.py` (12 lines). Syntax check (python, lsp): OK — no syntax errors.\n"
        "Warnings (not blocking) — 1:\n"
        "- line 2, col 1: [unused-import] os (pylint)\n\n"
        "(+1 more written file(s) not shown.)"
    )
    block = _report_block(body)
    rendered = ChatView()._render_tool_result(block)
    text = _plain(rendered)
    assert_true("Wrote `b.py` (12 lines)." in text, "clean intro shown")
    assert_true("Syntax check (python, lsp): OK" in text, "OK status shown")
    assert_true("(+1 more written file(s) not shown.)" in text, "trailing note kept")
    hits = {"intro": 0, "ok": 0}
    for row in _ansi(rendered, width=400).splitlines():
        if "Wrote `b.py`" in row:
            hits["intro"] += 1
            assert_true(_fg_triplet(theme.FG_MUTED) in row, f"clean intro muted: {row!r}")
        if ": OK — no syntax errors." in row:
            hits["ok"] += 1
            assert_true(_fg_triplet(theme.SUCCESS) in row, f"OK line status green: {row!r}")
    assert_true(hits == {"intro": 1, "ok": 1}, f"both styled checks fired: {hits}")


def test_autoread_embedded_section_echo_rows_render_pretty() -> None:
    # The auto-read's embedded Syntax check paragraph carries `edit id:` echo
    # rows; they must render like anchor rows (accent gutter label + code bg),
    # with the lexer picked from the intro's path.
    body = (
        "(no output)\n\n"
        "You just wrote `greet.py`. Its current line ids are below — edit it in "
        "place with edit; no separate beginTransaction needed:\n"
        "greet.py (ids: each line is '<line><hash>|<content>'):\n"
        "17b|def greet(:\n"
        "2a7|    return 1\n\n"
        "Syntax check (python) — 1 issue(s); fix each line below with edit, then it is re-checked:\n"
        "- line 1, col 1: unexpected 'def greet(:'\n"
        "  edit id: 17b|def greet(:"
    )
    block = _report_block(body)
    rendered = ChatView()._render_tool_result(block)
    text = _plain(rendered)
    assert_true("edit id: 17b|def greet(:" not in text, "raw echo pipe replaced")
    assert_true("edit id: 17b  def greet(:" in text, f"echo row wears the gutter form: {text}")
    echo_rows = 0
    for row in _ansi(rendered, width=400).splitlines():
        if "edit id: 17b" in row:
            echo_rows += 1
            assert_true(_bg_triplet(theme.CODE_BG) in row, f"echo row on the code bg: {row!r}")
            assert_true(_fg_triplet(theme.ACCENT) in row, f"echo gutter wears the accent: {row!r}")
    assert_true(echo_rows == 1, f"echo row found in ansi: {echo_rows}")


def _click_actions(renderable, width: int = 80) -> set[str]:
    """Every ``@click`` action carried by the rendered segments' style meta."""
    console = Console(width=width, file=StringIO())
    options = console.options.update_width(width)
    actions: set[str] = set()
    for seg in console.render(renderable, options):
        if seg.style is not None and seg.style.meta.get("@click"):
            actions.add(seg.style.meta["@click"])
    return actions


def test_terminal_window_height_contract() -> None:
    # Title bar + prompt/command row + one output row + footer = 4 rows, every
    # one newline-terminated so Textual's height measurement never crops.
    block = _shell_result_block()
    height = _textual_content_height(ChatView()._render_tool_result(block))
    assert_true(height == 4, f"window measures title+body+footer (got {height}, want 4)")

    block.collapsed = True
    height = _textual_content_height(ChatView()._render_tool_result(block))
    assert_true(height == 1, f"collapsed window is the title bar only (got {height}, want 1)")


def test_terminal_prompt_ps1_colors() -> None:
    rows = _command_rows(
        "echo hi", cursor=False, first_prompt=_prompt_text("meg", "~/qw35")
    )
    plain = _plain(_FullWidthLines(rows, wrap=True))
    assert_true("qwowl@meg:~/qw35$ echo hi" in plain, "PS1 prompt precedes the command")
    ansi = _row_ansi(rows, width=80)
    # user, @, host and path each wear their own color.
    assert_true(_fg_triplet(theme.SUCCESS) in ansi, "qwowl wears the success green")
    assert_true(_fg_triplet(theme.FG_MUTED) in ansi, "@ wears the muted grey")
    assert_true(_fg_triplet(theme.WARNING) in ansi, "host wears the warning color")
    assert_true(_fg_triplet(theme.ACCENT) in ansi, "path wears the accent color")


def test_window_title_is_neutral_not_green() -> None:
    # The title text must wear FG_MUTED on success (red stays for errors);
    # green belongs to the □ control and the prompt's user only.
    block = _shell_result_block()
    console = Console(width=80, file=StringIO())
    segs = list(console.render(ChatView()._render_tool_result(block), console.options))
    title_colors = {
        seg.style.color.get_truecolor() for seg in segs
        if seg.style and seg.style.color and "terminal #" in seg.text
    }
    h = theme.FG_MUTED.lstrip("#")
    want = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    assert_true(title_colors == {want}, f"title is muted (got {title_colors}, want {want})")

    block.is_error = True
    segs = list(console.render(ChatView()._render_tool_result(block), console.options))
    err_colors = {
        seg.style.color.get_truecolor() for seg in segs
        if seg.style and seg.style.color and "terminal #" in seg.text
    }
    e = theme.ERROR_SOFT.lstrip("#")
    want_err = tuple(int(e[i:i + 2], 16) for i in (0, 2, 4))
    assert_true(err_colors == {want_err}, f"failed window titles red (got {err_colors})")


def test_prompt_prefix_does_not_consume_reveal() -> None:
    # The prompt is chrome: reveal keeps slicing command chars only, so the
    # type-out shows exactly target[:reveal] regardless of the prompt length.
    cmd = "printf ok && echo done"
    prompt = _prompt_text("meg", "~")
    prefix = prompt.plain
    for reveal in range(1, len(cmd) + 1):
        rows = _command_rows(cmd, cursor=False, reveal=reveal, first_prompt=prompt)
        visible = rows[0].text.plain
        assert_true(
            visible == prefix + cmd[:reveal],
            f"prompt intact and prefix exact at reveal={reveal}: {visible!r}",
        )


def test_output_head_capped_with_marker() -> None:
    output = "\n".join(f"line {i}" for i in range(1, 31))
    block = _shell_result_block(output=output)
    text = _plain(ChatView()._render_tool_result(block))
    # Budget 10 minus the command row leaves 9 output rows (HEAD view).
    assert_true("line 1" in text and "line 9" in text, "head of the output shown")
    assert_true("line 10" not in text, "output beyond the window budget hidden")
    assert_true("... +21 lines" in text, "marker counts the hidden lines")

    block.expanded = True
    expanded = _plain(ChatView()._render_tool_result(block))
    assert_true("line 30" in expanded, "expanded shows the full output")
    assert_true("+21 lines" not in expanded, "no marker when expanded")


def test_head_capped_rows_command_takes_priority() -> None:
    cmd_rows = [_line_with_bg(Text(f"c{i}"), "#111318") for i in range(4)]
    out_rows = [_line_with_bg(Text(f"o{i}"), "#111318") for i in range(20)]
    rows, hidden = _head_capped_rows(cmd_rows, out_rows, budget=10)
    assert_true(len(rows) == 10 and hidden == 14, "command rows all kept, output fills the rest")
    rows, hidden = _head_capped_rows(cmd_rows, out_rows[:6], budget=10)
    assert_true(len(rows) == 10 and hidden == 0, "no marker when everything fits")


def test_window_controls_carry_click_meta() -> None:
    view = ChatView()
    block = _shell_result_block()
    actions = _click_actions(view._render_tool_result(block))
    assert_true(
        actions == {"win_collapse", "win_expand", "win_close", "win_copy"},
        f"result chrome exposes all four controls (got {actions})",
    )
    pending = ToolBlock("bash")
    pending.args_buf = '{"command":"date"}'
    live = _click_actions(view._render_tool_call(pending))
    assert_true(
        {"win_collapse", "win_expand", "win_close", "win_copy"} <= live,
        f"streaming chrome already clickable (got {live})",
    )


def test_window_controls_traffic_light_colors() -> None:
    ansi = _ansi(_window_title_row("terminal #79b1", theme.SUCCESS), width=60)
    assert_true(_fg_triplet(theme.WARNING) in ansi, "`-` wears the theme warning color")
    assert_true(_fg_triplet(theme.SUCCESS) in ansi, "`□` wears the theme success color")
    assert_true(_fg_triplet(theme.ERROR) in ansi, "`x` wears the theme error color")
    # Textual repaints @click segments with the widget's link-color unless
    # auto_links is off — with it on, the traffic lights above would flatten
    # to one color on screen even though this renderable-level test passes.
    assert_true(ToolBlock("bash").auto_links is False, "link-color overlay disabled")


def test_terminal_body_bg_darker_than_chat() -> None:
    # The body's derived background must differ from (and sit below) the main
    # chat background so the window reads as its own surface.
    body = _term_bg().lstrip("#")
    base = theme.BG_BASE.lstrip("#")
    body_rgb = [int(body[i:i + 2], 16) for i in (0, 2, 4)]
    base_rgb = [int(base[i:i + 2], 16) for i in (0, 2, 4)]
    assert_true(body != base, "terminal bg differs from the chat bg")
    assert_true(all(a <= b for a, b in zip(body_rgb, base_rgb)), "terminal bg is darker")
    block = _shell_result_block()
    ansi = _ansi(ChatView()._render_tool_result(block), width=80)
    assert_true(f"48;2;{body_rgb[0]};{body_rgb[1]};{body_rgb[2]}" in ansi,
                "result body painted on the terminal bg")


def test_footer_shows_time_and_token_estimate() -> None:
    block = _shell_result_block(output="hello\n")
    text = _plain(ChatView()._render_tool_result(block))
    assert_true(block.started_at in text, "footer shows the start time")
    expected = rough_token_count("date" + block.full_result)
    assert_true(f"~{expected} tok" in text, "footer shows the chars/4 token estimate")
    # While streaming the estimate is unknowable: time only.
    pending = ToolBlock("bash")
    pending.args_buf = '{"command":"date"}'
    live = _plain(ChatView()._render_tool_call(pending))
    assert_true(pending.started_at in live and " tok" not in live,
                "streaming footer shows time but no token guess")


def test_crlf_command_reveal_matches_prefix() -> None:
    # splitlines() treated \r\n as one boundary while the reveal charged one
    # char, desyncing the shown prefix; split("\n") keeps them in lockstep.
    # The lexer normalizes the (invisible) \r out of the rendered rows, so
    # compare with \r stripped from line ends — what matters is that rows and
    # boundaries stay in lockstep with target[:reveal].
    cmd = "echo a\r\necho b\r\necho c"
    for reveal in range(1, len(cmd) + 1):
        rows = _command_rows(cmd, cursor=False, reveal=reveal)
        visible = "\n".join(row.text.plain[2:] for row in rows)
        expected = [line.rstrip("\r") for line in cmd[:reveal].split("\n")]
        if expected and expected[-1] == "":
            expected = expected[:-1]
        assert_true(
            visible == "\n".join(expected),
            f"prefix matches at reveal={reveal}: {visible!r}",
        )


def main() -> None:
    test_bash_result_height_includes_output_row()
    test_bash_result_keeps_command_visible()
    test_bash_call_recovers_xml_command_for_terminal_preview()
    test_bash_call_uses_hidden_bash_syntax_pass()
    test_bash_reveal_plain_tracks_prefix()
    test_bash_reveal_highlights_full_command_not_prefix()
    test_bash_reveal_colours_are_stable_across_steps()
    test_long_multiline_write_types_out_line_by_line()
    test_long_multiline_write_reveal_matches_prefix()
    test_long_write_full_render_shows_all_lines()
    test_bash_result_recovers_malformed_quoted_command()
    test_bash_result_trims_dangling_recovered_quote()
    test_bash_result_recovers_command_equals_typo()
    test_split_bash_advisories()
    test_advisory_segments_classifies()
    test_advisory_classifies_post_write_report_shapes()
    test_post_write_report_renders_pretty()
    test_post_write_report_respects_preview_cap()
    test_post_write_all_unchanged_one_liner_keeps_status_header()
    test_clean_write_confirmation_routes_status_style()
    test_autoread_embedded_section_echo_rows_render_pretty()
    test_bash_post_write_anchors_render_as_file_view()
    test_render_is_display_only_and_non_mutating()
    test_terminal_window_height_contract()
    test_terminal_prompt_ps1_colors()
    test_window_title_is_neutral_not_green()
    test_prompt_prefix_does_not_consume_reveal()
    test_output_head_capped_with_marker()
    test_head_capped_rows_command_takes_priority()
    test_window_controls_carry_click_meta()
    test_window_controls_traffic_light_colors()
    test_terminal_body_bg_darker_than_chat()
    test_footer_shows_time_and_token_estimate()
    test_crlf_command_reveal_matches_prefix()
    print("shell renderer tests passed")


if __name__ == "__main__":
    main()
