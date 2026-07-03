"""Tests for tool-call rendering helpers."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from widgets.chat_log import ChatView, ToolBlock  # noqa: E402
from widgets.chat_log import _command_rows, _FullWidthLines, _line_with_bg  # noqa: E402
from widgets.chat_log import _advisory_segments, _split_bash_advisories  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def _plain(renderable) -> str:
    console = Console(width=100, record=True, file=StringIO())
    console.print(renderable)
    return console.export_text(styles=False)


def _ansi(renderable, width: int = 20) -> str:
    console = Console(
        width=width,
        record=True,
        force_terminal=True,
        color_system="truecolor",
        file=StringIO(),
    )
    console.print(renderable)
    return console.export_text(styles=True)


def test_block_rows_pad_background_to_width() -> None:
    ansi = _ansi(_FullWidthLines([_line_with_bg(Text("x"), "#111318")]), width=10)

    assert_true("\x1b[48;2;17;19;24m         " in ansi, "background padding emitted")


def _textual_content_height(renderable, width: int = 80) -> int:
    """Mimic Textual's ``RichVisual.get_height`` (textual/visual.py): the height it
    assigns an auto-height widget by COUNTING '\\n' in the rendered segments, not by
    counting logical lines. It then crops the widget's strips to that height. A box
    whose final row lacked a trailing newline measured one row short, so Textual
    cropped the last line — the bash output, a tool-call arg preview, or the final
    line of the approval command — and painted a blank in its place.
    """
    console = Console(width=width, file=StringIO())
    options = console.options.update_width(width).update(highlight=False)
    return sum(seg.text.count("\n") for seg in console.render(renderable, options))


def test_fullwidth_wrap_height_counts_every_row() -> None:
    # Regression (output-not-displayed bug): every wrapped row must terminate with a
    # newline, including the last, so Textual's newline-based height measurement
    # matches the real row count and never crops the final row.
    rows = [_line_with_bg(Text(f"row {i}"), "#111318") for i in range(4)]
    height = _textual_content_height(_FullWidthLines(rows, wrap=True))
    assert_true(height == 4, f"every wrapped row measured (got {height}, want 4)")


def test_bash_result_height_includes_output_row() -> None:
    # The concrete symptom: `$ date` and its output were measured as 2 rows, so
    # Textual cropped the output line to a blank. Badge + command + output = 3.
    block = ToolBlock("bash")
    block.args_buf = '{"command":"date"}'
    block.full_result = "Wed Jul  1 21:11:27 CEST 2026\n"
    height = _textual_content_height(ChatView()._render_tool_result(block))
    assert_true(height >= 3, f"result measured tall enough to show output (got {height})")


def test_tool_call_detail_height_includes_detail_row() -> None:
    # Same bug for non-bash calls: the badge + the single arg-preview row must both
    # be measured, or the detail line is cropped away ("with other commands").
    block = ToolBlock("beginTransaction")
    block.args_buf = '{"file":"example.py"}'
    height = _textual_content_height(ChatView()._render_tool_call(block))
    assert_true(height >= 2, f"call detail measured tall enough to show (got {height})")


def test_fullwidth_wrap_shows_all_lines_under_height_constraint() -> None:
    # Regression: the approval modal lives in an auto-height container, so the
    # render options carry a `height`. _FullWidthLines(wrap=True) must render each
    # logical line at its natural height — not pad every line to the container
    # height, which previously collapsed all but the first line into blanks.
    rows = [_line_with_bg(Text(f"line {i}"), "#111318") for i in range(5)]
    console = Console(width=30, file=StringIO())
    options = console.options.update(height=12)  # container imposes a height
    rendered = console.render_lines(_FullWidthLines(rows, wrap=True), options)
    joined = "\n".join("".join(seg.text for seg in row) for row in rendered)
    for i in range(5):
        assert_true(f"line {i}" in joined, f"line {i} visible under height constraint")


def test_bash_result_keeps_command_visible() -> None:
    block = ToolBlock("bash")
    block.args_buf = '{"command":"python3 -m py_compile qwowl35/widgets/chat_log.py"}'
    block.full_result = "ok\n"

    text = _plain(ChatView()._render_tool_result(block))

    assert_true(">_" in text, "bash badge shown")
    assert_true("$ python3 -m py_compile qwowl35/widgets/chat_log.py" in text, "command shown")
    # Output sits in the same mini-terminal box as the command, distinguished by
    # color only — no bold "Output" label anymore.
    assert_true("ok" in text, "output shown")


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

    assert_true("38;2;230;219;116;48;2;21;23;28m" in ansi, "bash string syntax color emitted")
    assert_true("48;2;21;23;28m" in ansi, "soft shell background emitted")
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
    from widgets.chat_log import _REVEAL_MAX_STEP, _REVEAL_MIN_STEP

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


def test_tool_call_detail_pads_background_to_width() -> None:
    block = ToolBlock("beginTransaction")
    block.args_buf = '{"file":"example.py"}'

    ansi = _ansi(ChatView()._render_tool_call(block), width=40)

    assert_true("<>  example.py" in _plain(ChatView()._render_tool_call(block)), "badge + path shown")
    assert_true("48;2;21;23;28m" in ansi, "detail background emitted")


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

    assert_true("<>  example.py" in text, "read badge includes path")
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
    assert_true(">_" in text and ">_" not in source, "bash badge is render-only")
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
    assert_true("<>" in rf_text and "<>" not in rf_source, "read badge is render-only")
    assert_true("17b|def f():" in rf_source and "17b|def f():" not in rf_text,
                "read source keeps the pipe; render uses the gutter")


def main() -> None:
    test_block_rows_pad_background_to_width()
    test_fullwidth_wrap_shows_all_lines_under_height_constraint()
    test_fullwidth_wrap_height_counts_every_row()
    test_bash_result_height_includes_output_row()
    test_tool_call_detail_height_includes_detail_row()
    test_split_bash_advisories()
    test_advisory_segments_classifies()
    test_bash_post_write_anchors_render_as_file_view()
    test_read_shows_syntax_status()
    test_render_is_display_only_and_non_mutating()
    test_bash_result_keeps_command_visible()
    test_bash_call_recovers_xml_command_for_terminal_preview()
    test_bash_call_uses_hidden_bash_syntax_pass()
    test_bash_reveal_plain_tracks_prefix()
    test_bash_reveal_highlights_full_command_not_prefix()
    test_bash_reveal_colours_are_stable_across_steps()
    test_bash_result_recovers_malformed_quoted_command()
    test_bash_result_trims_dangling_recovered_quote()
    test_bash_result_recovers_command_equals_typo()
    test_tool_call_detail_pads_background_to_width()
    test_read_result_keeps_anchors()
    test_edit_result_shows_diff_before_refreshed_labels()
    print("chat log tests passed")


if __name__ == "__main__":
    main()


def test_raw_mode_streaming_call_shows_growing_xml() -> None:
    # Before the function is recognized (empty tool name) the box shows the raw
    # tool-call XML itself, grown to the reveal cursor.
    block = ToolBlock("")
    block.args_buf = "\n<function=bash>\n<parameter=command>\necho hi\n"
    block.reveal = len("\n<function=bash>\n<parameter=")
    text = _plain(ChatView()._render_tool_call(block))
    assert_true("<function=bash>" in text, "raw XML header shown")
    assert_true("<parameter=" in text, "raw XML grows to the reveal cursor")
    assert_true("echo hi" not in text, "unrevealed tail stays hidden")


def test_call_target_raw_mode_is_untruncated() -> None:
    # The generic detail preview truncates at 240 chars; raw mode must not,
    # or a long heredoc's XML would stop growing mid-stream.
    block = ToolBlock("")
    block.args_buf = "\n<function=bash>\n<parameter=command>\n" + ("x" * 500)
    target = ChatView()._call_target(block)
    assert_true(target == block.args_buf, "raw target is the full XML buffer")


def test_name_tool_call_switches_to_bash_box_and_clamps_reveal() -> None:
    view = ChatView()
    block = ToolBlock("")
    view._tool_blocks[0] = block
    block.args_buf = "\n<function=bash>\n<parameter=command>\n"
    block.reveal = len(block.args_buf)  # deep into the raw XML view
    view.name_tool_call(0, "bash")
    assert_true(block.tool_name == "bash", "name applied")
    target = view._call_target(block)
    assert_true(block.reveal <= len(target), "reveal clamped into the new target")


def test_named_streaming_call_grows_command_from_partial_xml() -> None:
    # After recognition, fragments keep arriving; the bash box must extract the
    # growing command from the partial XML buffer.
    view = ChatView()
    block = ToolBlock("bash")
    view._tool_blocks[0] = block
    block.args_buf = "\n<function=bash>\n<parameter=command>\ncat <<'EOF' > f.py\nprint('a')\n"
    target = view._call_target(block)
    assert_true(target.startswith("cat <<'EOF' > f.py"), f"command extracted: {target!r}")
    assert_true("print('a')" in target, "later command lines included")
    assert_true("<parameter" not in target, "XML scaffolding stripped")


def test_finalize_tool_call_replaces_args_and_marks_done() -> None:
    view = ChatView()
    block = ToolBlock("bash")
    view._tool_blocks[0] = block
    block.args_buf = "\n<function=bash>\n<parameter=command>\necho hi\n</parameter>\n</function>\n"
    view.finalize_tool_call(0, '{"command":"echo hi"}')
    assert_true(block.stream_done, "stream_done set")
    assert_true(view._call_target(block) == "echo hi", "final JSON drives the command")


def test_demote_tool_call_removes_block() -> None:
    view = ChatView()
    block = ToolBlock("")
    removed = []
    block.remove = lambda: removed.append(True)  # type: ignore[method-assign]
    view._tool_blocks[0] = block
    view.demote_tool_call(0)
    assert_true(0 not in view._tool_blocks, "block dropped from in-flight map")
    assert_true(removed == [True], "widget removed from the log")


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
