"""Tests for the ChatView transcript: streaming lifecycle and the
per-tool render dispatch."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from widgets.chat import ChatView, ToolBlock  # noqa: E402
from widgets.chat.chat_view import _tool_badge  # noqa: E402
from widgets.chat.tool_args import _compact_args  # noqa: E402

import mascot  # noqa: E402

from chat_test_helpers import (  # noqa: E402
    _ansi,
    _plain,
    _shell_result_block,
    _textual_content_height,
    assert_true,
)


def test_tool_call_detail_height_includes_detail_row() -> None:
    # Same bug for non-bash calls: the badge + the single arg-preview row must both
    # be measured, or the detail line is cropped away ("with other commands").
    block = ToolBlock("beginTransaction")
    block.args_buf = '{"file":"example.py"}'
    height = _textual_content_height(ChatView()._render_tool_call(block))
    assert_true(height >= 2, f"call detail measured tall enough to show (got {height})")


def test_tool_call_detail_pads_background_to_width() -> None:
    block = ToolBlock("beginTransaction")
    block.args_buf = '{"file":"example.py"}'

    ansi = _ansi(ChatView()._render_tool_call(block), width=40)

    assert_true("Read  example.py" in _plain(ChatView()._render_tool_call(block)), "badge + path shown")
    assert_true("48;2;21;23;28m" in ansi, "detail background emitted")


def test_explore_and_resume_calls_stream_their_text_live() -> None:
    # The planner's `explore` task and the explorer's `resume` summary type
    # out live instead of raw XML until the parameter closes.
    view = ChatView()
    block = ToolBlock("explore")
    block.args_buf = (
        "\n<function=explore>\n<parameter=task>\nMap the theme registry and how"
    )
    target = view._call_target(block)
    assert_true(
        target == "Map the theme registry and how",
        f"reveal target is the task: {target!r}",
    )
    block.reveal = len(target)
    text = _plain(view._render_tool_call(block))
    assert_true("Map the theme registry" in text, f"task streams: {text}")
    assert_true("<parameter" not in text, f"no raw XML noise: {text}")
    assert_true("Spawn" in text and "Explorer" in text, f"spawn card chrome: {text}")

    block = ToolBlock("resume")
    block.args_buf = '{"summary":"theme/__init__.py owns the palette"}'
    target = view._call_target(block)
    assert_true(
        target == "theme/__init__.py owns the palette",
        f"reveal target is the summary: {target!r}",
    )
    block.reveal = len(target)
    text = _plain(view._render_tool_call(block))
    assert_true("owns the palette" in text, f"summary streams: {text}")
    assert_true("Resume" in text and "Explorer" in text, f"resume card chrome: {text}")


def test_edit_spawn_renders_card_and_targets_instructions() -> None:
    # The freestyle delegator's `edit` spawns the Editor: the instructions
    # type out inside a Spawn card with an Editor chip, above them the
    # pre-edit slice the Editor will receive (deterministic tmp file — a repo
    # filename would capture whatever is on disk).
    import tempfile

    view = ChatView()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "queue.py")
        Path(path).write_text("alpha = 1\nbeta = 2\ngamma = 3\n")
        block = ToolBlock("edit")
        block.args_buf = (
            f"\n<function=edit>\n<parameter=filename>\n{path}\n</parameter>\n"
            "<parameter=line_ranges>\n2-3\n</parameter>\n"
            "<parameter=instructions>\nWiden the queue panel"
        )
        target = view._call_target(block)
        assert_true(target == "Widen the queue panel", f"target is the instructions: {target!r}")
        block.reveal = len(target)
        text = _plain(view._render_tool_call(block))
    import re

    assert_true("Spawn" in text and "Editor" in text, f"editor spawn card: {text}")
    assert_true("queue.py" in text and "2-3" in text, f"file header shown: {text}")
    assert_true(re.search(r"2[0-9a-f]{2}  beta = 2", text),
                f"input slice rendered as hashline id rows: {text}")
    assert_true("alpha" not in text, f"only the named range shown: {text}")
    assert_true("Widen the queue panel" in text, f"instructions inside the card: {text}")


def test_hashline_edit_is_not_a_spawn() -> None:
    # The Editor's own `edit` tool shares the delegator's name but carries
    # file/content args — it must keep the streaming-code rendering.
    view = ChatView()
    block = ToolBlock("edit")
    block.args_buf = '{"file":"calc.py","content":"def add():\\n    pass"}'
    target = view._call_target(block)
    assert_true(target.startswith("def add()"), f"target stays the content: {target!r}")
    block.reveal = len(target)
    text = _plain(view._render_tool_call(block))
    assert_true("Spawn" not in text, f"no spawn card for a hashline edit: {text}")
    assert_true(block.spawn_snippet is None, "content-shaped edit never captures a slice")


def test_toggle_tools_repaints_pending_blocks() -> None:
    # Ctrl+o must reach still-streaming cards: toggle_tools marks pending
    # blocks dirty so the persistent tick repaints them with the new flag.
    view = ChatView()
    block = ToolBlock("edit")
    view._tool_blocks[0] = block
    block.args_dirty = False
    view.toggle_tools()
    assert_true(view.tools_expanded, "view-wide flag flipped")
    assert_true(block.args_dirty, "pending block marked for repaint")


def test_explore_result_suppresses_duplicate_findings() -> None:
    # The findings already streamed in the Resume card; the explore result
    # collapses to the Spawn card plus a one-line pointer, with the raw
    # findings still a Ctrl+o away.
    view = ChatView()
    block = ToolBlock("explore")
    block.args_buf = '{"task":"Map the theme registry"}'
    block.full_result = "Exploration findings:\ntheme/__init__.py owns the palette"
    text = _plain(view._render_tool_result(block))
    assert_true("Spawn" in text and "Explorer" in text, f"spawn card kept: {text}")
    assert_true("Map the theme registry" in text, f"task inside the card: {text}")
    assert_true("Resume card above" in text, f"one-line pointer shown: {text}")
    assert_true("owns the palette" not in text, f"findings not duplicated: {text}")
    assert_true("Result" not in text, f"no generic Result box: {text}")

    view.tools_expanded = True
    text = _plain(view._render_tool_result(block))
    assert_true("owns the palette" in text, f"Ctrl+o reveals the findings: {text}")


def test_explore_result_fallbacks_stay_visible() -> None:
    # Bodies without a Resume-card twin (no resume called, hard errors) keep
    # the generic Result box — suppressing them would destroy information.
    view = ChatView()
    block = ToolBlock("explore")
    block.args_buf = '{"task":"Map the theme registry"}'
    block.full_result = "The explorer finished without calling resume."
    text = _plain(view._render_tool_result(block))
    assert_true(
        "Result" not in text and "without calling resume" in text,
        f"fallback body stays visible, without the Result label: {text}",
    )

    block.is_error = True
    block.full_result = "Error: the explorer could not run"
    text = _plain(view._render_tool_result(block))
    assert_true("could not run" in text, f"error body stays visible: {text}")


def test_explore_findings_result_with_no_streamed_call_is_swallowed() -> None:
    # Index collision: the explorer sub-agent's own calls (same ChatView,
    # per-turn indices) orphan the parent's streamed `explore` block, so the
    # parent's findings-result arrives with no block to attach to. It must be
    # swallowed — a fabricated block could only draw an empty Spawn card, and
    # the findings already streamed in the Resume card.
    view = ChatView()
    appended: list = []
    view._append = appended.append  # type: ignore[method-assign]

    view.add_tool_result(0, "explore", "Exploration findings:\npalette lives in theme/")
    assert_true(appended == [], "no widget fabricated for orphaned findings")
    assert_true(view._tool_blocks == {}, "no block registered for orphaned findings")

    # Error and no-resume bodies have no Resume-card twin: the defensive
    # creation must keep them visible.
    view.add_tool_result(1, "explore", "Error: the explorer could not run", is_error=True)
    assert_true(len(appended) == 1, "error result still gets a widget")
    view.add_tool_result(2, "explore", "The explorer finished without calling `resume`; ...")
    assert_true(len(appended) == 2, "no-resume fallback still gets a widget")


def test_resume_result_renders_markdown_card() -> None:
    view = ChatView()
    block = ToolBlock("resume")
    block.args_buf = '{"summary":"# Findings\\n\\n- palette lives in `theme`"}'
    block.full_result = "Findings recorded; exploration complete."
    text = _plain(view._render_tool_result(block))
    assert_true("Resume" in text and "Explorer" in text, f"resume card chrome: {text}")
    assert_true(
        "Findings" in text and "# Findings" not in text,
        f"summary rendered as markdown, not raw: {text}",
    )
    assert_true(
        "Findings recorded" not in text,
        f"generic confirmation box dropped: {text}",
    )


def test_editor_result_with_no_streamed_call_renders_card() -> None:
    # Index collision: the editor sub-agent's own tool calls (same ChatView,
    # per-turn indices) reuse the delegator `edit` slot, so the parent's
    # result arrives on a fabricated block with no args. It must still render
    # as an "Editor result" card (the Spawn card streamed above), not a plain
    # "Edit" badge + text fallback.
    view = ChatView()
    block = ToolBlock("edit")          # empty args_buf == the collision case
    block.full_result = (
        "Editor result for /proj/flappy_bird.py (2 edits applied):\n"
        "Fixed the collision check to use Bbox attributes.\n\n"
        "Diff:\n@@ -1,2 +1,2 @@\n-old = 1\n+new = 1\n"
    )
    text = _plain(view._render_tool_result(block))
    assert_true("Editor result" in text and "Editor" in text,
                f"editor result card chrome: {text}")
    assert_true("flappy_bird.py" in text, f"filename shown: {text}")
    assert_true("Result\n" not in text, f"not the generic Result box: {text}")


def test_attention_flagged_edit_result_carves_syntax_section_from_diff() -> None:
    # The live screenshot shape: an attention-flagged (is_error) edit whose
    # result has a Diff: block but NO `Current <path> (ids, …)` snippet. The
    # trailing Syntax check section must render as a status block after the
    # diff — never swallowed into the diff text as removal rows with gutter
    # numbers — while the structured diff view (not the plain error fallback)
    # is kept.
    import re

    block = ToolBlock("edit")
    block.args_buf = '{"file":"app.py","id":"14d"}'
    block.is_error = True
    block.full_result = (
        "Edited line 5 in app.py.\n"
        "Diff:\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -5,1 +5,1 @@\n"
        "-x = foo()\n"
        "+x = bar()\n"
        "\n"
        "Syntax check (python, lsp) — 1 issue(s); fix each line below with edit, "
        "then it is re-checked:\n"
        "- line 5, col 1: [no-member] Module 'm' has no 'bar' member"
    )
    text = _plain(ChatView()._render_tool_result(block))
    assert_true("- x = foo()" in text and "+ x = bar()" in text, f"diff rows kept: {text}")
    assert_true("Syntax check (python, lsp)" in text, f"syntax status present: {text}")
    assert_true("- line 5, col 1:" in text, f"issue bullet present: {text}")
    assert_true(
        re.search(r"\d+ +- line 5", text) is None,
        f"no diff gutter number before the issue bullet: {text}",
    )
    assert_true("Result" not in text, f"structured view kept, no error fallback: {text}")


def test_every_tool_family_has_a_distinct_badge() -> None:
    # Every tool the model can call should get a real badge, not the generic
    # "?" fallback — a newly-added tool that skips _TOOL_BADGES regresses to
    # an anonymous box, which this guards against.
    for name in mascot._TOOL_STATES:
        badge = _tool_badge(name)
        assert_true(badge != "?", f"{name} has a non-generic badge (got {badge!r})")
        assert_true(badge.isascii(), f"{name} badge is ASCII (got {badge!r})")
    assert_true(_tool_badge("totally_unknown_tool") == "?", "unknown tool falls back to ?")


def test_compact_args_surfaces_new_tool_subjects() -> None:
    # The type-out reveal for a non-shell call previews its key arg; the newer
    # tools' args (url/path/query/question) must be recognized, not dropped.
    assert_true("https://x.dev" in _compact_args({"url": "https://x.dev"}), "url surfaced")
    assert_true("src/app" in _compact_args({"path": "src/app"}), "path surfaced")
    assert_true("needle" in _compact_args({"query": "needle"}), "query surfaced")
    assert_true("why?" in _compact_args({"question": "why?"}), "question surfaced")


def test_web_fetch_header_shows_url() -> None:
    block = ToolBlock("web_fetch")
    block.args_buf = '{"url": "https://example.dev/page"}'
    text = _plain(ChatView()._render_tool_call(block))
    assert_true("Fetch" in text, "web_fetch badge shown")
    assert_true("https://example.dev/page" in text, "web_fetch header shows url")


def test_web_fetch_result_shows_short_preview() -> None:
    # A fetched page is truncated hard in the transcript: a size summary + only
    # a few lines collapsed, the rest a Ctrl+o expand away (full text below).
    body = "\n".join(f"line {i}" for i in range(1, 31))
    block = ToolBlock("web_fetch")
    block.args_buf = '{"url": "https://example.dev/docs"}'
    block.full_result = body
    collapsed = _plain(ChatView()._render_tool_result(block))
    assert_true("Fetched 30 lines" in collapsed, f"size summary shown: {collapsed}")
    assert_true("line 5" in collapsed and "line 6" not in collapsed, "only 5 lines collapsed")
    assert_true("25 more lines (Ctrl+o to expand)" in collapsed, "expand hint shown")
    block.expanded = True
    assert_true("line 30" in _plain(ChatView()._render_tool_result(block)), "expands to full page")


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


def test_close_tool_block_bookkeeping() -> None:
    view = ChatView()
    block = ToolBlock("bash")
    block.args_buf = '{"command":"sleep 5"}'
    removed = []
    block.remove = lambda: removed.append(True)  # type: ignore[method-assign]
    view._tool_blocks[0] = block

    view.close_tool_block(block)
    assert_true(0 not in view._tool_blocks, "closed block leaves the in-flight map")
    assert_true(removed == [True], "widget removed from the log")
    assert_true(0 in view._closed_indices, "in-flight close tombstones its index")

    # Late traffic for the closed index must not resurrect the window.
    view.update_tool_call(0, "<parameter=command>more")
    assert_true(0 not in view._tool_blocks, "late args fragment ignored")
    view.add_tool_result(0, "bash", "done\n")
    assert_true(0 not in view._tool_blocks, "late result ignored")
    assert_true(0 not in view._closed_indices, "tombstone consumed by the result")

    # A finished (collapsible) window closes without a tombstone.
    done = _shell_result_block()
    done_removed = []
    done.remove = lambda: done_removed.append(True)  # type: ignore[method-assign]
    view._collapsibles.append(done)
    view.close_tool_block(done)
    assert_true(done not in view._collapsibles, "closed block leaves the collapsibles")
    assert_true(done_removed == [True] and not view._closed_indices,
                "finished close removes without a tombstone")


def main() -> None:
    test_tool_call_detail_height_includes_detail_row()
    test_tool_call_detail_pads_background_to_width()
    test_explore_and_resume_calls_stream_their_text_live()
    test_edit_spawn_renders_card_and_targets_instructions()
    test_hashline_edit_is_not_a_spawn()
    test_toggle_tools_repaints_pending_blocks()
    test_explore_result_suppresses_duplicate_findings()
    test_explore_result_fallbacks_stay_visible()
    test_explore_findings_result_with_no_streamed_call_is_swallowed()
    test_resume_result_renders_markdown_card()
    test_editor_result_with_no_streamed_call_renders_card()
    test_attention_flagged_edit_result_carves_syntax_section_from_diff()
    test_every_tool_family_has_a_distinct_badge()
    test_compact_args_surfaces_new_tool_subjects()
    test_web_fetch_header_shows_url()
    test_web_fetch_result_shows_short_preview()
    test_raw_mode_streaming_call_shows_growing_xml()
    test_call_target_raw_mode_is_untruncated()
    test_name_tool_call_switches_to_bash_box_and_clamps_reveal()
    test_named_streaming_call_grows_command_from_partial_xml()
    test_finalize_tool_call_replaces_args_and_marks_done()
    test_demote_tool_call_removes_block()
    test_close_tool_block_bookkeeping()
    print("chat view tests passed")


if __name__ == "__main__":
    main()
