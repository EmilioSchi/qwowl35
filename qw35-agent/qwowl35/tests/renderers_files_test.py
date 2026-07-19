"""Tests for the ls/glob/grep result cards."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from widgets.chat import ChatView, ToolBlock  # noqa: E402

from chat_test_helpers import _plain, assert_true  # noqa: E402


def test_parse_ls_result_rows_and_fallbacks() -> None:
    from widgets.chat.renderers.files import _parse_ls_result

    parsed = _parse_ls_result(
        "Listed 4 item(s) in /repo:\n"
        "---\n"
        "[DIR] docs\n"
        "[DIR] src\n"
        "README.md\n"
        "Cargo.toml"
    )
    assert_true(
        parsed
        == ("/repo", 4, [("docs", True), ("src", True), ("README.md", False), ("Cargo.toml", False)], 0),
        f"dirs and files parsed in order: {parsed}",
    )

    # run_ls's own >MAX_ENTRIES truncation tail is captured, not parsed as a row.
    truncated = _parse_ls_result(
        "Listed 250 item(s) in /big:\n---\nonly.txt\n\n(249 more entries not shown)"
    )
    assert_true(
        truncated == ("/big", 250, [("only.txt", False)], 249),
        f"server truncation tail captured: {truncated}",
    )

    # An empty directory parses to zero entries — rendered as the same card.
    assert_true(_parse_ls_result("Directory /empty is empty.") == ("/empty", 0, [], 0),
                "empty-dir message parses to a zero-entry tree")

    # Anything unexpected falls back to the plain result box.
    assert_true(_parse_ls_result("Error: Directory does not exist: /nope") is None,
                "error text is not a tree")
    assert_true(_parse_ls_result("Listed 1 item(s) in /x:\nno separator") is None,
                "missing --- separator rejected")


def test_ls_tree_text_header_guides_and_icons() -> None:
    from widgets.chat.renderers.files import _ls_tree_text

    tree = _ls_tree_text(
        "/repo", 4, [("docs", True), ("src", True), ("README.md", False), ("Cargo.toml", False)],
        0, expanded=False,
    )
    plain = _plain(tree)
    assert_true("/repo (4)" in plain, f"header shows path and count: {plain}")
    assert_true("docs/" in plain and "src/" in plain, "dir rows get a trailing slash")
    lines = plain.strip().splitlines()
    assert_true(all(" ├─ " in line for line in lines[1:-1]), f"middle rows use ├─: {plain}")
    assert_true(" └─ " in lines[-1] and "Cargo.toml" in lines[-1], f"last row uses └─: {plain}")
    # Nerd Font icons: folder glyph on dir rows, per-extension glyphs on files.
    assert_true("" in lines[1], "folder icon on dir row")
    assert_true("" in plain, "markdown icon on README.md")
    assert_true("" in plain, "gear icon on Cargo.toml")


def test_ls_tree_text_collapse_cap_and_server_truncation() -> None:
    from config import TOOL_PREVIEW_LINES
    from widgets.chat.renderers.files import _ls_tree_text

    entries = [(f"file{i}.txt", False) for i in range(TOOL_PREVIEW_LINES + 10)]
    collapsed = _plain(_ls_tree_text("/big", len(entries), entries, 0, expanded=False))
    assert_true("... 10 more lines (Ctrl+o to expand)" in collapsed,
                f"collapsed cap with Ctrl+o hint: {collapsed}")
    assert_true(f"file{TOOL_PREVIEW_LINES}.txt" not in collapsed, "rows beyond the cap hidden")
    assert_true(" └─ " not in collapsed, "no closing guide while rows are hidden")

    expanded = _plain(_ls_tree_text("/big", len(entries), entries, 0, expanded=True))
    assert_true(f"file{TOOL_PREVIEW_LINES + 9}.txt" in expanded, "expanded shows every row")
    assert_true("more lines" not in expanded, "no hint when fully shown")

    served = _plain(_ls_tree_text("/big", 250, [("only.txt", False)], 249, expanded=True))
    assert_true("(249 more entries not shown)" in served, f"server tail rendered dim note: {served}")


def test_ls_result_renders_tree_and_error_falls_back() -> None:
    block = ToolBlock("list_directory")
    block.args_buf = '{"path": "/repo"}'
    block.full_result = "Listed 2 item(s) in /repo:\n---\n[DIR] src\nREADME.md"
    source = block.full_result
    text = _plain(ChatView()._render_tool_result(block))
    assert_true("/repo (2)" in text, f"tree header rendered: {text}")
    assert_true("List" not in text.splitlines()[0], "redundant badge title row dropped")
    assert_true("src/" in text and "README.md" in text, "entries rendered")
    assert_true("Result" not in text, "generic Result label replaced by the tree")
    assert_true(block.full_result == source, "render did not mutate full_result")

    err = ToolBlock("list_directory")
    err.args_buf = '{"path": "/nope"}'
    err.full_result = "Error: Directory does not exist: /nope"
    err.is_error = True
    err_text = _plain(ChatView()._render_tool_result(err))
    assert_true("Result" in err_text and "Error: Directory does not exist" in err_text,
                "error keeps the plain fallback box")


def test_glob_result_renders_tree_card() -> None:
    from widgets.chat.renderers.files import _parse_glob_result

    body = (
        'Found 3 file(s) matching "**/*.py" within /repo, '
        "sorted by modification time (newest first):\n"
        "---\n"
        "/repo/qwowl35/app.py\n"
        "/repo/qwowl35/agent.py\n"
        "/repo/setup.py"
    )
    parsed = _parse_glob_result(body)
    assert_true(
        parsed
        == ("**/*.py", "/repo", 3,
            ["/repo/qwowl35/app.py", "/repo/qwowl35/agent.py", "/repo/setup.py"], 0),
        f"glob parsed: {parsed}",
    )
    truncated = _parse_glob_result(
        'Found 900 file(s) matching "**" within /big, '
        "sorted by modification time (newest first):\n---\n/big/a.txt\n\n"
        "(Results truncated: 899 more files matched)"
    )
    assert_true(truncated == ("**", "/big", 900, ["/big/a.txt"], 899), f"glob tail: {truncated}")
    assert_true(
        _parse_glob_result('No files found matching pattern "x" within /repo')
        == ("x", "/repo", 0, [], 0),
        "no-match message parses to a zero-path tree",
    )

    block = ToolBlock("glob")
    block.args_buf = '{"pattern": "**/*.py"}'
    block.full_result = body
    text = _plain(ChatView()._render_tool_result(block))
    assert_true("**/*.py (3)" in text and "within /repo" in text,
                f"glob header shows pattern + count + base: {text}")
    assert_true("qwowl35/app.py" in text and "setup.py" in text, "matched paths rendered")
    assert_true("/repo/qwowl35" not in "\n".join(text.splitlines()[1:]),
                "base prefix stripped from rows (shown once in the header)")
    assert_true(" └─ " in text.splitlines()[-1], "last row uses closing guide")
    assert_true("Result" not in text, "generic Result label replaced")


def test_grep_result_renders_match_tree() -> None:
    from widgets.chat.renderers.files import _parse_grep_result

    body = (
        'Found 3 matches for pattern "TODO" within /repo:\n'
        "---\n"
        "File: src/main.rs\n"
        "L12: // TODO: fix\n"
        "L30: // TODO: later\n"
        "---\n"
        "File: src/lib.rs\n"
        "L4: // TODO: docs\n"
        "---"
    )
    parsed = _parse_grep_result(body)
    assert_true(
        parsed
        == (
            "TODO",
            "within /repo",
            3,
            [
                ("src/main.rs", [("12", "// TODO: fix"), ("30", "// TODO: later")]),
                ("src/lib.rs", [("4", "// TODO: docs")]),
            ],
            [],
        ),
        f"grep parsed: {parsed}",
    )
    # compress_grep elision rows and footer notes survive the parse.
    compressed = _parse_grep_result(
        'Found 40 matches for pattern "x" within /repo:\n---\n'
        "File: a.py\nL1: x\n  … (+30 more matches in this file)\n---\n\n"
        "(Output truncated at 20000 characters; narrow the search with path/glob/limit.)"
    )
    assert_true(compressed is not None and compressed[3][0][1][1] == ("", "… (+30 more matches in this file)"),
                f"elision row kept: {compressed}")
    assert_true(compressed[4] and "Output truncated" in compressed[4][0], "footer note kept")
    assert_true(
        _parse_grep_result('No matches found for pattern "x" within /repo.')
        == ("x", "within /repo", 0, [], []),
        "no-match message parses to a zero-group tree",
    )
    assert_true(
        _parse_grep_result('No matches found for pattern "x" within /repo (filter: "*.py").')
        == ("x", 'within /repo (filter: "*.py")', 0, [], []),
        "filter suffix absorbed into the scope",
    )

    block = ToolBlock("grep_search")
    block.args_buf = '{"pattern": "TODO"}'
    block.full_result = body
    text = _plain(ChatView()._render_tool_result(block))
    assert_true("TODO (3 matches)" in text, f"grep header with count: {text}")
    assert_true("src/main.rs" in text and "src/lib.rs" in text, "file rows rendered")
    assert_true("L12 // TODO: fix" in text, f"match row with line number: {text}")
    assert_true(" │  " in text, "continuation guide under a non-last file")
    assert_true("Result" not in text, "generic Result label replaced")


def test_empty_results_render_as_tree_cards() -> None:
    # Empty results keep the tree-card shape: an empty dir shows the
    # ever-present ./ and ../ rows; glob/grep get a dim "no matches" row.
    # None of them fall back to the plain Result box.
    ls = ToolBlock("list_directory")
    ls.args_buf = '{"path": "/empty"}'
    ls.full_result = "Directory /empty is empty."
    text = _plain(ChatView()._render_tool_result(ls))
    assert_true("/empty (0)" in text, f"empty-dir header with zero count: {text}")
    assert_true(" ├─ " in text and "./" in text and "../" in text,
                f"./ and ../ rows fill the empty tree: {text}")
    assert_true("Result" not in text, f"no generic Result label: {text}")

    glob = ToolBlock("glob")
    glob.args_buf = '{"pattern": "*"}'
    glob.full_result = 'No files found matching pattern "*" within /empty'
    text = _plain(ChatView()._render_tool_result(glob))
    assert_true("* (0)" in text and "within /empty" in text,
                f"glob no-match header: {text}")
    assert_true(" └─ no matches" in text, f"dim no-matches row: {text}")
    assert_true("Result" not in text, f"no generic Result label: {text}")

    grep = ToolBlock("grep_search")
    grep.args_buf = '{"pattern": "x"}'
    grep.full_result = 'No matches found for pattern "x" within /repo.'
    text = _plain(ChatView()._render_tool_result(grep))
    assert_true("x (0 matches)" in text, f"grep no-match header: {text}")
    assert_true(" └─ no matches" in text, f"dim no-matches row: {text}")
    assert_true("Result" not in text, f"no generic Result label: {text}")


def main() -> None:
    test_parse_ls_result_rows_and_fallbacks()
    test_ls_tree_text_header_guides_and_icons()
    test_ls_tree_text_collapse_cap_and_server_truncation()
    test_ls_result_renders_tree_and_error_falls_back()
    test_glob_result_renders_tree_card()
    test_grep_result_renders_match_tree()
    test_empty_results_render_as_tree_cards()
    print("files renderer tests passed")


if __name__ == "__main__":
    main()
