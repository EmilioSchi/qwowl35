"""Tests for the ToolBlock widget's identity and copy payload."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from widgets.chat import ChatView, ToolBlock  # noqa: E402
from widgets.chat.tool_block import _copy_payload  # noqa: E402

from chat_test_helpers import _plain, _shell_result_block, assert_true  # noqa: E402


def test_prompt_host_prefers_registered_server_address() -> None:
    from widgets.chat import set_terminal_host

    set_terminal_host("192.168.1.7")
    try:
        block = ToolBlock("bash")
        assert_true(block.prompt_host == "192.168.1.7", "prompt host is the server address")
    finally:
        set_terminal_host(None)
    fallback = ToolBlock("bash")
    assert_true(bool(fallback.prompt_host) and fallback.prompt_host != "192.168.1.7",
                "unset falls back to the local hostname")


def test_window_title_is_a_stable_terminal_id() -> None:
    a, b = ToolBlock("bash"), ToolBlock("bash")
    a.args_buf = b.args_buf = '{"command":"date"}'
    a.full_result = b.full_result = "x\n"
    text_a = _plain(ChatView()._render_tool_result(a))
    # The wire name never shows; the window carries its own 4-hex-digit id.
    assert_true("run_shell_command" not in text_a, "wire tool name hidden")
    assert_true(f"terminal #{a.term_hash}" in text_a, "window titled by its id")
    assert_true(len(a.term_hash) == 4 and a.term_hash != b.term_hash,
                "each window gets its own id")
    # Identical command, identical result — the title still tells them apart,
    # and it survives repaints unchanged.
    assert_true(f"terminal #{a.term_hash}" in _plain(ChatView()._render_tool_result(a)),
                "id stable across repaints")


def test_copy_payload() -> None:
    running = ToolBlock("bash")
    running.args_buf = '{"command":"date"}'
    assert_true(_copy_payload(running) == "date", "pre-result copy is the command alone")

    done = _shell_result_block(output="Wed Jul  8\n")
    assert_true(_copy_payload(done) == "date\nWed Jul  8", "copy carries command + output")

    advisory = _shell_result_block(output="ok\n\nSyntax check (python): f.py OK\n")
    assert_true(_copy_payload(advisory) == "date\nok", "advisory blocks stripped from copy")


def main() -> None:
    test_prompt_host_prefers_registered_server_address()
    test_window_title_is_a_stable_terminal_id()
    test_copy_payload()
    print("tool block tests passed")


if __name__ == "__main__":
    main()
