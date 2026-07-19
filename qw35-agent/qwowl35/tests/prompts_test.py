"""Tests for the qwowl35 system prompt."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts import build_system_message  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def test_prompt_uses_xml_tool_examples() -> None:
    prompt = build_system_message("/tmp/qw35-test")["content"]

    assert_true("/tmp/qw35-test" in prompt, "cwd inserted")
    assert_true("<<CWD>>" not in prompt and "<<PLATFORM>>" not in prompt, "placeholders replaced")
    assert_true("echo 'DONE: description'" in prompt, "single-quoted status example")
    assert_true('echo "DONE: description"' not in prompt, "double-quoted status example absent")
    assert_true("grep -E 'a|b|c'" in prompt, "single-quoted grep example")
    assert_true('grep -E "a|b|c"' not in prompt, "double-quoted grep example absent")
    assert_true("For large or noisy files" not in prompt, "old large-file wording removed")
    xml_snippets = [
        "<function=run_shell_command>",
        "<parameter=command>touch file.py</parameter>",
        "<function=read_file>",
        "<parameter=file>path.py</parameter>",
        "<parameter=id>12af</parameter>",
        "<function=replace>",
        "<parameter=content>    return 2</parameter>",
        "<parameter=id>12af..189c</parameter>",
        "<function=insert>",
        "<parameter=position>after</parameter>",
        "<parameter=position>before</parameter>",
        "<function=delete>",
    ]
    for snippet in xml_snippets:
        assert_true(snippet in prompt, f"xml snippet present: {snippet}")
    assert_true("Hashline" not in prompt and "hashline" not in prompt, "internal term absent")
    assert_true('{"name":"read_file"' not in prompt, "flat JSON read_file example absent")
    assert_true('{"name":"edit_lines"' not in prompt, "flat JSON edit example absent")
    assert_true('<bash command=' not in prompt, "compact bash example absent")
    assert_true('<read_file file=' not in prompt, "compact read_file example absent")
    assert_true('<edit file=' not in prompt, "compact edit example absent")
    assert_true('<insert file=' not in prompt, "compact insert example absent")
    assert_true('<delete file=' not in prompt, "compact delete example absent")
    assert_true("<edit_lines " not in prompt, "old edit tool name absent")
    assert_true('"arguments":{"file"' not in prompt, "nested edit example absent")
    assert_true("Do not emit JSON inside <tool_call>" in prompt, "strict XML guidance present")
    assert_true("<insert_lines " not in prompt, "old insert tool name absent")
    assert_true("<delete_lines " not in prompt, "old delete tool name absent")


def main() -> None:
    test_prompt_uses_xml_tool_examples()
    print("prompt tests passed")


if __name__ == "__main__":
    main()
