"""Tests for qw35 streaming client helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client import _error_from_payload, _parse_tool_args  # noqa: E402


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def test_invalid_tool_args_are_compact() -> None:
    result = _parse_tool_args('{"content": "unterminated')
    assert_true(result.get("_invalid_json") is True, "invalid JSON marker")
    assert_true("line 1 column" in result.get("_json_error", ""), "compact parser detail")
    assert_true("unterminated" not in repr(result), "raw payload omitted")


def test_malformed_bash_args_are_invalid_json() -> None:
    result = _parse_tool_args('{"command": cal -n"}}')
    assert_true(result.get("_invalid_json") is True, "malformed bash JSON marker")
    assert_true("command" not in result, "malformed command not treated as missing field")


def test_bash_command_with_unescaped_quotes_is_recovered() -> None:
    result = _parse_tool_args('{"command":"printf "ok""}')
    assert_equal(result, {"command": 'printf "ok"'}, "recover quoted bash command")


def test_bash_command_with_dangling_trailing_quote_is_trimmed() -> None:
    result = _parse_tool_args('{"command":"touch cal.py && echo ""}')
    assert_equal(result, {"command": "touch cal.py && echo"}, "trim dangling quote")


def test_bash_command_with_equals_typo_is_recovered() -> None:
    result = _parse_tool_args('{"command="python3 cal.py"}}')
    assert_equal(result, {"command": "python3 cal.py"}, "recover command= typo")


def test_xml_tool_arguments_are_recovered() -> None:
    result = _parse_tool_args(
        "<tool_call>\n"
        "<function=bash>\n"
        "<parameter=command>printf &quot;ok&quot; &amp;&amp; echo done</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    assert_equal(
        result,
        {"command": 'printf "ok" && echo done'},
        "recover nested xml command",
    )


def test_non_object_tool_args_are_invalid() -> None:
    result = _parse_tool_args('["cal", "-n"]')
    assert_true(result.get("_invalid_json") is True, "array args marker")
    assert_true("must be a JSON object" in result.get("_json_error", ""), "object requirement")


def test_unquoted_escaped_content_is_recovered() -> None:
    raw = (
        '{"file":"solve_real_root.py","anchor":"22:1d","content":\\n'
        "    import matplotlib.pyplot as plt\\n"
        "    print(\\\"Plot saved as solve_real_root_plot.png\\\")\"}"
    )
    result = _parse_tool_args(raw)
    assert_equal(result["file"], "solve_real_root.py", "file")
    assert_equal(result["anchor"], "22:1d", "anchor")
    assert_true(
        result["content"].startswith("    import matplotlib"),
        "leading newline stripped but indentation kept",
    )
    assert_true('"Plot saved as solve_real_root_plot.png"' in result["content"], "quotes decoded")


def test_sse_error_payload_maps_to_qw35_error() -> None:
    err = _error_from_payload(
        {
            "code": "bad_request",
            "message": "prompt plus max_tokens requires 4203 context slots",
            "type": "qw35_error",
        }
    )
    assert_equal(err.code, "bad_request", "error code")
    assert_equal(err.kind, "qw35_error", "error kind")
    assert_true("4203" in err.message, "error message")


def test_prefill_chunk_carries_session_ctx() -> None:
    from client import PrefillProgress, _classify_chunk

    chunk = {
        "choices": [],
        "qw35_prefill": {"processed": 512, "total": 4096, "percent": 12.5, "session_ctx": 24576},
    }
    events = list(_classify_chunk(chunk))
    assert_equal(
        events,
        [PrefillProgress(percent=12.5, processed=512, total=4096, session_ctx=24576)],
        "prefill chunk with the serving session's live ctx",
    )
    legacy = list(_classify_chunk({"choices": [], "qw35_prefill": {"processed": 1, "total": 2}}))
    assert_equal(legacy[0].session_ctx, None, "absent session_ctx stays None")


def main() -> None:
    test_invalid_tool_args_are_compact()
    test_malformed_bash_args_are_invalid_json()
    test_bash_command_with_unescaped_quotes_is_recovered()
    test_bash_command_with_dangling_trailing_quote_is_trimmed()
    test_bash_command_with_equals_typo_is_recovered()
    test_xml_tool_arguments_are_recovered()
    test_non_object_tool_args_are_invalid()
    test_unquoted_escaped_content_is_recovered()
    test_sse_error_payload_maps_to_qw35_error()
    test_prefill_chunk_carries_session_ctx()
    print("client tests passed")


if __name__ == "__main__":
    main()


def test_qw35_tool_call_side_channel_chunks_classify() -> None:
    from client import (
        ToolCallDemoted,
        ToolCallFinal,
        ToolCallName,
        _classify_chunk,
    )

    name = list(_classify_chunk({"choices": [], "qw35_tool_call": {"event": "name", "index": 0, "name": "bash"}}))
    assert_equal(name, [ToolCallName(index=0, name="bash")], "name chunk")
    final = list(_classify_chunk({"choices": [], "qw35_tool_call": {"event": "final", "index": 1, "arguments": "{\"command\":\"ls\"}"}}))
    assert_equal(final, [ToolCallFinal(index=1, arguments='{"command":"ls"}')], "final chunk")
    demoted = list(_classify_chunk({"choices": [], "qw35_tool_call": {"event": "demoted", "index": 0}}))
    assert_equal(demoted, [ToolCallDemoted(index=0)], "demoted chunk")


def test_begin_with_empty_name_still_classifies() -> None:
    from client import ToolCallBegin, _classify_chunk

    chunk = {
        "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "", "arguments": ""}}]}}]
    }
    events = list(_classify_chunk(chunk))
    assert_equal(events, [ToolCallBegin(index=0, id="call_1", name="")], "empty-name begin")


def test_accumulator_final_replaces_raw_xml_fragments() -> None:
    from client import (
        StreamAccumulator,
        ToolCallArgsDelta,
        ToolCallBegin,
        ToolCallFinal,
        ToolCallName,
    )

    acc = StreamAccumulator()
    acc.add(ToolCallBegin(index=0, id="call_9", name=""))
    acc.add(ToolCallArgsDelta(index=0, fragment="\n<function=bash>\n<parameter=command>\n"))
    acc.add(ToolCallName(index=0, name="bash"))
    acc.add(ToolCallArgsDelta(index=0, fragment="echo hi\n</parameter>\n</function>\n"))
    acc.add(ToolCallFinal(index=0, arguments='{"command":"echo hi"}'))
    turn = acc.finalize()
    assert_equal(len(turn.tool_calls), 1, "one call")
    assert_equal(turn.tool_calls[0].name, "bash", "name from side-channel")
    assert_equal(turn.tool_calls[0].arguments, {"command": "echo hi"}, "final JSON wins")


def test_accumulator_raw_xml_fallback_when_stream_dies_before_final() -> None:
    from client import StreamAccumulator, ToolCallArgsDelta, ToolCallBegin, ToolCallName

    acc = StreamAccumulator()
    acc.add(ToolCallBegin(index=0, id="call_9", name=""))
    acc.add(ToolCallName(index=0, name="bash"))
    acc.add(
        ToolCallArgsDelta(
            index=0,
            fragment="\n<function=bash>\n<parameter=command>\necho hi\n</parameter>\n</function>\n",
        )
    )
    turn = acc.finalize()
    assert_equal(turn.tool_calls[0].arguments, {"command": "echo hi"}, "XML recovery fallback")


def test_accumulator_demoted_call_is_dropped_and_content_kept() -> None:
    from client import (
        ContentDelta,
        StreamAccumulator,
        ToolCallArgsDelta,
        ToolCallBegin,
        ToolCallDemoted,
    )

    acc = StreamAccumulator()
    acc.add(ToolCallBegin(index=0, id="call_9", name=""))
    acc.add(ToolCallArgsDelta(index=0, fragment="\nnope\n"))
    acc.add(ToolCallDemoted(index=0))
    acc.add(ContentDelta("<tool_call>\nnope\n</tool_call>"))
    turn = acc.finalize()
    assert_equal(turn.tool_calls, [], "demoted call dropped")
    assert_equal(turn.content, "<tool_call>\nnope\n</tool_call>", "demoted text kept as content")
