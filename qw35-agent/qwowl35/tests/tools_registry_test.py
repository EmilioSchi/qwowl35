"""Tests for model-facing tool registry behavior."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client import _parse_tool_args  # noqa: E402
from tools_registry import ToolRegistry  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def test_bash_can_write_files() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                registry = ToolRegistry()
                result = await registry.execute("bash", {"command": "printf 'hi\\n' > out.txt"})
                assert_true("bash is not allowed" not in result, "bash write is not categorically denied")
                assert_true(Path("out.txt").read_text(encoding="utf-8") == "hi\n", "bash redirect wrote the file")
            finally:
                os.chdir(cwd)

    asyncio.run(run())


def test_registry_exposes_only_active_tools() -> None:
    names = [schema["function"]["name"] for schema in ToolRegistry().schemas()]
    assert_true(
        names == [
            "beginTransaction",
            "edit",
            "insert",
            "delete",
            "bash",
        ],
        f"active tool list changed: {names}",
    )


def test_file_mutation_schemas_are_id_only() -> None:
    schemas = {
        schema["function"]["name"]: schema["function"]["parameters"]
        for schema in ToolRegistry().schemas()
        if schema["function"]["name"]
        in {
            "edit",
            "insert",
            "delete",
        }
    }
    assert_true(schemas["edit"]["required"] == ["file", "id", "content"], "edit required args")
    assert_true(schemas["insert"]["required"] == ["file", "id", "content"], "insert required args")
    assert_true(schemas["delete"]["required"] == ["file", "id"], "delete required args")
    for name, params in schemas.items():
        props = params["properties"]
        assert_true("start_query" not in props and "end_query" not in props, f"{name} hides query args")
        assert_true("anchor" not in props, f"{name} uses 'id', not 'anchor'")
    insert_id_description = schemas["insert"]["properties"]["id"]["description"].lower()
    assert_true("range" not in insert_id_description, "insert id schema must not advertise ranges")
    edit_id_description = schemas["edit"]["properties"]["id"]["description"].lower()
    delete_id_description = schemas["delete"]["properties"]["id"]["description"].lower()
    assert_true("range" in edit_id_description, "edit id schema advertises ranges")
    assert_true("range" in delete_id_description, "delete id schema advertises ranges")


def test_begin_transaction_schema_does_not_advertise_large_file_workflow() -> None:
    begin = next(schema["function"] for schema in ToolRegistry().schemas() if schema["function"]["name"] == "beginTransaction")
    serialized = json.dumps(begin)
    assert_true("large" not in serialized.lower(), "beginTransaction schema avoids large-file workflow")
    assert_true("bash/rg" not in serialized, "beginTransaction schema avoids bash/rg handoff")
    assert_true(sorted(begin["parameters"]["properties"]) == ["file"], "beginTransaction takes only 'file'")


def test_old_mutation_tool_names_are_not_accepted() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        cases = [
            ("edit_lines", {"file": "m.py", "anchor": "1:aa", "content": "x"}),
            ("insert_lines", {"file": "m.py", "anchor": "1:aa", "content": "x"}),
            ("delete_lines", {"file": "m.py", "anchor": "1:aa"}),
        ]
        for name, args in cases:
            result = await registry.execute(name, args)
            assert_true(f"unknown tool {name!r}" in result, f"old tool name rejected: {name}")

    asyncio.run(run())


def test_bash_reports_invalid_json_before_missing_command() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        result = await registry.execute("bash", _parse_tool_args('{"command": cal -n"}}'))
        assert_true("not a valid JSON object" in result, "invalid JSON reported")
        assert_true("'command' is required" not in result, "missing command not reported")

    asyncio.run(run())


def test_bash_reports_non_object_json_args() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        result = await registry.execute("bash", _parse_tool_args('"cal -n"'))
        assert_true("must be a JSON object" in result, "object requirement reported")
        assert_true("'command' is required" not in result, "missing command not reported")

    asyncio.run(run())


def test_bash_runs_recovered_quoted_command() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        result = await registry.execute("bash", _parse_tool_args('{"command":"printf "ok""}'))
        assert_true(result == "ok", f"recovered command ran: {result!r}")

    asyncio.run(run())


def test_bash_runs_recovered_dangling_quote_command() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                registry = ToolRegistry()
                args = _parse_tool_args('{"command":"touch cal.py && echo ""}')
                result = await registry.execute("bash", args)
                assert_true(Path("cal.py").exists(), "recovered touch command ran")
                assert_true("Exit code" not in result, f"command succeeded: {result!r}")
            finally:
                os.chdir(cwd)

    asyncio.run(run())


def main() -> None:
    test_bash_can_write_files()
    test_registry_exposes_only_active_tools()
    test_file_mutation_schemas_are_id_only()
    test_begin_transaction_schema_does_not_advertise_large_file_workflow()
    test_old_mutation_tool_names_are_not_accepted()
    test_bash_reports_invalid_json_before_missing_command()
    test_bash_reports_non_object_json_args()
    test_bash_runs_recovered_quoted_command()
    test_bash_runs_recovered_dangling_quote_command()
    print("tools registry tests passed")


if __name__ == "__main__":
    main()
