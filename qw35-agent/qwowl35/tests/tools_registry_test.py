"""Tests for model-facing tool registry behavior."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.pipeline import PipelineRegistry  # noqa: E402
from client import _parse_tool_args  # noqa: E402
from tools_registry import ToolRegistry  # noqa: E402

REPEATED_ECHO = (
    'for i in $(seq 1 300); do echo "the same log line over and over"; done'
)


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
            "read_file",
            "replace",
            "insert",
            "delete",
            "run_shell_command",
        ],
        f"active tool list changed: {names}",
    )


def test_file_mutation_schemas_are_id_only() -> None:
    schemas = {
        schema["function"]["name"]: schema["function"]["parameters"]
        for schema in ToolRegistry().schemas()
        if schema["function"]["name"]
        in {
            "replace",
            "insert",
            "delete",
        }
    }
    assert_true(schemas["replace"]["required"] == ["file", "id", "content"], "replace required args")
    assert_true(schemas["insert"]["required"] == ["file", "id", "content"], "insert required args")
    assert_true(schemas["delete"]["required"] == ["file", "id"], "delete required args")
    for name, params in schemas.items():
        props = params["properties"]
        assert_true("start_query" not in props and "end_query" not in props, f"{name} hides query args")
        assert_true("anchor" not in props, f"{name} uses 'id', not 'anchor'")
    insert_id_description = schemas["insert"]["properties"]["id"]["description"].lower()
    assert_true("range" not in insert_id_description, "insert id schema must not advertise ranges")
    replace_id_description = schemas["replace"]["properties"]["id"]["description"].lower()
    delete_id_description = schemas["delete"]["properties"]["id"]["description"].lower()
    assert_true("range" in replace_id_description, "replace id schema advertises ranges")
    assert_true("range" in delete_id_description, "delete id schema advertises ranges")


def test_read_file_schema_does_not_advertise_bash_handoff() -> None:
    begin = next(schema["function"] for schema in ToolRegistry().schemas() if schema["function"]["name"] == "read_file")
    serialized = json.dumps(begin)
    assert_true("bash/rg" not in serialized, "read_file schema avoids bash/rg handoff")
    assert_true("file_path" in begin["parameters"]["properties"], "read_file takes 'file_path'")
    assert_true(begin["parameters"]["required"] == ["file_path"], "only file_path is required")


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


def test_stage_allowlist_denies_out_of_stage_tools() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        registry.set_allowed(frozenset({"replace", "insert", "delete"}))
        result = await registry.execute("bash", {"command": "echo hi"})
        assert_true("not available in this stage" in result, f"denied: {result!r}")
        assert_true("replace" in result, "denial lists the allowed tools")
        # Unrestricting restores normal execution.
        registry.set_allowed(None)
        result = await registry.execute("bash", {"command": "echo hi"})
        assert_true(result.strip() == "hi", f"unrestricted bash runs: {result!r}")

    asyncio.run(run())


def test_bash_output_is_compressed() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        result = await registry.execute("bash", {"command": REPEATED_ECHO})
        assert_true("[compressed:" in result, f"marker present: {result[-200:]!r}")
        assert_true("repeated × 300" in result, "repeat count reported")
        assert_true(len(result) < 2000, f"result actually shrank: {len(result)}")
        assert_true(registry.compress_saved_chars > 0, "saved counter advanced")

    asyncio.run(run())


def test_compress_false_returns_raw() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        for flag in (False, "false"):
            result = await registry.execute(
                "bash", {"command": REPEATED_ECHO, "compress": flag}
            )
            assert_true("[compressed:" not in result, f"raw with compress={flag!r}")
            assert_true(
                result.count("the same log line over and over") == 300,
                f"full output with compress={flag!r}",
            )

    asyncio.run(run())


def test_compress_master_switch_off() -> None:
    async def run() -> None:
        registry = ToolRegistry(compress=False)
        result = await registry.execute("bash", {"command": REPEATED_ECHO})
        assert_true("[compressed:" not in result, "raw when disabled by config")

    asyncio.run(run())


def test_compress_arg_not_leaked_to_executor() -> None:
    async def run() -> None:
        args = {"command": "echo hi", "is_background": False, "compress": False}
        result = await ToolRegistry().execute("bash", args)
        assert_true(result.strip() == "hi", f"command still ran: {result!r}")
        assert_true(args.get("compress") is False, "caller's arguments not mutated")

    asyncio.run(run())


def test_pipeline_registry_forwards_rerank_flag() -> None:
    async def run() -> None:
        import agents.pipeline as pipeline_mod

        captured: dict = {}
        original = pipeline_mod.compress_tool_result

        def spy(name, args, text, rerank=True):
            captured["rerank"] = rerank
            return original(name, args, text, rerank=rerank)

        pipeline_mod.compress_tool_result = spy
        try:
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "f.txt").write_text("needle\n" * 5, encoding="utf-8")
                registry = pipeline_mod.PipelineRegistry(rerank=False)
                registry.set_stage("explore", frozenset({"grep_search"}))
                await registry.execute("grep_search", {"pattern": "needle", "path": tmp})
        finally:
            pipeline_mod.compress_tool_result = original
        assert_true(captured.get("rerank") is False, "rerank=False forwarded to compressor")

    asyncio.run(run())


def test_pipeline_registry_compresses_grep() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "data.py"
            target.write_text(
                "\n".join(
                    f"value = needle_{i}  # a reasonably long matching line" for i in range(60)
                ),
                encoding="utf-8",
            )
            registry = PipelineRegistry()
            registry.set_stage("explore", frozenset({"grep_search"}))
            args = {"pattern": "needle", "path": tmp}
            compressed = await registry.execute("grep_search", args)
            assert_true("[compressed:" in compressed, f"grep compressed: {compressed[-200:]!r}")
            assert_true("more matches in this file" in compressed, "per-file cap noted")
            raw = await registry.execute("grep_search", {**args, "compress": False})
            assert_true("[compressed:" not in raw, "raw grep with compress=false")
            assert_true(raw.count("needle_") == 60, "all matches present raw")

    asyncio.run(run())


def main() -> None:
    test_bash_can_write_files()
    test_bash_output_is_compressed()
    test_compress_false_returns_raw()
    test_compress_master_switch_off()
    test_compress_arg_not_leaked_to_executor()
    test_pipeline_registry_forwards_rerank_flag()
    test_pipeline_registry_compresses_grep()
    test_registry_exposes_only_active_tools()
    test_file_mutation_schemas_are_id_only()
    test_begin_transaction_schema_does_not_advertise_large_file_workflow()
    test_old_mutation_tool_names_are_not_accepted()
    test_bash_reports_invalid_json_before_missing_command()
    test_bash_reports_non_object_json_args()
    test_bash_runs_recovered_quoted_command()
    test_bash_runs_recovered_dangling_quote_command()
    test_stage_allowlist_denies_out_of_stage_tools()
    print("tools registry tests passed")


if __name__ == "__main__":
    main()
