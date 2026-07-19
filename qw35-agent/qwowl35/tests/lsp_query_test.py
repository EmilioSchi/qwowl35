"""Tests for the `lsp` navigation tool (tools/lsp/query.py) and its wiring.

No real language server is spawned: `diagnostics.get_ready_server` is stubbed
with a fake handle whose `sync_ls` records the calls it receives and returns
canned multilspy-shaped payloads. Registry exposure is asserted for the
explorer and editor sub-agents (and absence everywhere else).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.lsp import LSP_NAME, LSP_SCHEMA, LspQueryTool, diagnostics  # noqa: E402
from tools.lsp import query as lsp_query  # noqa: E402


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label}: {needle!r} not in {text!r}")


class FakeSyncLS:
    """Records request args; returns whatever the test loaded per method."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.definition = []
        self.references = []
        self.hover = None
        self.symbols = ([], None)
        self.raise_exc: Exception | None = None

    def _maybe_raise(self):
        if self.raise_exc is not None:
            raise self.raise_exc

    def request_definition(self, relpath, line, col):
        self.calls.append(("definition", relpath, line, col))
        self._maybe_raise()
        return self.definition

    def request_references(self, relpath, line, col):
        self.calls.append(("references", relpath, line, col))
        self._maybe_raise()
        return self.references

    def request_hover(self, relpath, line, col):
        self.calls.append(("hover", relpath, line, col))
        self._maybe_raise()
        return self.hover

    def request_document_symbols(self, relpath):
        self.calls.append(("symbols", relpath))
        self._maybe_raise()
        return self.symbols


class FakeHandle:
    def __init__(self, status: str = "ready") -> None:
        self.status = status
        self.lock = threading.Lock()
        self.sync_ls = FakeSyncLS()


class _Env:
    """Tempdir workspace + stubbed get_ready_server, restored on exit."""

    def __init__(self, status: str = "ready") -> None:
        self.handle = FakeHandle(status)

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self._tmp.name)
        Path(self.root, "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        self._old_root = lsp_query._WORKSPACE_ROOT
        self._old_get = diagnostics.get_ready_server
        lsp_query._WORKSPACE_ROOT = self.root
        diagnostics.get_ready_server = lambda language, root, wait=0: self.handle
        return self

    def __exit__(self, *exc):
        lsp_query._WORKSPACE_ROOT = self._old_root
        diagnostics.get_ready_server = self._old_get
        self._tmp.cleanup()
        return False


def loc(rel: str, line0: int, col0: int, abs_path: str = "") -> dict:
    return {
        "uri": f"file://{abs_path or '/' + rel}",
        "range": {
            "start": {"line": line0, "character": col0},
            "end": {"line": line0, "character": col0 + 1},
        },
        "absolutePath": abs_path or f"/{rel}",
        "relativePath": rel,
    }


def test_schema_matches_qwen_interface() -> None:
    assert_equal(LSP_SCHEMA["name"], "lsp", "wire name")
    params = LSP_SCHEMA["parameters"]
    assert_equal(params["required"], ["operation"], "only operation required")
    assert_equal(
        params["properties"]["operation"]["enum"],
        ["goToDefinition", "findReferences", "hover", "documentSymbol"],
        "exactly the 4 supported operations",
    )
    for name in ("filePath", "line", "character"):
        assert_true(name in params["properties"], f"qwen-code param {name} present")
    assert_true(
        "includeDeclaration" not in params["properties"],
        "includeDeclaration not advertised (multilspy hard-codes False)",
    )
    assert_true(
        "limit" not in params["properties"],
        "limit not advertised (fixed RESULT_LIMITS caps instead)",
    )


def test_argument_validation() -> None:
    tool = LspQueryTool()
    with _Env():
        bad_op = tool.execute({"operation": "codeActions", "filePath": "app.py"})
        assert_contains(bad_op, "'operation' must be one of", "unknown op → enum error")
        assert_contains(
            tool.execute({"operation": "hover"}), "'filePath' is required", "missing filePath"
        )
        assert_contains(
            tool.execute({"operation": "hover", "filePath": "app.py"}),
            "'line' must be a 1-based line number or a current line id",
            "missing line for position op",
        )
        assert_contains(
            tool.execute({"operation": "hover", "filePath": "app.py", "line": 0}),
            "'line' must be a 1-based line number or a current line id",
            "zero line rejected",
        )
        assert_contains(
            tool.execute(
                {"operation": "hover", "filePath": "app.py", "line": 1, "character": 0}
            ),
            "'character' must be a positive number",
            "zero character rejected",
        )
        assert_true(
            "Error" not in tool.execute(
                {"operation": "documentSymbol", "filePath": "app.py", "limit": 0}
            ),
            "unknown limit arg silently ignored",
        )


def test_position_conversion_wire_1based_to_lsp_0based() -> None:
    tool = LspQueryTool()
    with _Env() as env:
        env.handle.sync_ls.definition = [loc("app.py", 11, 4)]
        out = tool.execute(
            {"operation": "goToDefinition", "filePath": "app.py", "line": 12, "character": 5}
        )
        assert_equal(
            env.handle.sync_ls.calls[-1], ("definition", "app.py", 11, 4), "0-based to multilspy"
        )
        assert_contains(out, "Goto definition for app.py:12:5:", "1-based heading")
        assert_contains(out, "1. app.py:12:5", "1-based result row")
        # character defaults to 1 → column 0.
        tool.execute({"operation": "hover", "filePath": "app.py", "line": 2})
        assert_equal(env.handle.sync_ls.calls[-1], ("hover", "app.py", 1, 0), "default character")


def test_references_formatting_cap_and_empty() -> None:
    tool = LspQueryTool()
    cap = lsp_query.RESULT_LIMITS["findReferences"]
    with _Env() as env:
        env.handle.sync_ls.references = [loc("app.py", i, 0) for i in range(cap + 2)]
        out = tool.execute(
            {"operation": "findReferences", "filePath": "app.py", "line": 1}
        )
        assert_contains(out, "References for app.py:1:1:", "heading")
        assert_contains(out, f"{cap}. app.py:{cap}:1", "last row within cap present")
        assert_true(f"{cap + 1}. " not in out, "cap truncates rows")
        assert_contains(out, "- … and 2 more", "overflow note")
        env.handle.sync_ls.references = []
        empty = tool.execute({"operation": "findReferences", "filePath": "app.py", "line": 1})
        assert_equal(empty, "No references found for app.py:1:1.", "empty references")
        env.handle.sync_ls.definition = []
        assert_equal(
            tool.execute({"operation": "goToDefinition", "filePath": "app.py", "line": 1}),
            "No definition found for app.py:1:1.",
            "empty definition",
        )


def test_hover_content_variants() -> None:
    tool = LspQueryTool()
    with _Env() as env:
        args = {"operation": "hover", "filePath": "app.py", "line": 1, "character": 5}
        env.handle.sync_ls.hover = {"contents": {"kind": "markdown", "value": "def f()"}}
        assert_equal(
            tool.execute(args), "Hover for app.py:1:5:\ndef f()", "MarkupContent dict"
        )
        env.handle.sync_ls.hover = {"contents": "plain signature"}
        assert_contains(tool.execute(args), "plain signature", "bare string contents")
        env.handle.sync_ls.hover = {
            "contents": ["first", {"language": "python", "value": "second"}]
        }
        out = tool.execute(args)
        assert_contains(out, "first", "list item 1")
        assert_contains(out, "second", "list item 2 (MarkedString dict)")
        env.handle.sync_ls.hover = None
        assert_equal(
            tool.execute(args), "No hover information for app.py:1:5.", "missing hover"
        )
        env.handle.sync_ls.hover = {"contents": {"kind": "markdown", "value": "  "}}
        assert_equal(
            tool.execute(args), "No hover information for app.py:1:5.", "blank hover"
        )


def test_document_symbols_formatting() -> None:
    tool = LspQueryTool()
    with _Env() as env:
        env.handle.sync_ls.symbols = (
            [
                {
                    "name": "lorenz_system",
                    "kind": 12,
                    "selectionRange": {"start": {"line": 11, "character": 4}},
                },
                {
                    "name": "update",
                    "kind": 12,
                    "containerName": "main",
                    "range": {"start": {"line": 70, "character": 8}},
                },
                {"name": "odd", "kind": 99, "location": {"range": {"start": {"line": 0, "character": 0}}}},
            ],
            None,
        )
        out = tool.execute({"operation": "documentSymbol", "filePath": "app.py"})
        assert_contains(out, "Document symbols for app.py:", "heading")
        assert_contains(out, "1. lorenz_system (Function) - app.py:12:5", "kind name + position")
        assert_contains(out, "2. update (Function) in main - app.py:71:9", "container rendered")
        assert_contains(out, "3. odd (kind 99)", "unknown kind degrades readably")
        assert_equal(len(env.handle.sync_ls.calls), 1, "one request")
        env.handle.sync_ls.symbols = ([], None)
        assert_equal(
            tool.execute({"operation": "documentSymbol", "filePath": "app.py"}),
            "No symbols found in app.py.",
            "empty symbols",
        )


def test_hashline_rendering_for_editor() -> None:
    # With the private _hashline flag (injected by the editor registry only),
    # rows in the REQUESTED file render in the editor's dialect —
    # <line><hash>|content — while foreign-file and out-of-range rows keep
    # plain path:line:col. Without the flag, rendering is unchanged.
    from tools.files.hashline.document import Document
    from tools.files.hashline.hash import format_line_ref

    tool = LspQueryTool()
    with _Env() as env:
        doc = Document.load(Path(env.root, "app.py"))
        id1 = format_line_ref(1, doc.lines[0].short_hash)
        env.handle.sync_ls.definition = [loc("app.py", 0, 4)]
        out = tool.execute(
            {"operation": "goToDefinition", "filePath": "app.py", "line": 2, "_hashline": True}
        )
        assert_contains(out, f"1. {id1}|def f():", f"definition row in hashline dialect: {out}")
        assert_true(":1:5" not in out.split("\n", 1)[1], "no plain position on the id row")

        plain = tool.execute({"operation": "goToDefinition", "filePath": "app.py", "line": 2})
        assert_contains(plain, "1. app.py:1:5", "flag off → plain rendering unchanged")

        env.handle.sync_ls.symbols = (
            [{"name": "f", "kind": 12, "selectionRange": {"start": {"line": 0, "character": 4}}}],
            None,
        )
        out = tool.execute(
            {"operation": "documentSymbol", "filePath": "app.py", "_hashline": True}
        )
        assert_contains(out, f"1. f (Function) - {id1}|def f():", f"symbol row dialect: {out}")

        # Foreign file and out-of-range lines fall back to plain positions.
        env.handle.sync_ls.references = [
            loc("other.py", 3, 0, abs_path=os.path.join(env.root, "other.py")),
            loc("app.py", 99, 0),
        ]
        out = tool.execute(
            {"operation": "findReferences", "filePath": "app.py", "line": 1, "_hashline": True}
        )
        assert_contains(out, "1. other.py:4:1", "foreign-file row stays plain")
        assert_contains(out, "2. app.py:100:1", "out-of-range row stays plain")


def test_document_symbols_without_uri_resolve_to_queried_file() -> None:
    # Hierarchical DocumentSymbol rows carry no per-symbol URI — per LSP the
    # location is implicitly the QUERIED file — so rows must render its
    # relpath (never "<unknown>") and translate to id rows under _hashline.
    from tools.files.hashline.document import Document
    from tools.files.hashline.hash import format_line_ref

    tool = LspQueryTool()
    with _Env() as env:
        env.handle.sync_ls.symbols = (
            [
                {   # SymbolInformation whose Location lost its URI/paths
                    "name": "f",
                    "kind": 12,
                    "location": {"range": {"start": {"line": 0, "character": 0}}},
                },
                {   # pure hierarchical shape: no location keys at all
                    "name": "ret",
                    "kind": 13,
                    "selectionRange": {"start": {"line": 1, "character": 4}},
                },
            ],
            None,
        )
        out = tool.execute({"operation": "documentSymbol", "filePath": "app.py"})
        assert_contains(out, "1. f (Function) - app.py:1:1", "pathless Location → queried file")
        assert_contains(out, "2. ret (Variable) - app.py:2:5", "URI-less symbol → queried file")
        assert_true("<unknown>" not in out, f"no <unknown> rows: {out}")

        doc = Document.load(Path(env.root, "app.py"))
        id1 = format_line_ref(1, doc.lines[0].short_hash)
        id2 = format_line_ref(2, doc.lines[1].short_hash)
        out = tool.execute(
            {"operation": "documentSymbol", "filePath": "app.py", "_hashline": True}
        )
        assert_contains(
            out, f"1. f (Function) - {id1}|def f():", f"pathless row in id dialect: {out}"
        )
        assert_contains(
            out, f"2. ret (Variable) - {id2}|    return 1", f"URI-less row in id dialect: {out}"
        )


def test_hashline_line_id_accepted_for_position_operations() -> None:
    # The editor speaks hashline ids ("12af" = line 12, hash af): `line`
    # accepts that grammar in every mode and is resolved (hash-verified)
    # against the live file; plain ints keep working
    # (test_hashline_rendering_for_editor covers those). Under _hashline the
    # result rows additionally render in the id|content dialect.
    from tools.files.hashline.document import Document
    from tools.files.hashline.hash import format_line_ref

    tool = LspQueryTool()
    with _Env() as env:
        doc = Document.load(Path(env.root, "app.py"))
        id1 = format_line_ref(1, doc.lines[0].short_hash)
        id2 = format_line_ref(2, doc.lines[1].short_hash)
        env.handle.sync_ls.hover = {"contents": "int"}
        out = tool.execute(
            {
                "operation": "hover",
                "filePath": "app.py",
                "line": id2,
                "character": 5,
                "_hashline": True,
            }
        )
        assert_equal(
            env.handle.sync_ls.calls[-1], ("hover", "app.py", 1, 4), "id resolved to line 2"
        )
        assert_contains(out, "Hover for app.py:2:5:", "heading shows the resolved position")

        env.handle.sync_ls.definition = [loc("app.py", 0, 4)]
        out = tool.execute(
            {
                "operation": "goToDefinition",
                "filePath": "app.py",
                "line": id1,
                "_hashline": True,
            }
        )
        assert_equal(
            env.handle.sync_ls.calls[-1], ("definition", "app.py", 0, 0), "id resolved to line 1"
        )
        assert_contains(out, f"1. {id1}|def f():", "id-addressed query renders id rows")


def test_hashline_line_id_stale_and_malformed() -> None:
    from tools.files.hashline.document import Document
    from tools.files.hashline.hash import format_short_hash

    tool = LspQueryTool()
    with _Env() as env:
        doc = Document.load(Path(env.root, "app.py"))
        current = format_short_hash(doc.lines[0].short_hash)
        wrong = "00" if current != "00" else "01"
        out = tool.execute(
            {"operation": "hover", "filePath": "app.py", "line": f"1{wrong}", "_hashline": True}
        )
        assert_contains(
            out, f"Error: stale line id '1{wrong}' for app.py", "wrong hash → stale error"
        )
        assert_contains(out, f"current id is '1{current}'", "stale error offers the current id")
        assert_contains(
            out, "Re-read the file to refresh the ids you hold", "stale error says re-check ids"
        )

        out = tool.execute(
            {"operation": "hover", "filePath": "app.py", "line": "99ab", "_hashline": True}
        )
        assert_contains(
            out,
            "Error: stale line id '99ab' for app.py; the file has 2 lines",
            "past-EOF id is stale",
        )
        assert_equal(env.handle.sync_ls.calls, [], "stale ids never reach the language server")

        dual = "Error: 'line' must be a 1-based line number or a current line id like '12af'."
        for bad in ("zz", "12", "12AF", "0af", "", "1g2"):
            assert_equal(
                tool.execute(
                    {"operation": "hover", "filePath": "app.py", "line": bad, "_hashline": True}
                ),
                dual,
                f"malformed id {bad!r} → dual-form error",
            )
        assert_equal(
            tool.execute({"operation": "hover", "filePath": "app.py", "_hashline": True}),
            dual,
            "missing line under _hashline → dual-form error",
        )
        assert_equal(
            tool.execute(
                {"operation": "hover", "filePath": "app.py", "line": 0, "_hashline": True}
            ),
            dual,
            "invalid numeric line under _hashline → dual-form error",
        )


def test_line_string_id_without_flag_resolves_but_renders_plain() -> None:
    # `line` addresses by plain number first, with the hashline id grammar as
    # a universal fallback: a string id resolves (hash-verified) from ANY
    # agent, but the id|content result dialect stays gated on the editor's
    # private _hashline flag — non-editor agents always read path:line:col.
    from tools.files.hashline.document import Document
    from tools.files.hashline.hash import format_line_ref, format_short_hash

    tool = LspQueryTool()
    with _Env() as env:
        doc = Document.load(Path(env.root, "app.py"))
        id2 = format_line_ref(2, doc.lines[1].short_hash)
        env.handle.sync_ls.definition = [loc("app.py", 0, 4)]
        out = tool.execute(
            {"operation": "goToDefinition", "filePath": "app.py", "line": id2, "character": 5}
        )
        assert_equal(
            env.handle.sync_ls.calls[-1],
            ("definition", "app.py", 1, 4),
            "string id resolved without the flag",
        )
        assert_contains(out, "1. app.py:1:5", "flag off → id query still renders plain rows")
        assert_true("|" not in out, f"no id|content dialect without the flag: {out}")

        # Hash verification guards the flag-less path too.
        current = format_short_hash(doc.lines[0].short_hash)
        wrong = "00" if current != "00" else "01"
        assert_contains(
            tool.execute({"operation": "hover", "filePath": "app.py", "line": f"1{wrong}"}),
            f"Error: stale line id '1{wrong}' for app.py",
            "stale id caught without the flag",
        )

        # Non-id strings and invalid numbers get the dual-form error everywhere.
        dual = "Error: 'line' must be a 1-based line number or a current line id like '12af'."
        for bad in ("3", None, 0):
            assert_equal(
                tool.execute({"operation": "hover", "filePath": "app.py", "line": bad}),
                dual,
                f"line {bad!r} rejected with the dual-form message",
            )
        env.handle.sync_ls.hover = {"contents": "sig"}
        tool.execute({"operation": "hover", "filePath": "app.py", "line": 2, "character": 3})
        assert_equal(
            env.handle.sync_ls.calls[-1], ("hover", "app.py", 1, 2), "plain int unchanged"
        )


def test_guards_disabled_unsupported_missing_outside_and_status() -> None:
    tool = LspQueryTool()
    with _Env() as env:
        diagnostics.configure(False)
        try:
            assert_contains(
                tool.execute({"operation": "documentSymbol", "filePath": "app.py"}),
                "LSP is disabled",
                "master switch off",
            )
        finally:
            diagnostics.configure(True)
        Path(env.root, "a.cpp").write_text("int x;\n", encoding="utf-8")
        assert_contains(
            tool.execute({"operation": "documentSymbol", "filePath": "a.cpp"}),
            "no language server for .cpp",
            "unsupported extension",
        )
        assert_contains(
            tool.execute({"operation": "documentSymbol", "filePath": "gone.py"}),
            "file not found",
            "missing file",
        )
        outside = os.path.realpath(tempfile.gettempdir())
        probe = Path(outside, "lsp_query_probe.py")
        probe.write_text("x = 1\n", encoding="utf-8")
        try:
            assert_contains(
                tool.execute({"operation": "documentSymbol", "filePath": str(probe)}),
                "outside the workspace root",
                "outside-root file",
            )
        finally:
            probe.unlink()
        env.handle.status = "failed"
        assert_contains(
            tool.execute({"operation": "documentSymbol", "filePath": "app.py"}),
            "failed to start",
            "failed server",
        )
        env.handle.status = "starting"
        assert_contains(
            tool.execute({"operation": "documentSymbol", "filePath": "app.py"}),
            "still starting",
            "booting server",
        )


def test_request_errors_degrade_never_raise() -> None:
    tool = LspQueryTool()
    with _Env() as env:
        env.handle.sync_ls.raise_exc = RuntimeError("boom")
        out = tool.execute({"operation": "documentSymbol", "filePath": "app.py"})
        assert_contains(out, "Error: LSP request failed (boom)", "generic failure degrades")
        env.handle.sync_ls.raise_exc = TimeoutError()
        out = tool.execute({"operation": "documentSymbol", "filePath": "app.py"})
        assert_contains(out, "did not answer within", "timeout message")


def test_get_ready_server_waits_for_boot() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        handle = FakeHandle("starting")
        old = diagnostics._get_or_start
        diagnostics._get_or_start = lambda language, root: handle  # type: ignore[assignment]
        try:
            timer = threading.Timer(0.3, lambda: setattr(handle, "status", "ready"))
            timer.start()
            got = diagnostics.get_ready_server("python", tmp, wait=5.0)
            timer.cancel()
            assert_equal(got.status, "ready", "waited through the boot")
            handle.status = "starting"
            got = diagnostics.get_ready_server("python", tmp, wait=0.2)
            assert_equal(got.status, "starting", "bounded wait returns starting handle")
        finally:
            diagnostics._get_or_start = old


def test_registry_exposure() -> None:
    from agents import editor as editor_agent
    from agents import explorer as explorer_agent
    from agents.pipeline import ExplorerRegistry, PipelineRegistry
    from orchestrator import EditorRegistry
    from tools.files.adapter import HashlineTools

    assert_true("lsp" in explorer_agent.SPEC.allowed_tools, "explorer allowlist")
    assert_true("lsp" in editor_agent.SPEC.allowed_tools, "editor allowlist")

    pipeline = PipelineRegistry()
    explorer = ExplorerRegistry(pipeline)
    names = [s["function"]["name"] for s in explorer.schemas()]
    assert_true("lsp" in names, "explorer wire toolset advertises lsp")

    for stage, schemas in pipeline._stage_schemas.items():
        stage_names = [s["function"]["name"] for s in schemas]
        assert_true("lsp" not in stage_names, f"stage {stage} does not advertise lsp")

    class StubLsp:
        def execute(self, arguments: dict) -> str:
            return f"Hover for app.py:1:1:\nstubbed hashline={arguments.get('_hashline')!r}"

    editor_registry = EditorRegistry(HashlineTools(), StubLsp())
    names = [s["function"]["name"] for s in editor_registry.schemas()]
    assert_true("lsp" in names, "editor wire toolset advertises lsp")
    model_args = {"operation": "hover", "filePath": "app.py", "line": 1}
    result = asyncio.run(editor_registry.execute("lsp", model_args))
    assert_contains(result, "stubbed", "editor dispatches lsp")
    assert_contains(result, "hashline=True", "editor injects the hashline dialect flag")
    assert_true("_hashline" not in model_args, "the model's parsed args are never mutated")
    assert_equal(editor_registry.results, [], "lsp lookups are not recorded as edits")
    assert_equal(editor_registry.saw_attention, False, "lsp never flips attention")

    bare = EditorRegistry(HashlineTools())
    names = [s["function"]["name"] for s in bare.schemas()]
    assert_true("lsp" not in names, "no engine → not advertised")


def test_explorer_registry_dispatches_lsp() -> None:
    from agents.pipeline import ExplorerRegistry, PipelineRegistry

    pipeline = PipelineRegistry()

    class StubLsp:
        def execute(self, arguments: dict) -> str:
            return f"stub:{arguments.get('operation')}:hashline={arguments.get('_hashline')!r}"

    pipeline.lsp = StubLsp()
    explorer = ExplorerRegistry(pipeline)
    result = asyncio.run(
        explorer.execute("lsp", {"operation": "documentSymbol", "filePath": "app.py"})
    )
    assert_equal(
        result,
        "stub:documentSymbol:hashline=None",
        "explorer routes lsp to the shared engine WITHOUT the editor dialect flag",
    )


def main() -> None:
    test_schema_matches_qwen_interface()
    test_argument_validation()
    test_position_conversion_wire_1based_to_lsp_0based()
    test_references_formatting_cap_and_empty()
    test_hover_content_variants()
    test_document_symbols_formatting()
    test_hashline_rendering_for_editor()
    test_document_symbols_without_uri_resolve_to_queried_file()
    test_hashline_line_id_accepted_for_position_operations()
    test_hashline_line_id_stale_and_malformed()
    test_line_string_id_without_flag_resolves_but_renders_plain()
    test_guards_disabled_unsupported_missing_outside_and_status()
    test_request_errors_degrade_never_raise()
    test_get_ready_server_waits_for_boot()
    test_registry_exposure()
    test_explorer_registry_dispatches_lsp()
    print("lsp query tool tests passed")


if __name__ == "__main__":
    main()
