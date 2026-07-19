"""Tests for the editor's basic `grep_search` (tools/files/hashline/grep.py).

Same wire name as the explorer's grep_search but a different tool: one file,
no limit/glob/compress knobs, matches rendered as hashline id rows the edit
tools accept directly. Registry exposure is asserted for the editor sub-agent.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.files.hashline import grep as editor_grep  # noqa: E402
from tools.files.hashline.document import Document  # noqa: E402
from tools.files.hashline.output import line_view  # noqa: E402


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label}: {needle!r} not in {text!r}")


class _Root:
    """Tempdir workspace patched into the module root, restored on exit."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self._tmp.name)
        self._old = editor_grep._WORKSPACE_ROOT
        editor_grep._WORKSPACE_ROOT = self.root
        return self

    def __exit__(self, *exc):
        editor_grep._WORKSPACE_ROOT = self._old
        self._tmp.cleanup()
        return False

    def write(self, rel: str, content: str | bytes) -> Path:
        path = Path(self.root, rel)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path


def test_schema_is_the_basic_variant() -> None:
    schema = editor_grep.GREP_FILE_SCHEMA
    assert_equal(schema["name"], "grep_search", "same wire name as the explorer grep")
    params = schema["parameters"]
    assert_equal(sorted(params["required"]), ["path", "pattern"], "both args required")
    assert_equal(
        sorted(params["properties"]),
        ["path", "pattern"],
        "no limit/glob/compress knobs on the editor variant",
    )


def test_matches_render_as_id_rows() -> None:
    with _Root() as env:
        env.write("app.py", "def foo():\n    return FOO\n\nbar = 2\n")
        out = editor_grep.run_file_grep({"pattern": "foo", "path": "app.py"})
        doc = Document.load(Path(env.root, "app.py"))
        assert_contains(
            out, 'Found 2 matches for pattern "foo" in app.py (ids):', "header"
        )
        # Case-insensitive by default: line 2 (FOO) matches too.
        assert_contains(out, line_view(1, doc.lines[0]), "row 1 in id|content dialect")
        assert_contains(out, line_view(2, doc.lines[1]), "row 2 (case-insensitive)")
        assert_true("bar" not in out, "non-matching lines excluded")
        assert_true("L1:" not in out, "no explorer-style L<n>: markers")

        single = editor_grep.run_file_grep({"pattern": "bar", "path": "app.py"})
        assert_contains(single, "Found 1 match for", "singular wording")

        # Absolute paths inside the workspace are fine.
        out = editor_grep.run_file_grep(
            {"pattern": "foo", "path": str(Path(env.root, "app.py"))}
        )
        assert_contains(out, "in app.py (ids):", "absolute path renders relpath")


def test_errors_never_raise() -> None:
    with _Root() as env:
        env.write("app.py", "x = 1\n")
        env.write("bin.dat", b"\x00\x01\x02")
        os.mkdir(Path(env.root, "sub"))
        outside = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
        try:
            cases = [
                ({"path": "app.py"}, "'pattern' is required"),
                ({"pattern": "[", "path": "app.py"}, "invalid regular expression"),
                ({"pattern": "x"}, "'path' is required"),
                ({"pattern": "x", "path": "sub"}, "is a directory"),
                ({"pattern": "x", "path": "missing.py"}, "file not found"),
                ({"pattern": "x", "path": outside.name}, "outside the workspace root"),
                ({"pattern": "x", "path": "bin.dat"}, "cannot read bin.dat"),
                ({"pattern": "zz", "path": "app.py"}, 'No matches found for pattern "zz"'),
            ]
            for args, needle in cases:
                assert_contains(
                    editor_grep.run_file_grep(dict(args)), needle, f"args {args!r}"
                )
        finally:
            os.unlink(outside.name)


def test_output_capped_without_limit_arg() -> None:
    with _Root() as env:
        line = "needle " + "x" * 90
        env.write("big.py", "\n".join(line for _ in range(400)) + "\n")
        out = editor_grep.run_file_grep({"pattern": "needle", "path": "big.py"})
        assert_true(
            len(out) <= editor_grep.MAX_OUTPUT_CHARS + 100, f"output bounded ({len(out)})"
        )
        assert_contains(out, "more (output capped", "honest truncation note")
        assert_contains(out, "Found 400 matches", "true total still reported")


def test_editor_registry_exposure_and_dispatch() -> None:
    from agents import editor as editor_agent
    from orchestrator import EditorRegistry
    from tools.files.adapter import HashlineTools

    assert_true(
        "grep_search" in editor_agent.SPEC.allowed_tools, "editor allowlist has grep"
    )
    with _Root() as env:
        env.write("app.py", "alpha = 1\n")
        registry = EditorRegistry(HashlineTools())
        names = [s["function"]["name"] for s in registry.schemas()]
        assert_true("grep_search" in names, "editor wire toolset advertises grep")
        assert_equal(
            [s["function"] for s in registry.schemas() if s["function"]["name"] == "grep_search"],
            [editor_grep.GREP_FILE_SCHEMA],
            "the editor advertises the BASIC schema, not the explorer's",
        )
        out = asyncio.run(registry.execute("grep_search", {"pattern": "alpha", "path": "app.py"}))
        doc = Document.load(Path(env.root, "app.py"))
        assert_contains(out, line_view(1, doc.lines[0]), "dispatch returns id rows")
        assert_equal(registry.results, [], "grep lookups are not recorded as edits")
        assert_equal(registry.saw_attention, False, "grep never flips attention")


def main() -> None:
    test_schema_is_the_basic_variant()
    test_matches_render_as_id_rows()
    test_errors_never_raise()
    test_output_capped_without_limit_arg()
    test_editor_registry_exposure_and_dispatch()
    print("editor grep tests passed")


if __name__ == "__main__":
    main()
