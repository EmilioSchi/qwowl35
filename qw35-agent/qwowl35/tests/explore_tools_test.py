"""Tests for the explorer-stage qwen-code tool replicas.

Run directly: ``python qwowl35/tests/explore_tools_test.py``. Everything runs
against a temp directory — no network, no real project files.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.explore import (  # noqa: E402
    EXPLORE_TOOL_NAMES,
    ExploreTools,
    GLOB_NAME,
    GREP_NAME,
    INSPECT_FILE_NAME,
    LS_NAME,
)
from tools.explore import inspect_file as inspect_file_module  # noqa: E402
from tools.syntax import Validation  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def _tree(tmp: str) -> Path:
    root = Path(tmp)
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("def main():\n    return compute()\n")
    (root / "src" / "util.py").write_text("def compute():\n    return 42\n")
    (root / "README.md").write_text("# demo\nhello world\n")
    (root / ".git").mkdir()
    (root / ".git" / "junk.py").write_text("ignored = True\n")
    return root


def test_schemas_use_qwen_wire_names() -> None:
    tools = ExploreTools()
    names = [schema["function"]["name"] for schema in tools.schemas()]
    assert_equal(
        names,
        ["list_directory", "glob", "grep_search", "inspect_file"],
        "wire names match qwen-code, except read_file's inspect_file stand-in",
    )
    assert_equal(frozenset(names), EXPLORE_TOOL_NAMES, "name set consistent")


def test_inspect_file_description_is_read_only() -> None:
    from tools.explore import INSPECT_FILE_SCHEMA

    description = INSPECT_FILE_SCHEMA["description"].lower()
    assert_true("read-only" in description, "description states it is read-only")


def test_list_directory_dirs_first_and_format() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        out = ExploreTools().execute(LS_NAME, {"path": str(root)})
        assert_true(out.startswith(f"Listed 3 item(s) in {root}:\n---\n"), out)
        lines = out.split("---\n", 1)[1].splitlines()
        # Directories first (alphabetical within the group), then files.
        assert_equal(lines[0], "[DIR] .git", "dirs sorted alphabetically first")
        assert_equal(lines[1], "[DIR] src", "second directory follows")
        assert_equal(lines[2], "README.md", "files after directories")


def test_list_directory_requires_absolute_path() -> None:
    out = ExploreTools().execute(LS_NAME, {"path": "relative/dir"})
    assert_true(out.startswith("Error: Path must be absolute"), out)


def test_list_directory_ignore_patterns() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        out = ExploreTools().execute(LS_NAME, {"path": str(root), "ignore": ["*.md", ".git"]})
        assert_true("README.md" not in out, "ignored glob filtered")
        assert_true("src" in out, "others kept")


def test_glob_sorts_newest_first() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        old = root / "src" / "util.py"
        new = root / "src" / "main.py"
        os.utime(old, (1_000_000_000, 1_000_000_000))
        os.utime(new, (2_000_000_000, 2_000_000_000))
        out = ExploreTools().execute(GLOB_NAME, {"pattern": "**/*.py", "path": str(root)})
        assert_true(f'Found 2 file(s) matching "**/*.py" within {root}' in out, out)
        listing = out.split("---\n", 1)[1].splitlines()
        assert_true(listing[0].endswith("main.py"), "newest first")
        assert_true(listing[1].endswith("util.py"), "oldest last")
        assert_true(all(".git" not in line for line in listing), ".git excluded")


def test_glob_no_match_message() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = ExploreTools().execute(GLOB_NAME, {"pattern": "*.zig", "path": tmp})
        assert_true(out.startswith('No files found matching pattern "*.zig"'), out)


def test_grep_groups_by_file_with_line_markers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        out = ExploreTools().execute(GREP_NAME, {"pattern": "compute", "path": str(root)})
        assert_true(out.startswith('Found 2 matches for pattern "compute"'), out)
        assert_true("File: src/main.py" in out, f"relative file grouping: {out}")
        assert_true("L2: return compute()" in out, "line number + trimmed line")
        assert_true("File: src/util.py" in out, "second file present")


def test_grep_case_insensitive_and_filters() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        out = ExploreTools().execute(
            GREP_NAME, {"pattern": "HELLO", "path": str(root), "glob": "*.md"}
        )
        assert_true('(filter: "*.md")' in out, out)
        assert_true("File: README.md" in out, "case-insensitive by default")
        missing = ExploreTools().execute(
            GREP_NAME, {"pattern": "HELLO", "path": str(root), "glob": "*.py"}
        )
        assert_true(missing.startswith("No matches found for pattern"), missing)


def test_grep_limit_caps_lines() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        out = ExploreTools().execute(
            GREP_NAME, {"pattern": "def", "path": str(root), "limit": 1}
        )
        assert_true("Found 1 match " in out, out)


def test_inspect_file_full_and_paged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        target = root / "src" / "main.py"
        tools = ExploreTools()
        full = tools.execute(INSPECT_FILE_NAME, {"file_path": str(target)})
        assert_equal(full, "def main():\n    return compute()", "full read is raw content")
        paged = tools.execute(
            INSPECT_FILE_NAME, {"file_path": str(target), "offset": 1, "limit": 1}
        )
        assert_true(
            paged.startswith("Showing lines 2-2 of 2 total lines.\n\n---\n\n"),
            paged,
        )
        assert_true(paged.endswith("return compute()"), paged)


def test_inspect_file_byte_cap_bounds_huge_lines() -> None:
    # A line cap alone does not bound bytes: one file of enormous lines once
    # consumed 50K+ context tokens in a single result. The byte ceiling holds
    # regardless of line count.
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "huge.jsonl"
        target.write_text(("x" * 5000 + "\n") * 100)  # 100 lines, 500KB
        out = ExploreTools().execute(INSPECT_FILE_NAME, {"file_path": str(target)})
        assert_true(len(out) < 60_000, f"result bounded: {len(out)} chars")
        assert_true("result truncated at 50000 characters" in out, "truncation notice present")


def test_inspect_file_errors() -> None:
    tools = ExploreTools()
    assert_true(
        tools.execute(INSPECT_FILE_NAME, {"file_path": "rel.txt"}).startswith(
            "Error: File path must be absolute"
        ),
        "relative rejected",
    )
    assert_true(
        tools.execute(INSPECT_FILE_NAME, {"file_path": "/no/such/file.txt"}).startswith(
            "Error: File not found"
        ),
        "missing file",
    )


@contextlib.contextmanager
def _patched_validation(validate):
    """Swap inspect_file's validation hooks for a stub, restoring after."""
    saved = (inspect_file_module.validate_file, inspect_file_module.warm_lsp)
    inspect_file_module.validate_file = validate
    inspect_file_module.warm_lsp = lambda path: True
    try:
        yield
    finally:
        inspect_file_module.validate_file, inspect_file_module.warm_lsp = saved


def test_inspect_file_appends_lsp_diagnostics_uncapped() -> None:
    # 6 errors proves the section is NOT capped at the 5-issue report() limit.
    errors = [(i, 1, f"line {i}, col 1: boom {i} (Pyflakes)") for i in range(1, 7)]
    warnings = [(8, 2, "line 8, col 2: 'os' imported but unused (Pyflakes)")]
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        target = root / "src" / "main.py"
        stub = lambda path, source: Validation(errors, warnings, "python, lsp", True)
        with _patched_validation(stub):
            out = ExploreTools().execute(INSPECT_FILE_NAME, {"file_path": str(target)})
        assert_true(
            out.startswith("def main():\n    return compute()\n\n"),
            f"content first: {out}",
        )
        assert_true(
            "LSP diagnostics (python, lsp) — 6 error(s), 1 warning(s):" in out, out
        )
        for _line, _col, message in errors:
            assert_true(f"- {message}" in out, f"every error listed: {message}")
        assert_true("Warnings (not blocking):" in out, "warning sub-header present")
        assert_true(f"- {warnings[0][2]}" in out, "warning listed")


def test_inspect_file_clean_lsp_stays_silent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        target = root / "src" / "main.py"
        stub = lambda path, source: Validation([], [], "python, lsp", True)
        with _patched_validation(stub):
            out = ExploreTools().execute(INSPECT_FILE_NAME, {"file_path": str(target)})
        assert_equal(out, "def main():\n    return compute()", "clean read unchanged")


def test_inspect_file_ignores_tree_sitter_fallback() -> None:
    # LSP-only: syntax-only fallback findings never reach the read result.
    err = [(1, 1, "line 1, col 1: syntax error")]
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        target = root / "src" / "main.py"
        for label in ("python", "python — LSP unavailable, syntax-only"):
            stub = lambda path, source, label=label: Validation(err, [], label, True)
            with _patched_validation(stub):
                out = ExploreTools().execute(
                    INSPECT_FILE_NAME, {"file_path": str(target)}
                )
            assert_equal(
                out, "def main():\n    return compute()", f"no section for {label!r}"
            )


def test_inspect_file_paged_read_validates_full_text() -> None:
    seen: list[str] = []

    def stub(path, source):
        seen.append(source)
        return Validation([(1, 1, "line 1, col 1: bad (Pyflakes)")], [], "python, lsp", True)

    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        target = root / "src" / "main.py"
        with _patched_validation(stub):
            out = ExploreTools().execute(
                INSPECT_FILE_NAME, {"file_path": str(target), "offset": 1, "limit": 1}
            )
        assert_true(out.startswith("Showing lines 2-2 of 2 total lines."), out)
        assert_true(
            out.endswith(
                "return compute()\n\nLSP diagnostics (python, lsp) — "
                "1 error(s), 0 warning(s):\n- line 1, col 1: bad (Pyflakes)"
            ),
            f"section after the window: {out}",
        )
        assert_equal(
            seen, ["def main():\n    return compute()\n"], "validated FULL file text"
        )


def test_inspect_file_dedups_repeat_diagnostics_per_instance() -> None:
    # The same broken file inspected twice by ONE agent (one ExploreTools
    # instance): the second read keeps the honest headline but does not
    # repeat the rows; a FRESH instance (fresh agent) sees everything again.
    errors = [(1, 1, "line 1, col 1: boom (pylint)")]
    stub = lambda path, source: Validation(errors, [], "python, lsp", True)
    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        target = root / "src" / "main.py"
        tools = ExploreTools()
        with _patched_validation(stub):
            first = tools.execute(INSPECT_FILE_NAME, {"file_path": str(target)})
            second = tools.execute(INSPECT_FILE_NAME, {"file_path": str(target)})
            fresh = ExploreTools().execute(INSPECT_FILE_NAME, {"file_path": str(target)})
        assert_true("- line 1, col 1: boom (pylint)" in first, first)
        assert_true(
            second.endswith(
                "LSP diagnostics (python, lsp) — 1 error(s), 0 warning(s): "
                "all unchanged and already reported above."
            ),
            f"repeat collapses to the honest one-liner: {second}",
        )
        assert_true("- line 1, col 1: boom" not in second, "row not repeated")
        assert_true("- line 1, col 1: boom (pylint)" in fresh, "fresh agent sees all")


def test_inspect_file_partial_dedup_lists_only_new_rows() -> None:
    old = (1, 1, "line 1, col 1: boom (pylint)")
    new = (2, 5, "line 2, col 5: Undefined name 'compute' (Pyflakes)")
    calls: list[list] = [[old], [old, new]]

    def stub(path, source):
        return Validation(list(calls.pop(0)), [], "python, lsp", True)

    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        target = root / "src" / "main.py"
        tools = ExploreTools()
        with _patched_validation(stub):
            tools.execute(INSPECT_FILE_NAME, {"file_path": str(target)})
            second = tools.execute(INSPECT_FILE_NAME, {"file_path": str(target)})
        assert_true(
            "LSP diagnostics (python, lsp) — 2 error(s), 0 warning(s):" in second,
            f"headline counts the CURRENT state: {second}",
        )
        assert_true(f"- {new[2]}" in second, "new row listed")
        assert_true(f"- {old[2]}" not in second, "old row suppressed")
        assert_true(
            "- 1 unchanged error(s) already reported above (not repeated)" in second,
            second,
        )


def test_inspect_file_survives_validation_crash() -> None:
    def stub(path, source):
        raise RuntimeError("lsp exploded")

    with tempfile.TemporaryDirectory() as tmp:
        root = _tree(tmp)
        target = root / "src" / "main.py"
        with _patched_validation(stub):
            out = ExploreTools().execute(INSPECT_FILE_NAME, {"file_path": str(target)})
        assert_equal(out, "def main():\n    return compute()", "crash swallowed")


def main() -> None:
    test_schemas_use_qwen_wire_names()
    test_inspect_file_description_is_read_only()
    test_list_directory_dirs_first_and_format()
    test_list_directory_requires_absolute_path()
    test_list_directory_ignore_patterns()
    test_glob_sorts_newest_first()
    test_glob_no_match_message()
    test_grep_groups_by_file_with_line_markers()
    test_grep_case_insensitive_and_filters()
    test_grep_limit_caps_lines()
    test_inspect_file_full_and_paged()
    test_inspect_file_byte_cap_bounds_huge_lines()
    test_inspect_file_errors()
    test_inspect_file_appends_lsp_diagnostics_uncapped()
    test_inspect_file_clean_lsp_stays_silent()
    test_inspect_file_ignores_tree_sitter_fallback()
    test_inspect_file_dedups_repeat_diagnostics_per_instance()
    test_inspect_file_partial_dedup_lists_only_new_rows()
    test_inspect_file_paged_read_validates_full_text()
    test_inspect_file_survives_validation_crash()
    print("explore tools tests passed")


if __name__ == "__main__":
    main()
