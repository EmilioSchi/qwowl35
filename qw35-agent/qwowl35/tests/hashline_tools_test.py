"""Tests for hashline-backed file tools."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.files.adapter import (  # noqa: E402
    Document,
    HashlineTools,
    NewlineStyle,
    SearchDocument,
    byte_len,
    find_line_by_query,
    format_line_ref,
    format_short_hash,
    parse_anchor,
    resolve_query_region,
    short_hash_value,
    stream_replace_line,
    xxh32,
)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def _line_id(output: str, line_no: int) -> str:
    """Extract the ``<line><hash>`` id for ``line_no`` from a beginTransaction body.

    The id has no separator (the ``:`` was dropped for token efficiency); the hash
    is the fixed final two hex chars, so the line is the leading digits.
    """
    match = re.search(rf"^{line_no}([0-9a-f]{{2}})\|", output, re.MULTILINE)
    if not match:
        raise AssertionError(f"id for line {line_no} not found in:\n{output}")
    return f"{line_no}{match.group(1)}"


def test_xxh32_matches_hashline_reference_values() -> None:
    assert_equal(xxh32(b"", 0), 0x02CC5D05, "empty xxh32")
    assert_equal(xxh32(b"abc", 0), 0x32D153FF, "abc xxh32")
    assert_equal(format_short_hash(short_hash_value("abc")), "ff", "short hash is low byte")
    assert_equal(short_hash_value("abc   "), short_hash_value("abc"), "trailing whitespace ignored")


def test_line_ref_has_no_separator_and_round_trips() -> None:
    # The rendered id is line digits + 2-hex hash with NO separator, and it parses
    # back to the same line + hash (an optional ':' is still accepted).
    short = short_hash_value("    return 1")
    ref = format_line_ref(12, short)
    assert_equal(ref, f"12{format_short_hash(short)}", "no separator in id")
    parsed = parse_anchor(ref)
    assert_equal(parsed.line, 12, "id parses back to line")
    assert_equal(parsed.short, short & 0xFF, "id parses back to hash")
    assert_equal(parse_anchor("12:af").line, 12, "optional ':' still parses (line)")
    assert_equal(parse_anchor("12:af").short, 0xAF, "optional ':' still parses (hash)")


def test_schema_advertises_begin_transaction_only() -> None:
    tools = HashlineTools()
    schemas = {s["function"]["name"]: s["function"] for s in tools.schemas()}
    assert_true("beginTransaction" in schemas, "beginTransaction advertised")
    assert_true("read" not in schemas, "old 'read' name gone")
    props = schemas["beginTransaction"]["parameters"]["properties"]
    assert_equal(sorted(props), ["file"], "beginTransaction takes only 'file'")
    for name in ("edit", "insert", "delete"):
        assert_true("id" in schemas[name]["parameters"]["properties"], f"{name} uses 'id'")
        assert_true("anchor" not in schemas[name]["parameters"]["properties"], f"{name} has no 'anchor'")
    # The retired tool name is not dispatchable.
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "m.py").write_text("x = 1\n", encoding="utf-8")
        assert_true(
            tools.execute("read", {"file": str(Path(tmp, "m.py"))}).startswith("Error: unknown tool"),
            "execute('read') is now unknown",
        )


def test_begin_transaction_edit_insert_delete_flow() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            tools = HashlineTools()
            shown = tools.execute("beginTransaction", {"file": "m.py"})
            return_id = _line_id(shown, 2)

            edited = tools.execute("edit", {"file": "m.py", "id": return_id, "content": "    return 2"})
            assert_true(edited.startswith("Edited line 2"), f"edit result: {edited}")
            assert_true("|    return 2" in edited, "fresh ids after edit")
            assert_true(_line_id(edited, 2), "edit snippet shows a line-2 id")
            assert_equal(Path("m.py").read_text(encoding="utf-8"), "def f():\n    return 2\n", "file edited")

            reread = tools.execute("beginTransaction", {"file": "m.py"})
            def_id = _line_id(reread, 1)
            inserted = tools.execute("insert", {"file": "m.py", "id": def_id, "position": "before", "content": "import os"})
            assert_true(inserted.startswith("Inserted line 1"), f"insert result: {inserted}")
            assert_true(Path("m.py").read_text(encoding="utf-8").startswith("import os\n"), "file inserted")

            after_insert = tools.execute("beginTransaction", {"file": "m.py"})
            import_id = _line_id(after_insert, 1)
            deleted = tools.execute("delete", {"file": "m.py", "id": import_id})
            assert_true(deleted.startswith("Deleted line 1"), f"delete result: {deleted}")
            assert_equal(Path("m.py").read_text(encoding="utf-8"), "def f():\n    return 2\n", "file deleted")
        finally:
            os.chdir(cwd)


def test_insert_range_id_reports_single_id_guidance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("m.py").write_text("import math\n\ndef f():\n    return 1\n", encoding="utf-8")
            tools = HashlineTools()
            shown = tools.execute("beginTransaction", {"file": "m.py"})
            import_id = _line_id(shown, 1)
            def_id = _line_id(shown, 3)
            result = tools.execute(
                "insert",
                {
                    "file": "m.py",
                    "id": f"{import_id}..{def_id}",
                    "content": "import os",
                },
            )
            assert_true(
                "requires one line id, not a range" in result,
                f"range insert guidance: {result}",
            )
            assert_true("position='after'" in result and "position='before'" in result, f"range insert repair hints: {result}")
            assert_equal(Path("m.py").read_text(encoding="utf-8"), "import math\n\ndef f():\n    return 1\n", "range insert did not write")
        finally:
            os.chdir(cwd)


def test_stale_id_reports_context_and_flips_on_edit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("m.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            tools = HashlineTools()
            shown = tools.execute("beginTransaction", {"file": "m.txt"})
            beta_id = _line_id(shown, 2)
            # The content cross-check must flip when the line content changes.
            Path("m.txt").write_text("alpha\nchanged\ngamma\n", encoding="utf-8")
            reopened = tools.execute("beginTransaction", {"file": "m.txt", "_force": True})
            assert_true(_line_id(reopened, 2) != beta_id, "id flips when line content changes")
            result = tools.execute("edit", {"file": "m.txt", "id": beta_id, "content": "new"})
            assert_true(result.startswith("Error: stale anchor"), f"stale result: {result}")
            assert_true(">>> 2" in result, "stale context points at current line")
        finally:
            os.chdir(cwd)


def test_id_parser_rejects_copied_line_rows() -> None:
    try:
        parse_anchor("1aa|content")
    except Exception:
        return
    raise AssertionError("parse_anchor accepted a copied read row")


def test_query_region_and_search_document_match_hashline_names() -> None:
    doc = Document.from_str("demo.py", "def a():\n    pass\n\ndef b():\n    return 1\n")
    assert_equal(find_line_by_query(doc, "def b"), 4, "find_line_by_query")
    region = resolve_query_region(doc, "def a", "    pass")
    assert_true(region is not None, "region exists")
    assert_equal((region.start_line, region.end_line), (1, 2), "resolve_query_region")

    search = SearchDocument.new("alpha\nbeta\n")
    matches = search.grep_lines("beta", False)
    assert_equal(len(matches), 1, "grep match count")
    assert_equal(matches[0].n, 2, "grep line number")


def test_document_stats_and_stream_replace_line() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.txt"
        path.write_text("alpha\nbeta\n", encoding="utf-8")
        doc = Document.load(path)
        stats = doc.compute_stats()
        assert_equal(stats.line_count, 2, "stats line count")
        expected = short_hash_value("beta")
        stream_replace_line(path, 1, "gamma", expected, NewlineStyle.Lf, True)
        assert_equal(path.read_text(encoding="utf-8"), "alpha\ngamma\n", "stream replace")


def test_document_content_len_uses_utf8_bytes_and_loads_meta() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.txt"
        path.write_text("café\nβ\n", encoding="utf-8")
        doc = Document.load(path)
        expected_len = byte_len("café") + byte_len("β")
        assert_equal(doc.content_len, expected_len, "content_len is byte length")
        assert_true(doc.file_meta is not None, "file_meta loaded")


def test_start_end_keyword_ids_resolve_positionally() -> None:
    # The model emits bare "start"/"end" ids by nature; they must resolve to the
    # first/last line without any hash, across all four tools.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("m.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
            tools = HashlineTools()
            tools.execute("beginTransaction", {"file": "m.py"})

            appended = tools.execute(
                "insert",
                {"file": "m.py", "id": "end", "position": "after", "content": "d = 4"},
            )
            assert_true(appended.startswith("Inserted line 4"), f"append at end: {appended}")
            assert_equal(
                Path("m.py").read_text(encoding="utf-8"),
                "a = 1\nb = 2\nc = 3\nd = 4\n",
                "end/after appends",
            )

            prepended = tools.execute(
                "insert",
                {"file": "m.py", "id": "start", "position": "before", "content": "import x"},
            )
            assert_true(prepended.startswith("Inserted line 1"), f"prepend at start: {prepended}")
            assert_true(
                Path("m.py").read_text(encoding="utf-8").startswith("import x\n"),
                "start/before prepends",
            )

            edited = tools.execute(
                "edit", {"file": "m.py", "id": "END", "content": "d = 40"}
            )
            assert_true(edited.startswith("Edited line 5"), f"edit end (caps): {edited}")
            assert_true(
                Path("m.py").read_text(encoding="utf-8").rstrip().endswith("d = 40"),
                "end edits the last line",
            )

            deleted = tools.execute(
                "delete", {"file": "m.py", "id": "start"}
            )
            assert_true(deleted.startswith("Deleted line 1"), f"delete start: {deleted}")
            assert_true(
                not Path("m.py").read_text(encoding="utf-8").startswith("import x"),
                "start deletes the first line",
            )

            shown = tools.execute("beginTransaction", {"file": "m.py"})
            assert_true("d = 40" in shown, f"beginTransaction shows the tail: {shown}")

            bogus = tools.execute("edit", {"file": "m.py", "id": "nope", "content": "x"})
            assert_true(bogus.startswith("Error: invalid anchor"), f"bogus id still errors: {bogus}")
        finally:
            os.chdir(cwd)


def test_noop_edit_reports_no_changes() -> None:
    """An edit whose content matches the targeted lines must announce the no-op
    rather than emit a bare success line the model mistakes for a real change."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            tools = HashlineTools()
            shown = tools.execute("beginTransaction", {"file": "m.py"})
            return_id = _line_id(shown, 2)

            # Resend the exact existing content: a byte-identical no-op edit.
            result = tools.execute("edit", {"file": "m.py", "id": return_id, "content": "    return 1"})
            assert_true("No changes were made" in result, f"no-op edit must say so: {result}")
            assert_true("byte-identical" in result, f"no-op note explains why: {result}")
            assert_true("Diff:" not in result, f"no-op edit has no diff: {result}")
            assert_equal(Path("m.py").read_text(encoding="utf-8"), "def f():\n    return 1\n", "file unchanged")
        finally:
            os.chdir(cwd)


def test_mutation_denied_without_begin_transaction() -> None:
    """edit/insert/delete on a file never opened with beginTransaction are denied
    with advice to open it first, and the file on disk stays untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            original = "def f():\n    return 1\n"
            Path("m.py").write_text(original, encoding="utf-8")
            Path("other.py").write_text("x = 1\n", encoding="utf-8")
            tools = HashlineTools()

            for name, extra in (
                ("edit", {"content": "    return 2"}),
                ("insert", {"content": "import os", "position": "before"}),
                ("delete", {}),
            ):
                result = tools.execute(name, {"file": "m.py", "id": "1aa", **extra})
                assert_true(result.startswith(f"Error: {name} denied"), f"{name} denied: {result}")
                assert_true("beginTransaction" in result, f"{name} denial advises beginTransaction: {result}")
                assert_equal(Path("m.py").read_text(encoding="utf-8"), original, f"file untouched after denied {name}")

            # Opening a DIFFERENT file does not unlock this one.
            tools.execute("beginTransaction", {"file": "other.py"})
            still = tools.execute("edit", {"file": "m.py", "id": "1aa", "content": "x"})
            assert_true(still.startswith("Error: edit denied"), f"gate is per-file: {still}")

            # After opening the file, the same mutation with a real id succeeds.
            shown = tools.execute("beginTransaction", {"file": "m.py"})
            return_id = _line_id(shown, 2)
            edited = tools.execute("edit", {"file": "m.py", "id": return_id, "content": "    return 2"})
            assert_true(edited.startswith("Edited line 2"), f"edit allowed after open: {edited}")

            # A follow-up mutation without re-opening still passes the gate
            # (the session remembers the file; stale ids stay covered by hashes).
            second_id = _line_id(edited, 2)
            second = tools.execute("edit", {"file": "m.py", "id": second_id, "content": "    return 3"})
            assert_true(second.startswith("Edited line 2"), f"second edit allowed: {second}")
        finally:
            os.chdir(cwd)


def main() -> None:
    test_xxh32_matches_hashline_reference_values()
    test_line_ref_has_no_separator_and_round_trips()
    test_schema_advertises_begin_transaction_only()
    test_begin_transaction_edit_insert_delete_flow()
    test_mutation_denied_without_begin_transaction()
    test_noop_edit_reports_no_changes()
    test_insert_range_id_reports_single_id_guidance()
    test_stale_id_reports_context_and_flips_on_edit()
    test_id_parser_rejects_copied_line_rows()
    test_query_region_and_search_document_match_hashline_names()
    test_document_stats_and_stream_replace_line()
    test_document_content_len_uses_utf8_bytes_and_loads_meta()
    test_start_end_keyword_ids_resolve_positionally()
    print("hashline tool tests passed")


if __name__ == "__main__":
    main()
