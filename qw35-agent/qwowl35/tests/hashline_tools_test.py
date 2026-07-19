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
from tools.files.adapter import TOOL_ATTENTION_MARKER  # noqa: E402


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def _line_id(output: str, line_no: int) -> str:
    """Extract the ``<line><hash>`` id for ``line_no`` from a read_file body.

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


def test_schema_advertises_read_file_only() -> None:
    tools = HashlineTools()
    schemas = {s["function"]["name"]: s["function"] for s in tools.schemas()}
    assert_true("read_file" in schemas, "read_file advertised")
    assert_true("read" not in schemas, "old 'read' name gone")
    assert_true("beginTransaction" not in schemas, "old 'beginTransaction' name gone")
    props = schemas["read_file"]["parameters"]["properties"]
    assert_true("file_path" in props, "read_file takes 'file_path'")
    assert_equal(
        schemas["read_file"]["parameters"]["required"], ["file_path"],
        "only file_path is required",
    )
    for name in ("replace", "insert", "delete"):
        assert_true("id" in schemas[name]["parameters"]["properties"], f"{name} uses 'id'")
        assert_true("anchor" not in schemas[name]["parameters"]["properties"], f"{name} has no 'anchor'")
    # The retired tool name is not dispatchable.
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "m.py").write_text("x = 1\n", encoding="utf-8")
        assert_true(
            tools.execute("read", {"file": str(Path(tmp, "m.py"))}).startswith("Error: unknown tool"),
            "execute('read') is now unknown",
        )


def test_read_file_edit_insert_delete_flow() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            tools = HashlineTools()
            shown = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})
            return_id = _line_id(shown, 2)

            edited = tools.execute("replace",{"file": "m.py", "id": return_id, "content": "    return 2"})
            assert_true(edited.startswith("Edited line 2"), f"edit result: {edited}")
            assert_true("|    return 2" in edited, "fresh ids after edit")
            assert_true(_line_id(edited, 2), "edit snippet shows a line-2 id")
            assert_equal(Path("m.py").read_text(encoding="utf-8"), "def f():\n    return 2\n", "file edited")

            reread = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})
            def_id = _line_id(reread, 1)
            inserted = tools.execute("insert", {"file": "m.py", "id": def_id, "position": "before", "content": "import os"})
            assert_true(inserted.startswith("Inserted line 1"), f"insert result: {inserted}")
            assert_true(Path("m.py").read_text(encoding="utf-8").startswith("import os\n"), "file inserted")

            after_insert = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})
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
            shown = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})
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
            shown = tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
            beta_id = _line_id(shown, 2)
            # The content cross-check must flip when the line content changes.
            Path("m.txt").write_text("alpha\nchanged\ngamma\n", encoding="utf-8")
            reopened = tools.execute("read_file", {"file_path": "m.txt", "_force": True})
            assert_true(_line_id(reopened, 2) != beta_id, "id flips when line content changes")
            result = tools.execute("replace",{"file": "m.txt", "id": beta_id, "content": "new"})
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
            tools.execute("read_file", {"file_path": os.path.abspath("m.py")})

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
                "replace", {"file": "m.py", "id": "END", "content": "d = 40"}
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

            shown = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})
            assert_true("d = 40" in shown, f"read_file shows the tail: {shown}")

            bogus = tools.execute("replace",{"file": "m.py", "id": "nope", "content": "x"})
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
            shown = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})
            return_id = _line_id(shown, 2)

            # Resend the exact existing content: a byte-identical no-op edit.
            result = tools.execute("replace",{"file": "m.py", "id": return_id, "content": "    return 1"})
            assert_true("No changes were made" in result, f"no-op edit must say so: {result}")
            assert_true("byte-identical" in result, f"no-op note explains why: {result}")
            assert_true("Diff:" not in result, f"no-op edit has no diff: {result}")
            assert_equal(Path("m.py").read_text(encoding="utf-8"), "def f():\n    return 1\n", "file unchanged")
        finally:
            os.chdir(cwd)


def test_mutation_denied_without_begin_transaction() -> None:
    """replace/insert/delete on a file never opened with read_file are denied
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
                ("replace", {"content": "    return 2"}),
                ("insert", {"content": "import os", "position": "before"}),
                ("delete", {}),
            ):
                result = tools.execute(name, {"file": "m.py", "id": "1aa", **extra})
                assert_true(result.startswith(f"Error: {name} denied"), f"{name} denied: {result}")
                assert_true("read_file" in result, f"{name} denial advises read_file: {result}")
                assert_equal(Path("m.py").read_text(encoding="utf-8"), original, f"file untouched after denied {name}")

            # Opening a DIFFERENT file does not unlock this one.
            tools.execute("read_file", {"file_path": os.path.abspath("other.py")})
            still = tools.execute("replace",{"file": "m.py", "id": "1aa", "content": "x"})
            assert_true(still.startswith("Error: replace denied"), f"gate is per-file: {still}")

            # After opening the file, the same mutation with a real id succeeds.
            shown = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})
            return_id = _line_id(shown, 2)
            edited = tools.execute("replace",{"file": "m.py", "id": return_id, "content": "    return 2"})
            assert_true(edited.startswith("Edited line 2"), f"edit allowed after open: {edited}")

            # A follow-up mutation without re-opening still passes the gate
            # (the session remembers the file; stale ids stay covered by hashes).
            second_id = _line_id(edited, 2)
            second = tools.execute("replace",{"file": "m.py", "id": second_id, "content": "    return 3"})
            assert_true(second.startswith("Edited line 2"), f"second edit allowed: {second}")
        finally:
            os.chdir(cwd)


def _in_tmp(fn) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            fn()
        finally:
            os.chdir(cwd)


def test_read_file_relative_path_rejected_force_accepted() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\n", encoding="utf-8")
        tools = HashlineTools()
        rejected = tools.execute("read_file", {"file_path": "m.txt"})
        assert_equal(rejected, "Error: File path must be absolute: m.txt", "relative path rejected")
        forced = tools.execute("read_file", {"file_path": "m.txt", "_force": True})
        assert_true("|a" in forced, f"_force waives the absolute check: {forced}")

    _in_tmp(body)


def test_read_file_offset_limit_window() -> None:
    def body() -> None:
        Path("m.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)), encoding="utf-8")
        tools = HashlineTools()
        paged = tools.execute(
            "read_file", {"file_path": os.path.abspath("m.txt"), "offset": 4, "limit": 3}
        )
        assert_true(
            "Showing lines 5-7 of 10 total lines." in paged, f"window banner: {paged}"
        )
        assert_true("|line5" in paged and "|line7" in paged, f"window rows shown: {paged}")
        assert_true("|line4" not in paged and "|line8" not in paged, f"rows outside window absent: {paged}")
        # The rendered ids keep their true 1-based line numbers.
        assert_true(_line_id(paged, 5), "id for line 5 present")
        # A paged read satisfies the mutation gate: its ids are genuine.
        edited = tools.execute(
            "replace", {"file": "m.txt", "id": _line_id(paged, 5), "content": "LINE5"}
        )
        assert_true(edited.startswith("Edited line 5"), f"paged read unlocks mutations: {edited}")

    _in_tmp(body)


def test_read_file_offset_past_end_errors() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\nc\n", encoding="utf-8")
        tools = HashlineTools()
        result = tools.execute(
            "read_file", {"file_path": os.path.abspath("m.txt"), "offset": 50, "limit": 5}
        )
        assert_equal(
            result, "Error: offset 50 is past the end of the file (3 lines).",
            "past-end offset errors",
        )

    _in_tmp(body)


def test_read_file_offset_without_limit_reads_to_end() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
        tools = HashlineTools()
        paged = tools.execute("read_file", {"file_path": os.path.abspath("m.txt"), "offset": 2})
        assert_true("Showing lines 3-5 of 5 total lines." in paged, f"lenient offset-only: {paged}")
        assert_true("|c" in paged and "|e" in paged and "|a" not in paged, f"tail window: {paged}")

    _in_tmp(body)


def test_read_file_string_offset_limit_coerced() -> None:
    # XML tool-call parameters can reach the tool as strings.
    def body() -> None:
        Path("m.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
        tools = HashlineTools()
        paged = tools.execute(
            "read_file", {"file_path": os.path.abspath("m.txt"), "offset": "1", "limit": "2"}
        )
        assert_true("Showing lines 2-3 of 4 total lines." in paged, f"string args coerced: {paged}")

    _in_tmp(body)


def test_paged_read_does_not_set_or_consult_f1_baseline() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
        tools = HashlineTools()
        # Paged first: must not record the full-read baseline...
        tools.execute("read_file", {"file_path": os.path.abspath("m.txt"), "offset": 1, "limit": 2})
        full = tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        assert_true("Skipped re-opening" not in full, f"full read after paged is served: {full}")
        # ...while after a full read, an explicit page always serves too.
        paged = tools.execute(
            "read_file", {"file_path": os.path.abspath("m.txt"), "offset": 0, "limit": 2}
        )
        assert_true("Skipped re-opening" not in paged, f"explicit page never suppressed: {paged}")
        assert_true("|a" in paged and "|b" in paged, f"page served with rows: {paged}")
        # The full-read baseline recorded above no longer suppresses a redundant full read.
        again = tools.execute("read_file", {"file_path": os.path.abspath("m.txt")})
        assert_true("Skipped re-opening" not in again, f"full read always served: {again}")

    _in_tmp(body)


def test_read_file_ids_header_only_on_first_open() -> None:
    def body() -> None:
        Path("m.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
        tools = HashlineTools()
        first = tools.execute(
            "read_file", {"file_path": os.path.abspath("m.txt"), "offset": 0, "limit": 2}
        )
        assert_true("(ids: each line is" in first, f"first open carries the ids header: {first}")
        second = tools.execute(
            "read_file", {"file_path": os.path.abspath("m.txt"), "offset": 2, "limit": 2}
        )
        assert_true("(ids: each line is" not in second, f"header not repeated: {second}")

    _in_tmp(body)


# --- execute_batch: coalesced parallel edits (one write, one diff, one check) ---


def test_execute_batch_replaces_apply_once() -> None:
    """Several replace calls that arrive together apply in one pass: all land,
    exactly one result carries the combined diff, the rest are terse."""
    def body() -> None:
        Path("m.py").write_text("a = 1\nb = 2\nc = 3\nd = 4\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})
        ops = [
            ("replace", {"file": "m.py", "id": _line_id(shown, 1), "content": "a = 10"}),
            ("replace", {"file": "m.py", "id": _line_id(shown, 2), "content": "b = 20"}),
            ("replace", {"file": "m.py", "id": _line_id(shown, 4), "content": "d = 40"}),
        ]
        results = tools.execute_batch(ops)
        assert_equal(len(results), 3, "one result per op")
        assert_equal(
            Path("m.py").read_text(encoding="utf-8"),
            "a = 10\nb = 20\nc = 3\nd = 40\n",
            "all three edits applied",
        )
        with_diff = [r for r in results if "Diff:" in r]
        assert_equal(len(with_diff), 1, f"exactly one combined diff: {results}")
        assert_true(
            "a = 10" in with_diff[0] and "d = 40" in with_diff[0],
            "combined diff spans every edit",
        )
        terse = [r for r in results if "Diff:" not in r]
        assert_true(
            all(r.startswith("Edited line") for r in terse),
            f"other ops get terse success lines: {terse}",
        )

    _in_tmp(body)


def test_execute_batch_writes_once() -> None:
    """A whole group is a single atomic write, not one per op."""
    import tools.files.hashline.tool_calling as tc

    def body() -> None:
        orig = tc.atomic_write_document
        calls: list[str] = []

        def spy(path, doc):
            calls.append(str(path))
            return orig(path, doc)

        tc.atomic_write_document = spy
        try:
            Path("m.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
            tools = HashlineTools()
            shown = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})
            ops = [
                ("replace", {"file": "m.py", "id": _line_id(shown, 1), "content": "a = 10"}),
                ("replace", {"file": "m.py", "id": _line_id(shown, 2), "content": "b = 20"}),
                ("replace", {"file": "m.py", "id": _line_id(shown, 3), "content": "c = 30"}),
            ]
            tools.execute_batch(ops)
            assert_equal(len(calls), 1, f"exactly one atomic write for the batch: {calls}")
        finally:
            tc.atomic_write_document = orig

    _in_tmp(body)


def test_execute_batch_mixed_ops_apply_in_order() -> None:
    """replace + delete + insert in one group apply bottom-to-top, so every
    op lands where the model's original ids pointed."""
    def body() -> None:
        Path("m.py").write_text("a = 1\nb = 2\nc = 3\nd = 4\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})
        ops = [
            ("replace", {"file": "m.py", "id": _line_id(shown, 1), "content": "a = 10"}),
            ("delete", {"file": "m.py", "id": _line_id(shown, 3)}),
            ("insert", {"file": "m.py", "id": _line_id(shown, 4), "position": "after", "content": "e = 5"}),
        ]
        results = tools.execute_batch(ops)
        assert_equal(len(results), 3, "one result per op")
        assert_equal(
            Path("m.py").read_text(encoding="utf-8"),
            "a = 10\nb = 2\nd = 4\ne = 5\n",
            "mixed ops applied correctly against the original snapshot",
        )

    _in_tmp(body)


def test_execute_batch_stale_op_errors_others_apply() -> None:
    """A stale anchor fails only its own op (plain Error, not flagged); the
    valid ops still apply."""
    def body() -> None:
        Path("m.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})
        ops = [
            ("replace", {"file": "m.py", "id": _line_id(shown, 1), "content": "a = 10"}),
            ("replace", {"file": "m.py", "id": "59ff", "content": "nope"}),  # no line 59
        ]
        results = tools.execute_batch(ops)
        assert_true(results[1].startswith("Error"), f"stale op errors: {results[1]}")
        assert_true(not results[1].startswith(TOOL_ATTENTION_MARKER), "stale error is not flagged")
        assert_equal(
            Path("m.py").read_text(encoding="utf-8"),
            "a = 10\nb = 2\nc = 3\n",
            "only the valid edit changed the file",
        )

    _in_tmp(body)


def test_execute_batch_overlapping_ops_first_wins() -> None:
    """Two ops on the same line: the first applies, the second reports an
    overlap instead of silently clobbering."""
    def body() -> None:
        Path("m.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})
        same = _line_id(shown, 2)
        ops = [
            ("replace", {"file": "m.py", "id": same, "content": "b = 20"}),
            ("replace", {"file": "m.py", "id": same, "content": "b = 99"}),
        ]
        results = tools.execute_batch(ops)
        assert_true("overlaps another edit" in results[1], f"second op flagged: {results[1]}")
        assert_equal(
            Path("m.py").read_text(encoding="utf-8"),
            "a = 1\nb = 20\nc = 3\n",
            "first op wins the overlap",
        )

    _in_tmp(body)


def test_execute_batch_broken_result_is_marked() -> None:
    """When the group leaves the file unparseable, the combined result (on the
    last applicable op) carries the attention marker and the syntax block."""
    def body() -> None:
        Path("m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        tools = HashlineTools()
        shown = tools.execute("read_file", {"file_path": os.path.abspath("m.py")})
        ops = [
            ("replace", {"file": "m.py", "id": _line_id(shown, 2), "content": "    return 2"}),
            ("replace", {"file": "m.py", "id": _line_id(shown, 1), "content": "def f("}),  # breaks parse
        ]
        results = tools.execute_batch(ops)
        assert_true(
            results[1].startswith(TOOL_ATTENTION_MARKER),
            f"broken batch flags attention on the combined result: {results[1]!r}",
        )
        assert_true("Syntax check (python)" in results[1], "combined result names the break")

    _in_tmp(body)


# --- empty-file population (editor has no bash; a 0-byte stub is fillable) ---


def test_insert_into_empty_file_populates_it() -> None:
    """An existing empty file can be filled by insert without a prior read_file,
    and any id is accepted; the file is then registered for follow-up edits."""
    def body() -> None:
        Path("new.py").write_text("", encoding="utf-8")  # 0-byte stub
        tools = HashlineTools()
        result = tools.execute("insert", {"file": "new.py", "id": "start", "content": "x = 1"})
        clean = (
            result[len(TOOL_ATTENTION_MARKER):]
            if result.startswith(TOOL_ATTENTION_MARKER)
            else result
        )
        assert_true("denied" not in clean, f"empty-file insert is not gated: {clean}")
        assert_equal(Path("new.py").read_text(encoding="utf-8"), "x = 1\n", "file populated")
        # The file is now registered, so a normal follow-up edit works.
        shown = tools.execute("read_file", {"file_path": os.path.abspath("new.py")})
        edited = tools.execute("replace", {"file": "new.py", "id": _line_id(shown, 1), "content": "x = 2"})
        assert_true(edited.startswith("Edited line 1"), f"follow-up edit works: {edited}")
        assert_equal(Path("new.py").read_text(encoding="utf-8"), "x = 2\n", "follow-up edit applied")

    _in_tmp(body)


def test_replace_on_empty_file_populates_it() -> None:
    """replace on an empty file is treated as 'set the content' — any id works."""
    def body() -> None:
        Path("new.py").write_text("", encoding="utf-8")
        tools = HashlineTools()
        result = tools.execute("replace", {"file": "new.py", "id": "1af", "content": "y = 1"})
        assert_true("denied" not in result, f"empty-file replace is not gated: {result}")
        assert_equal(Path("new.py").read_text(encoding="utf-8"), "y = 1\n", "file populated via replace")

    _in_tmp(body)


def test_empty_file_populate_splits_multiline_content() -> None:
    def body() -> None:
        Path("new.py").write_text("", encoding="utf-8")
        tools = HashlineTools()
        tools.execute("insert", {"file": "new.py", "id": "start", "content": "def f():\n    return 1"})
        assert_equal(
            Path("new.py").read_text(encoding="utf-8"),
            "def f():\n    return 1\n",
            "multi-line content becomes the file's lines",
        )

    _in_tmp(body)


def test_mutation_on_nonexistent_file_is_still_denied() -> None:
    """The empty-file exception is only for files that EXIST — a missing file
    still routes creation through bash (the gate denies the guess)."""
    def body() -> None:
        tools = HashlineTools()
        result = tools.execute("insert", {"file": "ghost.py", "id": "start", "content": "z = 1"})
        assert_true(result.startswith("Error"), f"missing file is an error: {result}")
        assert_true(not Path("ghost.py").exists(), "no file was created by the mutation")

    _in_tmp(body)


def main() -> None:
    test_xxh32_matches_hashline_reference_values()
    test_line_ref_has_no_separator_and_round_trips()
    test_schema_advertises_read_file_only()
    test_read_file_edit_insert_delete_flow()
    test_mutation_denied_without_begin_transaction()
    test_noop_edit_reports_no_changes()
    test_insert_range_id_reports_single_id_guidance()
    test_stale_id_reports_context_and_flips_on_edit()
    test_id_parser_rejects_copied_line_rows()
    test_query_region_and_search_document_match_hashline_names()
    test_document_stats_and_stream_replace_line()
    test_document_content_len_uses_utf8_bytes_and_loads_meta()
    test_start_end_keyword_ids_resolve_positionally()
    test_read_file_relative_path_rejected_force_accepted()
    test_read_file_offset_limit_window()
    test_read_file_offset_past_end_errors()
    test_read_file_offset_without_limit_reads_to_end()
    test_read_file_string_offset_limit_coerced()
    test_paged_read_does_not_set_or_consult_f1_baseline()
    test_read_file_ids_header_only_on_first_open()
    test_execute_batch_replaces_apply_once()
    test_execute_batch_writes_once()
    test_execute_batch_mixed_ops_apply_in_order()
    test_execute_batch_stale_op_errors_others_apply()
    test_execute_batch_overlapping_ops_first_wins()
    test_execute_batch_broken_result_is_marked()
    test_insert_into_empty_file_populates_it()
    test_replace_on_empty_file_populates_it()
    test_empty_file_populate_splits_multiline_content()
    test_mutation_on_nonexistent_file_is_still_denied()
    print("hashline tool tests passed")


if __name__ == "__main__":
    main()
