"""Tests for tools/compress — the tool-output compression layer.

Run directly: ``python qwowl35/tests/compressor_test.py``. Pure functions
only: no shell, no network, no filesystem.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.compress import (  # noqa: E402
    MARKER_PREFIX,
    MIN_CHARS_TO_COMPRESS,
    compress_requested,
    compress_tool_result,
    strip_compress_arg,
)
from tools.compress import detect as detect_mod  # noqa: E402
from tools.compress import rerank as rerank_mod  # noqa: E402
from tools.compress.comments import prune_comment_lines  # noqa: E402
from tools.compress.detect import detect_language  # noqa: E402
from tools.compress.rerank import BM25Scorer  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


# --- log strategy (run_shell_command) ---------------------------------------


def test_log_identical_lines_collapse_with_count() -> None:
    text = "\n".join(["the same log line over and over"] * 200)
    result = compress_tool_result("run_shell_command", {}, text)
    assert_true(result.was_compressed, "repetitive log compressed")
    assert_true("[line repeated × 200]" in result.text, result.text[:200])
    assert_equal(
        result.text.count("the same log line over and over"), 1, "one copy kept"
    )
    assert_true(result.saved_chars > 0.8 * result.original_chars, "big saving")


def test_log_protected_lines_survive_collapse() -> None:
    errors = [f"ERROR: disk failure at block {i}" for i in range(10)]
    filler = ["copying one more chunk of data"] * 200
    text = "\n".join(filler[:100] + errors + filler[100:])
    result = compress_tool_result("run_shell_command", {}, text)
    assert_true(result.was_compressed, "filler collapsed")
    for line in errors:
        assert_true(line in result.text, f"error line kept: {line}")


def test_log_near_duplicates_cluster() -> None:
    lines = [f"request {i} served from 0x{i:04x} in {i * 3}ms" for i in range(80)]
    text = "\n".join(lines)
    result = compress_tool_result("run_shell_command", {}, text)
    assert_true(result.was_compressed, "near-dup log compressed")
    assert_true("similar lines]" in result.text, result.text[:300])
    assert_true(lines[0] in result.text, "first line kept")
    assert_true(lines[-1] in result.text, "last line kept")


def test_log_head_tail_keeps_exit_code_stderr_and_needle() -> None:
    body = [f"step {chr(65 + i % 26)}{chr(65 + (i // 26) % 26)} finished cleanly - phase {chr(97 + i % 7)}" for i in range(400)]
    body[200] = "ERROR: the mid-log needle"
    text = "\n".join(body) + "\nstderr:\nsome stray line\n\nExit code: 1"
    result = compress_tool_result("run_shell_command", {}, text)
    assert_true(result.was_compressed, "long distinct log compressed")
    assert_true("lines elided]" in result.text, "elision marked")
    assert_true("ERROR: the mid-log needle" in result.text, "protected needle kept")
    assert_true("stderr:" in result.text, "stderr section kept")
    assert_true("Exit code: 1" in result.text, "exit code kept")


def test_log_json_output_passes_through() -> None:
    text = json.dumps([{"id": i, "name": "row", "flag": True} for i in range(100)], indent=1)
    assert_true(len(text) > MIN_CHARS_TO_COMPRESS, "test input big enough")
    result = compress_tool_result("run_shell_command", {}, text)
    assert_true(not result.was_compressed, "JSON output untouched in v1")
    assert_equal(result.text, text, "JSON byte-identical")


# --- code strategy (inspect_file) ---------------------------------------------


def test_code_realistic_source_is_untouched() -> None:
    lines = [f"def handler_{i}(x):\n    return x + {i}\n" for i in range(120)]
    text = "\n".join(lines)
    result = compress_tool_result("inspect_file", {}, text)
    assert_true(not result.was_compressed, "normal source stays verbatim")
    assert_equal(result.text, text, "source byte-identical")


def test_code_flagrant_repeats_collapse_with_count() -> None:
    filler = [f"row_{i} = load({i})" for i in range(80)]
    repeated = ["INSERT INTO t VALUES (0);"] * 40
    text = "\n".join(filler + repeated + filler)
    result = compress_tool_result("inspect_file", {}, text)
    assert_true(result.was_compressed, "generated-file repeats compressed")
    assert_true("[line repeated × 40]" in result.text, result.text[-300:])


def test_code_paging_header_is_preserved() -> None:
    header = "Showing lines 1-2000 of 5000 total lines.\n\n---\n\n"
    text = header + "\n".join(["SAME LINE OF DATA IN A BIG FILE"] * 120)
    result = compress_tool_result("inspect_file", {}, text)
    assert_true(result.was_compressed, "repetitive body compressed")
    assert_true(result.text.startswith(header), "paging header byte-identical")


# --- comment pruning (detection → tree-sitter) ----------------------------------


def _no_magika(func) -> None:
    """Force the extension-fallback lane regardless of whether magika is installed."""
    saved = (detect_mod._MAGIKA, detect_mod._MAGIKA_FAILED)
    detect_mod._MAGIKA, detect_mod._MAGIKA_FAILED = None, True
    try:
        func()
    finally:
        detect_mod._MAGIKA, detect_mod._MAGIKA_FAILED = saved


class _Obj:
    """Attribute bag for stub Magika result shapes."""

    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)


class _StubMagika:
    def __init__(self, result) -> None:
        self._result = result

    def identify_bytes(self, data) -> object:
        return self._result


def _with_magika_stub(result, func) -> None:
    saved = (detect_mod._MAGIKA, detect_mod._MAGIKA_FAILED)
    detect_mod._MAGIKA, detect_mod._MAGIKA_FAILED = _StubMagika(result), False
    try:
        func()
    finally:
        detect_mod._MAGIKA, detect_mod._MAGIKA_FAILED = saved


def test_code_python_comment_run_pruned_end_to_end() -> None:
    def run() -> None:
        comments = [f"# commentary line {i} explaining the surrounding module" for i in range(60)]
        code = [f"def handler_{i}(x):\n    return x + {i}" for i in range(40)]
        text = "\n".join(comments) + "\n" + "\n".join(code)
        result = compress_tool_result("inspect_file", {"file_path": "/w/app.py"}, text)
        assert_true(result.was_compressed, "comment-heavy python compressed")
        assert_true(comments[0] not in result.text, "comment lines gone")
        assert_true("[… " not in result.text, "no per-run inline marker")
        assert_true("60 comment lines" in result.text, result.text[:300])
        for line in code:
            assert_true(line in result.text, "code lines intact")
        assert_true(MARKER_PREFIX in result.text, "global marker present")

    _no_magika(run)


def test_code_appended_lsp_diagnostics_survive_pruning() -> None:
    # inspect_file may append an "LSP diagnostics (…)" section after the file
    # body; the framework carves that section off BEFORE any strategy runs and
    # re-attaches it verbatim after the marker, so no code-cutting logic ever
    # touches a diagnostics line and the final text still ends with a
    # canonical trailing section.
    def run() -> None:
        comments = [f"# commentary line {i} explaining the surrounding module" for i in range(60)]
        code = [f"def handler_{i}(x):\n    return x + {i}" for i in range(40)]
        diagnostics = [
            "LSP diagnostics (python, lsp) — 2 error(s), 1 warning(s):",
            "- line 12, col 5: Undefined name 'foo' (Pyflakes)",
            "- line 30, col 1: Undefined name 'bar' (Pyflakes)",
            "Warnings (not blocking):",
            "- line 3, col 1: 'os' imported but unused (Pyflakes)",
        ]
        section = "\n".join(diagnostics)
        text = "\n".join(comments) + "\n" + "\n".join(code) + "\n\n" + section
        result = compress_tool_result("inspect_file", {"file_path": "/w/app.py"}, text)
        assert_true(result.was_compressed, "comment-heavy python compressed")
        assert_true(comments[0] not in result.text, "comment lines gone")
        assert_true(result.text.endswith(section), "section re-attached verbatim, last")
        marker_at = result.text.index(MARKER_PREFIX)
        assert_true(
            marker_at < result.text.index("LSP diagnostics ("),
            "marker describes the body and sits before the diagnostics section",
        )

    _no_magika(run)


def test_code_diagnostics_section_excluded_from_marker_accounting() -> None:
    # The marker's "X of Y chars elided" must describe the compressed BODY
    # only: the carved diagnostics section is neither elided nor elidable.
    def run() -> None:
        comments = [f"# commentary line {i} explaining the surrounding module" for i in range(60)]
        code = [f"def handler_{i}(x):\n    return x + {i}" for i in range(40)]
        body = "\n".join(comments) + "\n" + "\n".join(code)
        section = (
            "LSP diagnostics (python, lsp) — 1 error(s), 0 warning(s):\n"
            "- line 12, col 5: Undefined name 'foo' (Pyflakes)"
        )
        with_section = compress_tool_result(
            "inspect_file", {"file_path": "/w/app.py"}, body + "\n\n" + section
        )
        body_only = compress_tool_result("inspect_file", {"file_path": "/w/app.py"}, body)
        assert_true(with_section.was_compressed and body_only.was_compressed, "both compressed")
        marker = body_only.text[body_only.text.index(MARKER_PREFIX) :]
        assert_true(marker in with_section.text, "identical marker numbers with/without section")

    _no_magika(run)


def test_prune_c_line_and_block_comments() -> None:
    src = "\n".join(
        ["// header note one", "// header note two", "// header note three"]
        + ["#include <stdio.h>"]
        + ["/*", " * block documentation line", " * more documentation", " */"]
        + ["int main(void) {", "    return 0;", "}"]
    )
    out, count = prune_comment_lines(src, "c")
    assert_equal(count, 7, "both runs (3 + 4 lines) combined into one total")
    assert_true("[… " not in out, "no per-run inline marker")
    assert_true("#include <stdio.h>" in out, "code between runs kept")
    assert_true("// header note one" not in out, "line comments gone")
    assert_true("block documentation" not in out, "block comment gone")
    assert_true("int main(void) {" in out, "code kept")


def test_prune_string_literal_hash_kept() -> None:
    src = "\n".join(
        ['s = "# not a comment"', 'u = "http://example.com"', "value = s + u"]
    )
    out, count = prune_comment_lines(src, "python")
    assert_equal(out, src, "string literals byte-identical")
    assert_equal(count, 0, "nothing elided")


def test_prune_trailing_comment_kept() -> None:
    src = "\n".join(["x = 1  # note about x", "y = 2  # note about y", "z = x + y"])
    out, count = prune_comment_lines(src, "python")
    assert_equal(out, src, "trailing comments never alter code lines")
    assert_equal(count, 0, "nothing elided")


def test_prune_shebang_elided() -> None:
    src = "\n".join(
        ["#!/usr/bin/env python3"]
        + [f"# preamble comment number {i} in the header block" for i in range(5)]
        + ["print('hi')"]
    )
    out, count = prune_comment_lines(src, "python")
    assert_true("#!/usr/bin/env python3" not in out, "shebang elided")
    assert_equal(count, 6, "shebang + preamble block elided")
    assert_true("print('hi')" in out, "code kept")


def test_prune_single_comment_line_elided() -> None:
    src = "\n".join(["x = 1", "# a lone explanatory comment", "y = 2"])
    out, count = prune_comment_lines(src, "python")
    assert_equal(out, "x = 1\n\ny = 2", "lone comment line blanked in place")
    assert_equal(count, 1, "one line elided")


def test_prune_python_docstrings_elided() -> None:
    src = "\n".join(
        [
            '"""Module docstring',
            "spanning multiple lines.",
            '"""',
            "import sys",
            "",
            "class Greeter:",
            '    """Class docstring."""',
            "",
            "    def greet(self, name):",
            "        '''Function docstring.'''",
            "        return f'hi {name}'",
        ]
    )
    out, count = prune_comment_lines(src, "python")
    assert_true("Module docstring" not in out, "module docstring elided")
    assert_true("Class docstring" not in out, "class docstring elided")
    assert_true("Function docstring" not in out, "function docstring elided")
    assert_equal(count, 5, "all docstring lines elided")
    assert_true("import sys" in out, "code kept")
    assert_true("class Greeter:" in out, "class kept")
    assert_true("        return f'hi {name}'" in out, "body kept")


def test_prune_string_assignment_kept() -> None:
    src = "\n".join(
        [
            'BANNER = """',
            "multi-line string assigned to a name",
            '"""',
            "print(BANNER)",
        ]
    )
    out, count = prune_comment_lines(src, "python")
    assert_equal(out, src, "assigned strings are not docstrings")
    assert_equal(count, 0, "nothing elided")


def test_prune_bare_string_statement_elided() -> None:
    src = "\n".join(["x = 1", '"stray bare string, a no-op statement"', "y = 2"])
    out, count = prune_comment_lines(src, "python")
    assert_equal(out, "x = 1\n\ny = 2", "bare string statement blanked in place")
    assert_equal(count, 1, "one line elided")


def test_prune_preserves_line_numbering() -> None:
    # THE line-coherence contract: pruning must never shift a code line's
    # number. Deleting comment blocks once moved `def train` from line 127 to
    # 92 in the model's view — it then dismissed correct LSP findings whose
    # real-file line numbers pointed at "wrong" code and reported a clean file.
    src = "\n".join(
        ['"""Module docstring', "two lines long.", '"""']
        + [f"# banner comment {i}" for i in range(10)]
        + ["import os", ""]
        + [f"# section note {i}" for i in range(5)]
        + ["def train(learning_rate=1.0):", "    return learning_rate"]
    )
    out, count = prune_comment_lines(src, "python")
    src_lines = src.split("\n")
    out_lines = out.split("\n")
    assert_equal(len(out_lines), len(src_lines), "line count identical")
    assert_equal(count, 18, "docstring + both comment runs blanked")
    for needle in ("import os", "def train(learning_rate=1.0):"):
        assert_equal(
            out_lines.index(needle) + 1,
            src_lines.index(needle) + 1,
            f"{needle!r} keeps its real line number",
        )


def test_code_blanked_runs_survive_collapse_end_to_end() -> None:
    # A pruned 12-line comment block becomes 12 blank lines; the repeat
    # collapse (CODE_REPEAT_MIN=8) must NOT fold them into a marker, or the
    # numbering the pruning just preserved would shift right back.
    def run() -> None:
        src_lines = (
            [f"# license header line {i} of the generated banner" for i in range(12)]
            + ["def handler(x):", "    return x + 1"]
        )
        text = "\n".join(src_lines)
        result = compress_tool_result("inspect_file", {"file_path": "/w/app.py"}, text)
        assert_true(result.was_compressed, "comment block compressed")
        body = result.text.split(MARKER_PREFIX, 1)[0]
        body_lines = body.split("\n")
        assert_true("[line repeated" not in body, "blank run not collapsed")
        assert_equal(
            body_lines.index("def handler(x):") + 1,
            src_lines.index("def handler(x):") + 1,
            "code keeps its real line number end-to-end",
        )

    _no_magika(run)


def test_prune_all_supported_languages() -> None:
    """Single-line AND multi-line comments elide in every supported grammar."""
    cases = {
        "python": ("# single note\n# second note\nx = 1\n", ["x = 1"]),
        "javascript": ("#!/usr/bin/env node\n// note\n/* block\n spanning */\nlet x = 1;\n", ["let x = 1;"]),
        "typescript": ("// note\n/* block\n spanning */\nconst x = 1;\n", ["const x = 1;"]),
        "tsx": ("// note\n/* block */\nconst x = 1;\n", ["const x = 1;"]),
        "c": ("// note\n/* block\n spanning */\nint x;\n", ["int x;"]),
        "cpp": ("// note\n/* block */\nint x;\n", ["int x;"]),
        "rust": ("// note\n/* block */\n/// doc line\n//! inner doc\nfn f() {}\n", ["fn f() {}"]),
        "go": ("// note\n/* block\n spanning */\npackage main\n", ["package main"]),
        "java": ("// note\n/* block */\n/** javadoc\n line */\nclass A {}\n", ["class A {}"]),
        "ruby": ("#!/usr/bin/env ruby\n# note\n=begin\nblock body\n=end\nx = 1\n", ["x = 1"]),
        "bash": ("#!/bin/sh\n# note\nx=1\n", ["x=1"]),
        "yaml": ("# note\nkey: value\n", ["key: value"]),
        "toml": ("# note\nkey = 1\n", ["key = 1"]),
    }
    from tools.syntax.checker import _get_parser

    for language, (src, code_lines) in cases.items():
        if _get_parser(language) is None:
            continue  # guarded dep philosophy: missing grammar is a no-op elsewhere too
        out, count = prune_comment_lines(src, language)
        expected_elided = len([ln for ln in src.split("\n") if ln.strip()]) - len(code_lines)
        assert_equal(count, expected_elided, f"{language}: all comment lines elided")
        for line in code_lines:
            assert_true(line in out, f"{language}: code line kept: {line}")
        for word in ("note", "block", "doc", "javadoc"):
            assert_true(word not in out, f"{language}: comment text gone ({word})")


def test_prune_unknown_language_is_noop() -> None:
    src = "# whatever\n# whatever\ncode()"
    out, count = prune_comment_lines(src, "nosuchlang")
    assert_equal(out, src, "unknown grammar fails open")
    assert_equal(count, 0, "nothing elided")


def test_detect_magika_stub_shapes() -> None:
    old_shape = _Obj(output=_Obj(ct_label="python", score=0.99))
    _with_magika_stub(
        old_shape,
        lambda: assert_equal(detect_language("def f(): pass"), "python", "0.5.x shape"),
    )
    new_shape = _Obj(prediction=_Obj(output=_Obj(label="python"), score=0.99))
    _with_magika_stub(
        new_shape,
        lambda: assert_equal(detect_language("def f(): pass"), "python", "0.6 shape"),
    )
    unmapped = _Obj(output=_Obj(ct_label="markdown", score=0.99))
    _with_magika_stub(
        unmapped,
        lambda: assert_equal(
            detect_language("# title", "/w/notes.py"), None, "confident unmapped wins over extension"
        ),
    )
    unsure = _Obj(output=_Obj(ct_label="python", score=0.3))
    _with_magika_stub(
        unsure,
        lambda: assert_equal(
            detect_language("text", "/w/app.py"), "python", "low score falls back to extension"
        ),
    )


def test_detect_magika_absent_falls_back_to_extension() -> None:
    def run() -> None:
        assert_equal(detect_language("code", "/w/app.py"), "python", ".py fallback")
        assert_equal(detect_language("code", "/w/run.sh"), "bash", ".sh fallback")
        assert_equal(detect_language("code", ""), None, "no path, no detection")

    _no_magika(run)


def test_code_paged_window_comments_pruned() -> None:
    def run() -> None:
        header = "Showing lines 100-160 of 900 total lines.\n\n---\n\n"
        # Window cut mid-function: unbalanced braces make an error tree.
        body = "\n".join(
            ["    return partial_result +"]
            + [f"# window commentary line {i} carried from the original file" for i in range(40)]
            + [f"x_{i} = compute_{i}()" for i in range(20)]
        )
        result = compress_tool_result(
            "inspect_file", {"file_path": "/w/big.py"}, header + body
        )
        assert_true(result.was_compressed, "paged window compressed")
        assert_true(result.text.startswith(header), "paging header byte-identical")
        assert_true("[… " not in result.text, "no per-run inline marker")
        assert_true("40 comment lines" in result.text, result.text[:400])

    _no_magika(run)


def test_code_pruning_composes_with_repeat_collapse() -> None:
    def run() -> None:
        text = "\n".join(
            [f"# generated-file banner line number {i} of the header" for i in range(30)]
            + ["ROW = (0, 0, 0, 0)"] * 40
            + [f"col_{i} = {i}" for i in range(30)]
        )
        result = compress_tool_result("inspect_file", {"file_path": "/w/gen.py"}, text)
        assert_true(result.was_compressed, "mixed file compressed")
        assert_true("30 comment lines" in result.text, "comment pass ran, reported once")
        assert_true("[line repeated × 40]" in result.text, "repeat collapse ran")

    _no_magika(run)


def test_code_small_file_comments_pruned_despite_size_gate() -> None:
    """inspect_file bypasses MIN_CHARS_TO_COMPRESS: pruning applies to every file."""

    def run() -> None:
        text = "\n".join(
            ["def greet(name):"]
            + [
                f"    # note {i} about the greeting logic below and its edge cases"
                for i in range(3)
            ]
            + ["    return f'hi {name}'"]
        )
        assert_true(len(text) < MIN_CHARS_TO_COMPRESS, "fixture stays under the size gate")
        result = compress_tool_result("inspect_file", {"file_path": "/w/tiny.py"}, text)
        assert_true(result.was_compressed, "small file still compressed")
        assert_true("note 0 about" not in result.text, "comment lines gone")
        assert_true("def greet(name):" in result.text, "code kept")
        assert_true("3 comment lines" in result.text, result.text)

    _no_magika(run)


# --- grep strategy ------------------------------------------------------------


def _grep_output(per_file: int, files: int = 1) -> str:
    total = per_file * files
    parts = [f'Found {total} matches for pattern "needle" in the workspace directory:', "---"]
    for f in range(files):
        parts.append(f"File: pkg/module_{f}.py")
        parts.extend(f"L{i + 1}: value = needle_{f}_{i}  # a reasonably long matching line" for i in range(per_file))
        parts.append("---")
    return "\n".join(parts).rstrip("-\n").rstrip() + "\n---"


def test_grep_caps_matches_per_file() -> None:
    text = _grep_output(per_file=60)
    result = compress_tool_result("grep_search", {}, text)
    assert_true(result.was_compressed, "60-match grep compressed")
    assert_true('Found 60 matches for pattern "needle"' in result.text, "header kept")
    assert_true("(+52 more matches in this file)" in result.text, result.text[:400])
    import re

    kept = [line for line in result.text.splitlines() if re.match(r"L\d+: ", line)]
    assert_equal(len(kept), 8, "eight matches kept")


def test_grep_non_conforming_text_untouched() -> None:
    text = "x" * 3000
    result = compress_tool_result("grep_search", {}, text)
    assert_true(not result.was_compressed, "non-grep shape untouched")
    assert_equal(result.text, text, "byte-identical")


# --- web strategy -------------------------------------------------------------


def test_web_repeated_paragraphs_dedup_and_header_survives() -> None:
    header = "Content from https://example.com (you wanted: docs):\n\n"
    banner = (
        "Subscribe to our newsletter for updates, offers, product news, community "
        "highlights, event invitations, partner promotions, survey requests, and "
        "the occasional very long announcement from the marketing department that "
        "repeats at the top and the bottom of every page on this entire site."
    )
    article = "\n\n".join(
        f"Paragraph {i}: this article text carries the actual signal the model wants to read."
        for i in range(20)
    )
    text = header + banner + "\n\n" + article + "\n\n" + banner + "\n\n" + banner
    result = compress_tool_result("web_fetch", {}, text)
    assert_true(result.was_compressed, "repeated banners compressed")
    assert_true(result.text.startswith(header), "header preserved")
    assert_equal(result.text.count("Subscribe to our newsletter"), 1, "banner deduped")
    assert_true("repeated paragraphs deduplicated" in result.text, "dedup noted")


def test_web_nav_link_runs_collapse() -> None:
    nav = "\n".join(
        ["Home", "About Us", "Products", "Pricing", "Docs", "Blog"]
        + [f"Category {chr(65 + i % 26)}{chr(65 + i // 26)}" for i in range(74)]
    )
    body = "\n\n".join(
        f"Section {i}: long-form content sentence that ends properly and matters. " * 3
        for i in range(12)
    )
    text = "Content from https://example.com:\n\n" + nav + "\n\n" + body
    result = compress_tool_result("web_fetch", {}, text)
    assert_true(result.was_compressed, "nav boilerplate compressed")
    assert_true("short nav/link lines]" in result.text, result.text[:400])
    assert_true("Home" in result.text, "first nav lines kept")


# --- web rerank lane (query-aware) ---------------------------------------------


def _web_page(query_paragraphs: list[str], filler_count: int = 60) -> str:
    header = "Content from https://example.com (you wanted: okapi bm25 parameters):\n\n"
    filler = [
        f"Section {i}: this long-form passage discusses gardening, travel plans, "
        f"cooking recipe number {i}, and the history of local festivals in detail, "
        "with full sentences that end properly and carry no search terms at all. "
        f"It continues with an anecdote about market day number {i}, the seasonal "
        "harvest, and a neighborly dispute over fence paint that resolved amicably."
        for i in range(filler_count)
    ]
    middle = filler[: filler_count // 2] + query_paragraphs + filler[filler_count // 2 :]
    return header + "\n\n".join(middle)


def _with_bm25_default(func) -> None:
    rerank_mod._DEFAULT_SCORER = BM25Scorer()
    try:
        func()
    finally:
        rerank_mod._DEFAULT_SCORER = None


def test_web_query_rerank_keeps_relevant_chunks() -> None:
    def run() -> None:
        relevant = [
            "The okapi bm25 ranking function typically uses the parameters k1=1.5 "
            "and b=0.75, controlling term saturation and length normalization.",
            "Tuning bm25: raising k1 increases term-frequency influence while the b "
            "parameter scales the document-length penalty in okapi scoring.",
        ]
        text = _web_page(relevant)
        result = compress_tool_result(
            "web_fetch",
            {"url": "https://example.com", "prompt": "okapi bm25 k1 b parameters"},
            text,
        )
        assert_true(result.was_compressed, "reranked page compressed")
        assert_true(
            result.text.startswith("Content from https://example.com"), "header preserved"
        )
        for paragraph in relevant:
            assert_true(paragraph in result.text, "relevant paragraph kept verbatim")
        assert_true(
            "not relevant to the query, elided]" in result.text, "gap marker present"
        )
        assert_true("scorer=bm25]" in result.text, "summary names the scorer")
        assert_true(
            result.text.index(relevant[0]) < result.text.index(relevant[1]),
            "original order kept",
        )

    _with_bm25_default(run)


def test_web_no_query_skips_rerank() -> None:
    def run() -> None:
        text = _web_page(["The okapi bm25 function uses k1 and b."])
        result = compress_tool_result("web_fetch", {"url": "https://example.com"}, text)
        assert_true("not relevant to the query" not in result.text, "no rerank markers")
        assert_true("scorer=" not in result.text, "no scorer line without a query")

    _with_bm25_default(run)


def test_web_rerank_is_idempotent() -> None:
    def run() -> None:
        text = _web_page(["The okapi bm25 function uses k1=1.5 and b=0.75 parameters."])
        args = {"url": "https://example.com", "prompt": "okapi bm25 k1 b parameters"}
        once = compress_tool_result("web_fetch", args, text)
        assert_true(once.was_compressed, "first pass compressed")
        twice = compress_tool_result("web_fetch", args, once.text)
        assert_true(not twice.was_compressed, "second pass is a no-op")
        assert_equal(twice.text, once.text, "byte-identical on re-compress")

    _with_bm25_default(run)


def test_web_rerank_kwarg_off_disables_the_lane() -> None:
    def run() -> None:
        text = _web_page(["The okapi bm25 function uses k1=1.5 and b=0.75 parameters."])
        args = {"url": "https://example.com", "prompt": "okapi bm25 k1 b parameters"}
        result = compress_tool_result("web_fetch", args, text, rerank=False)
        assert_true("not relevant to the query" not in result.text, "rerank lane off")
        assert_true("scorer=" not in result.text, "no scorer line when disabled")

    _with_bm25_default(run)


# --- framework ----------------------------------------------------------------


def test_small_results_are_skipped() -> None:
    text = "short output\n" * 10
    result = compress_tool_result("run_shell_command", {}, text)
    assert_true(not result.was_compressed, "small result untouched")
    assert_equal(result.text, text, "byte-identical")


def test_error_results_are_skipped() -> None:
    text = "Error: something broke\n" + "the same line\n" * 300
    result = compress_tool_result("run_shell_command", {}, text)
    assert_true(not result.was_compressed, "error result untouched")


def test_non_compressible_tools_are_skipped() -> None:
    text = "entry\n" * 1000
    for name in ("list_directory", "glob", "edit", "resume", "plan"):
        result = compress_tool_result(name, {}, text)
        assert_true(not result.was_compressed, f"{name} untouched")


def test_marker_format_and_recovery_hint() -> None:
    text = "\n".join(["the same log line over and over"] * 200)
    result = compress_tool_result("run_shell_command", {}, text)
    assert_true(MARKER_PREFIX in result.text, "marker present")
    assert_true(result.text.endswith("]"), "marker is the final line")
    assert_true("compress:false" in result.text, "recovery hint present")
    assert_true(str(result.original_chars) in result.text, "original size reported")


def test_compression_is_idempotent() -> None:
    text = "\n".join(["the same log line over and over"] * 200)
    once = compress_tool_result("run_shell_command", {}, text)
    twice = compress_tool_result("run_shell_command", {}, once.text)
    assert_true(not twice.was_compressed, "already-marked text not recompressed")
    assert_equal(twice.text, once.text, "second pass is a no-op")


def test_tiny_savings_return_the_original() -> None:
    distinct = "\n".join(f"unique line {chr(65 + i % 26)}{chr(65 + (i // 26) % 26)} with steady content" for i in range(60))
    text = distinct + "\n" + "dup line\n" * 3
    result = compress_tool_result("run_shell_command", {}, text)
    assert_true(not result.was_compressed, "sub-threshold saving discarded")
    assert_equal(result.text, text, "original returned")
    assert_true(MARKER_PREFIX not in result.text, "no marker on skipped result")


def test_compress_requested_handles_bools_and_strings() -> None:
    assert_true(compress_requested({}), "missing means yes")
    assert_true(compress_requested({"compress": True}), "True means yes")
    assert_true(compress_requested({"compress": "true"}), '"true" means yes')
    assert_true(not compress_requested({"compress": False}), "False means no")
    assert_true(not compress_requested({"compress": "false"}), '"false" means no')
    assert_true(not compress_requested({"compress": " False "}), "padded string handled")


def test_strip_compress_arg_does_not_mutate() -> None:
    args = {"command": "ls", "compress": False}
    stripped = strip_compress_arg(args)
    assert_equal(stripped, {"command": "ls"}, "compress removed")
    assert_equal(args, {"command": "ls", "compress": False}, "input not mutated")
    plain = {"command": "ls"}
    assert_true(strip_compress_arg(plain) == plain, "no-op without the key")


def main() -> None:
    test_log_identical_lines_collapse_with_count()
    test_log_protected_lines_survive_collapse()
    test_log_near_duplicates_cluster()
    test_log_head_tail_keeps_exit_code_stderr_and_needle()
    test_log_json_output_passes_through()
    test_code_realistic_source_is_untouched()
    test_code_flagrant_repeats_collapse_with_count()
    test_code_paging_header_is_preserved()
    test_code_python_comment_run_pruned_end_to_end()
    test_code_appended_lsp_diagnostics_survive_pruning()
    test_prune_c_line_and_block_comments()
    test_prune_string_literal_hash_kept()
    test_prune_trailing_comment_kept()
    test_prune_shebang_elided()
    test_prune_single_comment_line_elided()
    test_prune_python_docstrings_elided()
    test_prune_string_assignment_kept()
    test_prune_bare_string_statement_elided()
    test_prune_preserves_line_numbering()
    test_code_blanked_runs_survive_collapse_end_to_end()
    test_prune_all_supported_languages()
    test_prune_unknown_language_is_noop()
    test_detect_magika_stub_shapes()
    test_detect_magika_absent_falls_back_to_extension()
    test_code_paged_window_comments_pruned()
    test_code_pruning_composes_with_repeat_collapse()
    test_code_small_file_comments_pruned_despite_size_gate()
    test_grep_caps_matches_per_file()
    test_grep_non_conforming_text_untouched()
    test_web_repeated_paragraphs_dedup_and_header_survives()
    test_web_nav_link_runs_collapse()
    test_web_query_rerank_keeps_relevant_chunks()
    test_web_no_query_skips_rerank()
    test_web_rerank_is_idempotent()
    test_web_rerank_kwarg_off_disables_the_lane()
    test_small_results_are_skipped()
    test_error_results_are_skipped()
    test_non_compressible_tools_are_skipped()
    test_marker_format_and_recovery_hint()
    test_compression_is_idempotent()
    test_tiny_savings_return_the_original()
    test_compress_requested_handles_bools_and_strings()
    test_strip_compress_arg_does_not_mutate()
    print("compressor tests passed")


if __name__ == "__main__":
    main()
