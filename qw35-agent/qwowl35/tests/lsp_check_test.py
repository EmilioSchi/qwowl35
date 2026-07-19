"""Tests for the LSP diagnostics layer (tools/lsp) and the validation router.

No real language server is spawned: the multilspy plumbing is exercised with a
stubbed server handle, and the router/guard paths are driven so that every
"could not check" outcome falls back to the tree-sitter checker exactly like a
missing install would.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.lsp as lsp_pkg  # noqa: E402
from tools.lsp import diagnostics  # noqa: E402
from tools.syntax import validate  # noqa: E402
from tools.syntax.checker import check_file_structured  # noqa: E402
from tools.syntax.validate import Validation, validate_file, validation_report  # noqa: E402


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


BROKEN_PY = "a = 1\nb = 2\ndef f()\n    return 1\n"
CLEAN_PY = "def f():\n    return 1\n"


class _Bag:
    """Attribute bag for building stub Magika result shapes."""

    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)


class _StubMagika:
    """Stands in for a constructed Magika singleton, returning a fixed verdict.

    Matches the ``output.ct_label`` / ``output.score`` shape that
    detect._magika_identify probes for.
    """

    def __init__(self, label: str, score: float) -> None:
        self._label, self._score = label, score

    def identify_bytes(self, data: bytes) -> _Bag:
        return _Bag(output=_Bag(ct_label=self._label, score=self._score))


def test_supported_language_and_configure() -> None:
    assert_equal(diagnostics.supported_language("a.py"), "python", ".py → python")
    assert_equal(diagnostics.supported_language("a.tsx"), "typescript", ".tsx → typescript (not tsx)")
    assert_equal(diagnostics.supported_language("a.rs"), "rust", ".rs → rust")
    # No LSP backend in the installed multilspy release → always tree-sitter.
    assert_equal(diagnostics.supported_language("a.cpp"), None, ".cpp unmapped")
    assert_equal(diagnostics.supported_language("a.json"), None, ".json unmapped")
    diagnostics.configure(False)
    try:
        assert_equal(diagnostics.supported_language("a.py"), None, "disabled → None")
    finally:
        diagnostics.configure(True)


def test_supported_language_content_first() -> None:
    """With source, a confident Magika verdict wins; extension is the fallback."""
    import tools.compress.detect as detect_mod

    def with_stub(label, score, fn) -> None:
        saved = (detect_mod._MAGIKA, detect_mod._MAGIKA_FAILED)
        detect_mod._MAGIKA, detect_mod._MAGIKA_FAILED = _StubMagika(label, score), False
        try:
            fn()
        finally:
            detect_mod._MAGIKA, detect_mod._MAGIKA_FAILED = saved

    # Confident content label overrides even a known, conflicting extension.
    with_stub("typescript", 0.99, lambda: assert_equal(
        diagnostics.supported_language("a.js", "class X {}"),
        "typescript", "content 'typescript' overrides .js extension"))
    # Magika labels C# as "cs"; it must map to the csharp server, and here the
    # extension (.txt) has no LSP backend at all — content alone routes it.
    with_stub("cs", 0.99, lambda: assert_equal(
        diagnostics.supported_language("a.txt", "class X {}"),
        "csharp", "content 'cs' → csharp despite unsupported .txt extension"))
    # Below MAGIKA_MIN_SCORE → ignore the label, use the extension.
    with_stub("typescript", 0.10, lambda: assert_equal(
        diagnostics.supported_language("a.py", "x = 1"),
        "python", "low-confidence label falls back to .py extension"))
    # Confident but no LSP backend for the label → use the extension.
    with_stub("markdown", 0.99, lambda: assert_equal(
        diagnostics.supported_language("a.py", "# hi"),
        "python", "unmapped label falls back to .py extension"))
    # No source → Magika never consulted, pure extension path.
    with_stub("typescript", 0.99, lambda: assert_equal(
        diagnostics.supported_language("a.js"),
        "javascript", "no source → extension path, .js → javascript"))

    # Magika unavailable (failed import cached) → extension fallback.
    saved = (detect_mod._MAGIKA, detect_mod._MAGIKA_FAILED)
    detect_mod._MAGIKA, detect_mod._MAGIKA_FAILED = None, True
    try:
        assert_equal(
            diagnostics.supported_language("a.rs", "fn main() {}"),
            "rust", "magika absent → .rs extension fallback")
    finally:
        detect_mod._MAGIKA, detect_mod._MAGIKA_FAILED = saved


def test_router_prefers_lsp() -> None:
    original = lsp_pkg.lsp_check_file
    lsp_pkg.lsp_check_file = lambda path, source, root: (  # type: ignore[assignment]
        [(3, 1, "line 3, col 1: boom")],
        [(1, 1, "line 1, col 1: unused (linter)")],
    )
    try:
        v = validate_file("m.py", BROKEN_PY)
        assert_equal(v.label, "python, lsp", "LSP answered → lsp label")
        assert_equal(v.errors, [(3, 1, "line 3, col 1: boom")], "errors passed through")
        assert_equal(v.warnings, [(1, 1, "line 1, col 1: unused (linter)")], "warnings passed through")
        assert_true(v.checked, "an LSP answer counts as checked")
    finally:
        lsp_pkg.lsp_check_file = original  # type: ignore[assignment]


def test_router_falls_back_on_none() -> None:
    # "m.py" resolves inside the workspace root and .py is LSP-supported, so a
    # None answer is a DEGRADATION (booting/timeout/crash) — the fallback label
    # must say so instead of passing tree-sitter off as a full check.
    original = lsp_pkg.lsp_check_file
    lsp_pkg.lsp_check_file = lambda path, source, root: None  # type: ignore[assignment]
    try:
        v = validate_file("m.py", BROKEN_PY)
        assert_equal(
            v.label,
            "python — LSP unavailable, syntax-only",
            "in-root fallback announces the degradation",
        )
        assert_equal(v.errors, check_file_structured("m.py", BROKEN_PY), "identical to checker")
        assert_equal(v.warnings, [], "tree-sitter has no warnings")
        assert_true(v.checked, "tree-sitter parsed → checked")
        block = v.report()
        assert_true(block.startswith("Syntax check ("), f"TUI prefix contract holds: {block}")
    finally:
        lsp_pkg.lsp_check_file = original  # type: ignore[assignment]


def test_router_falls_back_when_disabled_or_import_fails() -> None:
    diagnostics.configure(False)
    try:
        v = validate_file("m.py", CLEAN_PY)
        assert_equal(v.label, "python", "disabled → tree-sitter label")
        assert_true(v.checked and not v.errors, "clean parse")
    finally:
        diagnostics.configure(True)
    original = lsp_pkg.supported_language

    def _boom(path):
        raise RuntimeError("import exploded")

    lsp_pkg.supported_language = _boom  # type: ignore[assignment]
    try:
        v = validate_file("m.py", CLEAN_PY)
        assert_equal(v.label, "python", "broken lsp layer → tree-sitter label")
    finally:
        lsp_pkg.supported_language = original  # type: ignore[assignment]


def test_outside_workspace_root_falls_back() -> None:
    # Tempdirs live outside the import-time workspace root, so lsp_check_file
    # bails before ever spawning a server — this is also why every existing
    # tempdir-based test keeps its tree-sitter labels.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.py"
        path.write_text(BROKEN_PY, encoding="utf-8")
        result = diagnostics.lsp_check_file(str(path), BROKEN_PY, validate._WORKSPACE_ROOT)
        assert_equal(result, None, "outside root → None")
        assert_true("python" not in diagnostics._SERVERS, "no server spawned")
        v = validate_file(str(path), BROKEN_PY)
        assert_equal(v.label, "python", "router fell back to tree-sitter")


def test_disk_mismatch_returns_none_before_server_start() -> None:
    # Inside the workspace root but with source differing from disk: the guard
    # must return None BEFORE _get_or_start, or a real jedi server would boot.
    root = validate._WORKSPACE_ROOT
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=root, delete=False, encoding="utf-8"
    ) as fh:
        fh.write("x = 1\n")
        path = fh.name
    try:
        result = diagnostics.lsp_check_file(path, "y = 2\n", root)
        assert_equal(result, None, "disk/source mismatch → None")
        assert_true("python" not in diagnostics._SERVERS, "no server spawned on mismatch")
    finally:
        os.unlink(path)


def test_convert_splits_severities_and_offsets() -> None:
    payload = {
        "uri": "file:///w/m.py",
        "diagnostics": [
            {
                "range": {"start": {"line": 2, "character": 0}, "end": {"line": 2, "character": 5}},
                "message": "cannot find value `boom`",
                "severity": 1,
                "source": "rustc",
            },
            {
                "range": {"start": {"line": 0, "character": 4}, "end": {"line": 0, "character": 9}},
                "message": "unused variable",
                "severity": 2,
            },
            {
                "range": {"start": {"line": 5, "character": 1}, "end": {"line": 5, "character": 2}},
                "message": "no severity means error",
            },
            {
                "range": {"start": {"line": 7, "character": 0}, "end": {"line": 7, "character": 1}},
                "message": "just a hint",
                "severity": 4,
            },
        ],
    }
    errors, warnings = diagnostics._convert(payload)
    assert_equal(
        errors,
        [
            (3, 1, "line 3, col 1: cannot find value `boom` (rustc)"),
            (6, 2, "line 6, col 2: no severity means error"),
        ],
        "0-based → 1-based, severity 1 + missing are errors, source appended",
    )
    assert_equal(warnings, [(1, 5, "line 1, col 5: unused variable")], "severity 2 → warning")
    assert_equal(diagnostics._convert({"diagnostics": []}), ([], []), "clean payload → empty lists")


def test_drop_untyped_js_type_errors_scope() -> None:
    D = diagnostics
    JS = "class X { constructor(c){ this.c = c } }\n"
    assert_true(D._drop_untyped_js_type_errors("a.js", JS), ".js (no pragma) → suppress")
    assert_true(D._drop_untyped_js_type_errors("a.jsx", JS), ".jsx → suppress")
    assert_true(D._drop_untyped_js_type_errors("a.mjs", JS), ".mjs → suppress")
    # Opt-in via @ts-check keeps JS type checking on.
    assert_true(not D._drop_untyped_js_type_errors("a.js", "// @ts-check\n" + JS),
                ".js with @ts-check → keep type errors")
    # Genuinely typed files are never suppressed.
    assert_true(not D._drop_untyped_js_type_errors("a.ts", JS), ".ts → keep (typed)")
    assert_true(not D._drop_untyped_js_type_errors("a.tsx", JS), ".tsx → keep (typed)")
    assert_true(not D._drop_untyped_js_type_errors("a.py", JS), ".py → not JS-family")
    # Code classifier: 1xxx syntax kept, >=2000 semantic dropped.
    assert_true(not D._is_ts_semantic_code(1005), "TS 1005 (syntax) not semantic")
    assert_true(D._is_ts_semantic_code(2339), "TS 2339 (property) is semantic")
    assert_true(D._is_ts_semantic_code("18047"), "string code coerces")
    assert_true(not D._is_ts_semantic_code(None), "missing code is not semantic")


def test_convert_drops_untyped_js_type_errors() -> None:
    def diag(message, code, source="typescript", severity=1, line=0):
        return {
            "range": {"start": {"line": line, "character": 0}, "end": {"line": line, "character": 3}},
            "message": message, "severity": severity, "source": source, "code": code,
        }

    payload = {"uri": "file:///w/main.js", "diagnostics": [
        diag("Property 'ctx' does not exist on type 'Environment'.", 2339, line=3),  # semantic → drop
        diag("Type 'string' is not assignable to type 'number'.", 2322, line=4),     # semantic → drop
        diag("',' expected.", 1005, line=5),                                         # syntax → keep
        diag("undefined name 'foo'", "E0602", source="pyflakes", line=6),            # non-TS → keep
    ]}
    # Suppression ON (untyped JS): only the syntax error and the non-TS diag survive.
    errors, _ = diagnostics._convert(payload, drop_ts_type_errors=True)
    assert_equal(errors, [
        (6, 1, "line 6, col 1: ',' expected. (typescript)"),
        (7, 1, "line 7, col 1: undefined name 'foo' (pyflakes)"),
    ], "JS: TS type errors dropped; syntax + non-TS kept")
    # Suppression OFF (default, e.g. a .ts file): every diagnostic is reported.
    all_errs, _ = diagnostics._convert(payload, drop_ts_type_errors=False)
    assert_equal(len(all_errs), 4, "suppression off → all four reported")


def test_convert_demotes_pylint_module_no_member_to_warning() -> None:
    # pylint cannot introspect C-extension modules (pygame, cv2, …), so its
    # module-attribute no-member is a false-positive class: demoted to a
    # warning (visible, never blocking). Instance no-member stays an error.
    def diag(message, code="E1101:no-member", source="pylint", severity=1, line=0):
        return {
            "range": {"start": {"line": line, "character": 0}, "end": {"line": line, "character": 1}},
            "message": message,
            "severity": severity,
            "source": source,
            "code": code,
        }

    payload = {
        "uri": "file:///w/m.py",
        "diagnostics": [
            diag("[no-member] Module 'pygame' has no 'init' member", line=13),
            diag("[no-member] Instance of 'Bird' has no 'flap' member", line=20),
            # Same message from a non-pylint source keeps its severity: the
            # demotion is about pylint's specific C-extension blindness.
            diag("[no-member] Module 'pygame' has no 'QUIT' member", source="mylint", line=30),
            # Code-only variant (no [no-member] prefix in the message).
            diag("Module 'cv2' has no 'imread' member", code="no-member", line=40),
        ],
    }
    errors, warnings = diagnostics._convert(payload)
    assert_equal(
        [line for line, _col, _m in errors],
        [21, 31],
        f"instance no-member and foreign sources stay errors: {errors}",
    )
    assert_equal(
        [line for line, _col, _m in warnings],
        [14, 41],
        f"pylint module no-member demoted to warnings: {warnings}",
    )
    assert_true(
        any("Module 'pygame' has no 'init'" in m for _l, _c, m in warnings),
        "the demoted row keeps its full message",
    )


def test_uri_matches_normalises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dir with space" / "m.py"
        path.parent.mkdir()
        path.write_text("x = 1\n", encoding="utf-8")
        quoted = "file://" + str(path).replace(" ", "%20")
        assert_true(diagnostics._uri_matches(quoted, str(path)), "percent-encoded URI matches")
        # macOS: tempdirs are /var/… symlinked to /private/var/… — realpath unifies.
        aliased = "file://" + os.path.realpath(str(path))
        assert_true(diagnostics._uri_matches(aliased, str(path)), "symlinked alias matches")
        assert_true(
            not diagnostics._uri_matches("file:///elsewhere/m.py", str(path)),
            "different file does not match",
        )


class _StubSyncLs:
    """Stands in for SyncLanguageServer: open_file fires the diagnostics
    handler the way multilspy's private event loop would. The factory gets
    the handle so it can address the currently armed target uri."""

    def __init__(self, handle, payload_factory) -> None:
        self._handle = handle
        self._handler = diagnostics._make_handler(handle)
        self._payload_factory = payload_factory

    @contextlib.contextmanager
    def open_file(self, relative_file_path: str):
        for payload in self._payload_factory(self._handle, relative_file_path):
            asyncio.run(self._handler(payload))
        yield


def _ready_handle(payload_factory) -> diagnostics._ServerHandle:
    handle = diagnostics._ServerHandle("python")
    handle.status = "ready"
    handle.sync_ls = _StubSyncLs(handle, payload_factory)
    return handle


def test_run_check_takes_last_payload() -> None:
    root = validate._WORKSPACE_ROOT
    abs_path = os.path.join(root, "fake.py")
    uri = "file://" + abs_path

    def two_rounds(_handle, _rel):
        first = {"uri": uri, "diagnostics": [{"range": {"start": {"line": 0, "character": 0}}, "message": "early", "severity": 1}]}
        second = {"uri": uri, "diagnostics": [{"range": {"start": {"line": 1, "character": 0}}, "message": "final", "severity": 1}]}
        return [first, second]

    handle = _ready_handle(two_rounds)
    result = diagnostics._run_check(handle, abs_path, root)
    assert_true(result is not None, "diagnostics arrived")
    errors, warnings = result
    assert_equal(errors, [(2, 1, "line 2, col 1: final")], "last payload wins")
    assert_equal(warnings, [], "no warnings in final round")
    assert_equal(handle.target_uri, None, "mailbox disarmed after the check")


def test_run_check_times_out_to_none() -> None:
    saved = diagnostics.FIRST_DIAG_TIMEOUT
    diagnostics.FIRST_DIAG_TIMEOUT = 0.05
    try:
        handle = _ready_handle(lambda _handle, _rel: [])  # server never publishes
        result = diagnostics._run_check(handle, os.path.join(validate._WORKSPACE_ROOT, "f.py"), validate._WORKSPACE_ROOT)
        assert_equal(result, None, "silent server → None → tree-sitter fallback")
    finally:
        diagnostics.FIRST_DIAG_TIMEOUT = saved


class _ClearOnCloseSyncLs:
    """Simulates a clear-on-close server (jedi): the real diagnostics arrive a
    while after didOpen (past the settle window, like jedi's ~1s parse), and an
    empty clearing set follows didClose within milliseconds — each delivered
    from a background thread the way multilspy's event loop would."""

    PARSE_DELAY = 0.5  # > SETTLE_WINDOW, < FIRST_DIAG_TIMEOUT
    CLEAR_DELAY = 0.01

    def __init__(self, handle, uri, diags) -> None:
        self._handler = diagnostics._make_handler(handle)
        self._uri = uri
        self._diags = diags
        self.threads: list[threading.Thread] = []

    def _deliver_later(self, delay, payload) -> None:
        def run() -> None:
            time.sleep(delay)
            asyncio.run(self._handler(payload))

        thread = threading.Thread(target=run)
        thread.start()
        self.threads.append(thread)

    @contextlib.contextmanager
    def open_file(self, relative_file_path: str):
        self._deliver_later(self.PARSE_DELAY, {"uri": self._uri, "diagnostics": self._diags})
        try:
            yield
        finally:
            self._deliver_later(self.CLEAR_DELAY, {"uri": self._uri, "diagnostics": []})


def test_run_check_back_to_back_survives_clear_on_close() -> None:
    # Regression: the mailbox race. jedi publishes an empty clearing set right
    # after didClose; a back-to-back re-check of the same file used to capture
    # that stale clear as its own first payload and report a false "no
    # problems" while the real re-parse publish was still ~1s away. The drain
    # in _run_check's finally must swallow the clear before disarming.
    root = validate._WORKSPACE_ROOT
    abs_path = os.path.join(root, "race.py")
    uri = "file://" + abs_path
    diags = [
        {"range": {"start": {"line": 2, "character": 27}}, "message": "expected ':'", "severity": 1}
    ]
    handle = diagnostics._ServerHandle("python")
    handle.status = "ready"
    stub = _ClearOnCloseSyncLs(handle, uri, diags)
    handle.sync_ls = stub

    first = diagnostics._run_check(handle, abs_path, root)
    second = diagnostics._run_check(handle, abs_path, root)  # no gap: the race window
    for thread in stub.threads:
        thread.join()

    assert_true(first is not None and first[0], f"first check reports the error: {first!r}")
    assert_true(
        second is not None and second[0],
        f"back-to-back re-check reports the error, not a false OK: {second!r}",
    )


def test_run_check_serialises_across_threads() -> None:
    root = validate._WORKSPACE_ROOT

    def echo_target(handle, rel):
        # Answer whichever check is currently armed, tagging the payload with
        # the file it was produced for — a leak across cycles would surface as
        # a foreign tag in the other thread's result.
        return [
            {
                "uri": "file://" + str(handle.target_uri),
                "diagnostics": [
                    {"range": {"start": {"line": 0, "character": 0}}, "message": f"for {rel}", "severity": 1}
                ],
            }
        ]

    handle = _ready_handle(echo_target)
    results: dict[str, object] = {}
    errors: list[str] = []

    def worker(name: str) -> None:
        try:
            results[name] = diagnostics._run_check(handle, os.path.join(root, f"{name}.py"), root)
        except BaseException as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert_equal(errors, [], "no cross-thread failures")
    assert_equal(sorted(results), ["t1", "t2"], "both checks completed")
    # handle.lock serialises the cycles, so each thread must get exactly the
    # payload produced for its own file.
    for name, result in results.items():
        assert_true(result is not None, f"{name} should have received diagnostics")
        errs, _warns = result  # type: ignore[misc]
        assert_equal(len(errs), 1, f"{name} got one diagnostic")
        assert_true(errs[0][2].endswith(f"for {name}.py"), f"{name} got foreign payload: {errs}")


def test_validation_report_block_contract() -> None:
    # The TUI colors any block starting "Syntax check (" and greens on ": OK" —
    # both labels must satisfy that contract.
    fallback_ok = validation_report("m.py", CLEAN_PY)
    assert_true(fallback_ok.startswith("Syntax check ("), f"prefix: {fallback_ok}")
    assert_true(": OK" in fallback_ok.splitlines()[0], f"OK on first line: {fallback_ok}")
    original = lsp_pkg.lsp_check_file
    lsp_pkg.lsp_check_file = lambda path, source, root: ([], [(1, 1, "line 1, col 1: unused")])  # type: ignore[assignment]
    try:
        lsp_ok = validation_report("m.py", CLEAN_PY)
        first = lsp_ok.splitlines()[0]
        assert_true(first.startswith("Syntax check (python, lsp)"), f"lsp label: {lsp_ok}")
        assert_true(": OK" in first, f"warnings keep the OK first line: {lsp_ok}")
        assert_true("Warnings (not blocking) — 1:" in lsp_ok, f"warnings listed: {lsp_ok}")
    finally:
        lsp_pkg.lsp_check_file = original  # type: ignore[assignment]
    # Unknown language stays silent, exactly like syntax_report.
    assert_equal(validation_report("notes.md", "x { ["), "", "unknown language → no report")


def test_warm_lsp_fast_negatives() -> None:
    # Unsupported language and out-of-root files must return False WITHOUT
    # booting a server — warming there would be pure waste.
    assert_true(not validate.warm_lsp("notes.md"), "unsupported extension → False")
    assert_true("python" not in diagnostics._SERVERS, "no server spawned for .md")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.py"
        path.write_text(CLEAN_PY, encoding="utf-8")
        assert_true(not validate.warm_lsp(str(path)), "outside root → False")
        assert_true("python" not in diagnostics._SERVERS, "no server spawned out of root")
    diagnostics.configure(False)
    try:
        assert_true(not validate.warm_lsp("m.py"), "LSP disabled → False")
    finally:
        diagnostics.configure(True)


def test_warm_lsp_waits_on_ready_status() -> None:
    # In-root supported file: warm_lsp returns get_ready_server's verdict.
    original = lsp_pkg.get_ready_server

    class _FakeHandle:
        def __init__(self, status: str) -> None:
            self.status = status

    seen: dict[str, object] = {}

    def fake_ready(language, root, wait):
        seen["args"] = (language, root, wait)
        return _FakeHandle("ready")

    lsp_pkg.get_ready_server = fake_ready  # type: ignore[assignment]
    try:
        assert_true(validate.warm_lsp("m.py"), "ready server → True")
        language, root, wait = seen["args"]  # type: ignore[misc]
        assert_equal(language, "python", "language routed from extension")
        assert_equal(root, os.path.realpath(validate._WORKSPACE_ROOT), "workspace root passed")
        assert_equal(wait, validate.LSP_BOOT_WAIT, "default bounded wait")
        lsp_pkg.get_ready_server = lambda *a: _FakeHandle("starting")  # type: ignore[assignment]
        assert_true(not validate.warm_lsp("m.py"), "still booting → False")
        lsp_pkg.get_ready_server = lambda *a: _FakeHandle("failed")  # type: ignore[assignment]
        assert_true(not validate.warm_lsp("m.py"), "failed boot (cached) → False")
    finally:
        lsp_pkg.get_ready_server = original  # type: ignore[assignment]


def test_validation_dataclass_report_shapes() -> None:
    v = Validation(errors=[(3, 1, "line 3, col 1: boom")], label="python, lsp", checked=True)
    block = v.report()
    assert_true(block.startswith("Syntax check (python, lsp) — 1 issue(s):"), f"error block: {block}")
    empty = Validation()
    assert_equal(empty.report(), "", "unchecked → empty report")


def test_server_builders_prefer_pylsp_when_installed() -> None:
    saved = diagnostics.shutil.which
    try:
        diagnostics.shutil.which = lambda cmd: "/x/pylsp" if cmd == "pylsp" else None
        assert_equal(
            diagnostics._server_builders("python"),
            [diagnostics._build_pylsp, diagnostics._build_default],
            "python prefers the pylsp lane when the binary exists",
        )
        assert_equal(
            diagnostics._server_builders("rust"),
            [diagnostics._build_default],
            "other languages keep only the stock lane",
        )
        diagnostics.shutil.which = lambda cmd: None
        assert_equal(
            diagnostics._server_builders("python"),
            [diagnostics._build_default],
            "no pylsp binary → stock jedi lane only",
        )
    finally:
        diagnostics.shutil.which = saved


class _StubBootSync:
    """Stands in for a SyncLanguageServer that boots cleanly."""

    def __init__(self) -> None:
        class _Proto:
            def on_notification(self, method, callback) -> None:
                pass

        class _Inner:
            server = _Proto()

        self.language_server = _Inner()

    @contextlib.contextmanager
    def start_server(self):
        yield self


def test_attach_falls_back_to_next_lane() -> None:
    handle = diagnostics._ServerHandle("python")

    def broken(config, logger, root):
        raise RuntimeError("pylsp lane exploded")

    def working(config, logger, root):
        return _StubBootSync()

    assert_true(
        not diagnostics._attach(handle, broken, None, None, "/root"),
        "failing lane reports False",
    )
    assert_equal(handle.status, "starting", "failed lane leaves the handle undecided")
    assert_true(
        diagnostics._attach(handle, working, None, None, "/root"),
        "next lane boots",
    )
    assert_equal(handle.status, "ready", "fallback lane wires the handle")
    assert_equal(
        handle.first_diag_timeout,
        diagnostics.FIRST_DIAG_TIMEOUT,
        "stock lane keeps the default first-publish ceiling",
    )


def test_attach_applies_pylsp_first_diag_timeout() -> None:
    handle = diagnostics._ServerHandle("python")

    def pylsp_like(config, logger, root):
        return _StubBootSync()

    pylsp_like.first_diag_timeout = diagnostics.PYLSP_FIRST_DIAG_TIMEOUT
    assert_true(
        diagnostics._attach(handle, pylsp_like, None, None, "/root"),
        "pylsp-like lane boots",
    )
    assert_equal(
        handle.first_diag_timeout,
        diagnostics.PYLSP_FIRST_DIAG_TIMEOUT,
        "pylsp lane raises the first-publish ceiling",
    )


def main() -> None:
    test_supported_language_and_configure()
    test_supported_language_content_first()
    test_router_prefers_lsp()
    test_router_falls_back_on_none()
    test_router_falls_back_when_disabled_or_import_fails()
    test_outside_workspace_root_falls_back()
    test_disk_mismatch_returns_none_before_server_start()
    test_convert_splits_severities_and_offsets()
    test_drop_untyped_js_type_errors_scope()
    test_convert_drops_untyped_js_type_errors()
    test_convert_demotes_pylint_module_no_member_to_warning()
    test_uri_matches_normalises()
    test_run_check_takes_last_payload()
    test_run_check_times_out_to_none()
    test_run_check_back_to_back_survives_clear_on_close()
    test_run_check_serialises_across_threads()
    test_validation_report_block_contract()
    test_warm_lsp_fast_negatives()
    test_warm_lsp_waits_on_ready_status()
    test_validation_dataclass_report_shapes()
    test_server_builders_prefer_pylsp_when_installed()
    test_attach_falls_back_to_next_lane()
    test_attach_applies_pylsp_first_diag_timeout()
    print("tools/lsp tests passed")


if __name__ == "__main__":
    main()
