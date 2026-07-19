"""LSP semantic diagnostics for edited files via multilspy.

The primary validation layer behind ``tools/syntax/validate.py``: real language
servers (jedi-language-server, rust-analyzer, gopls, …) driven through the
optional ``multilspy`` package report semantic problems (unresolved symbols,
type errors) that the tree-sitter fallback cannot see.

multilspy exposes no diagnostics API of its own (only definitions/references/
completions/hover), so this module captures the server's asynchronous
``textDocument/publishDiagnostics`` notifications: it registers a handler on
multilspy's protocol handler, opens the just-written file (didOpen), and waits
— bounded — for the diagnostics push. Registration must happen AFTER
``start_server`` returns: each server subclass installs a ``do_nothing``
handler for that method during startup, and the last registration wins.

Design rules (shared with ``tools/syntax/checker.py`` and
``tools/compress/detect.py``):
- **Never raise.** Every public function degrades to "could not check".
- **Optional dependency.** ``multilspy`` (and each language-server binary) is
  imported/launched lazily; a failure is cached per language and never retried,
  so a missing backend costs one attempt per session (the Magika idiom).
- **Never stall an edit.** Servers boot on a daemon thread; until a server is
  ready, checks return ``None`` and the caller uses tree-sitter instead.
- **Thread-safe.** The file tools run on ``asyncio.to_thread`` pool threads;
  each server handle carries a lock serialising its open/wait/convert cycle.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

# Extension → multilspy Language value. Curated to the languages the installed
# multilspy release (0.0.15) actually ships servers for — csharp, python, rust,
# java, kotlin, typescript, javascript, go, ruby, dart. NOT the same map as
# checker._EXT_TO_LANG (grammar names): ``.tsx`` is "typescript" here but the
# "tsx" grammar there, and cpp/json/yaml/… have no LSP backend, so they always
# take the tree-sitter path.
_EXT_TO_LSP_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".dart": "dart",
}

# Magika content-type label → multilspy Language value. Content detection
# (tools/compress/detect) runs first in supported_language; a confident label
# here wins even over a known extension — mirroring that module's Magika-first
# policy — so a mis- or un-extensioned file still reaches the right server. Note
# Magika labels C# as "cs" (not "csharp"), and "tsx"/"jsx" are not Magika labels
# (it returns "typescript"/"javascript"). Labels with no LSP backend are absent,
# so they fall through to the extension map rather than mapping to nothing.
_MAGIKA_TO_LSP_LANG: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "rust": "rust",
    "go": "go",
    "java": "java",
    "ruby": "ruby",
    "cs": "csharp",
    "kotlin": "kotlin",
    "dart": "dart",
}

# JavaScript-family extensions: files the TypeScript server checks but which are
# untyped, so its *semantic* (type) diagnostics are false positives on valid,
# runtime-correct code — see _drop_untyped_js_type_errors.
_JS_FAMILY_EXTS = frozenset({".js", ".jsx", ".mjs", ".cjs"})

# How long to wait for the first publishDiagnostics after didOpen before giving
# up on the LSP answer (tree-sitter takes over for that check). Bounds the
# per-edit latency a slow or silent server can add.
FIRST_DIAG_TIMEOUT = 2.0
# The pylsp lane runs pylint per check, which is markedly slower than jedi's
# parse — but it always publishes once lint finishes, so the wait ends at the
# publish, not the ceiling. The higher ceiling only bites when pylsp is
# genuinely wedged.
PYLSP_FIRST_DIAG_TIMEOUT = 6.0
# After the first payload, keep listening this long for follow-up rounds (e.g.
# rust-analyzer pushes a quick syntax pass, then cargo-check results); the last
# payload wins because servers publish full replacement sets, not deltas.
SETTLE_WINDOW = 0.3
# After didClose, clear-on-close servers (jedi does) publish an empty set a few
# milliseconds later. Each check cycle waits this long to swallow that clear
# while it still holds the handle lock; otherwise back-to-back checks of the
# same file capture it as the NEXT cycle's first payload and report a false
# "no problems" (the real re-parse publish lands ~1s later, past the settle
# window). Cheap: the clear arrives in ~10ms, so the wait almost never runs
# its full course on clearing servers; non-clearing servers pay it once per
# check as idle time.
CLOSE_CLEAR_WINDOW = 0.3
# Mirror checker._MAX_SOURCE_BYTES: beyond this we skip checking entirely.
_MAX_SOURCE_BYTES = 1_000_000
# Bound on any single SyncLanguageServer request_* call (definition, references,
# hover, document symbols). Diagnostics itself never goes through .result() —
# it waits on its own mailbox with FIRST_DIAG_TIMEOUT — so this only protects
# the navigation/query path from a wedged server.
SYNC_REQUEST_TIMEOUT = 15

# Master switch, set from Config.lsp via configure() (app.py / headless.py).
_ENABLED = True

# One handle per multilspy language, living for the whole process (the
# workspace root never changes after launch). Failures are cached: a language
# whose server cannot start stays on the tree-sitter path for the session.
_SERVERS: dict[str, "_ServerHandle"] = {}
_SERVERS_GUARD = threading.Lock()
_ATEXIT_REGISTERED = False


class _ServerHandle:
    """A per-language server plus the diagnostics mailbox its handler fills."""

    def __init__(self, language: str) -> None:
        self.language = language
        self.status = "starting"  # "starting" | "ready" | "failed"
        self.sync_ls: Any = None
        self.ctx: Any = None  # the entered start_server() context manager
        # Per-backend ceiling on the first-publish wait (_attach sets the
        # pylsp lane's higher one).
        self.first_diag_timeout = FIRST_DIAG_TIMEOUT
        # Serialises check cycles: the file tools run on a thread pool, and one
        # open/wait/convert cycle must not see another's notifications.
        self.lock = threading.Lock()
        # Mailbox armed per check; filled by the async notification handler
        # running on multilspy's private event-loop thread.
        self.target_uri: str | None = None
        self.payloads: list[dict] = []
        self.event = threading.Event()


def configure(enabled: bool) -> None:
    """Master switch (Config.lsp): False routes every check to tree-sitter."""
    global _ENABLED
    _ENABLED = bool(enabled)


def is_enabled() -> bool:
    """Whether the LSP layer is enabled (Config.lsp via :func:`configure`)."""
    return _ENABLED


def supported_language(path: str | Path, source: str | None = None) -> str | None:
    """multilspy language for ``path``, or ``None`` (incl. disabled).

    Content-first: when ``source`` is given, a confident Magika verdict picks
    the language and wins even over a known extension (mirroring
    tools/compress/detect's Magika-first policy), so a mis- or un-extensioned
    file still reaches the right server. The extension map is the fallback —
    used when ``source`` is absent, Magika is uninstalled/unsure, or its label
    has no LSP backend. Never raises.
    """
    try:
        if not _ENABLED:
            return None
        if source:
            # Local import: keep the optional Magika/compress dependency off the
            # module-load path and out of the lsp→compress import cycle.
            from tools.compress.detect import magika_label

            label = magika_label(source)
            if label is not None:
                lang = _MAGIKA_TO_LSP_LANG.get(label)
                if lang is not None:
                    return lang
        return _EXT_TO_LSP_LANG.get(Path(path).suffix.lower())
    except Exception:  # noqa: BLE001 - never raise to callers
        return None


def lsp_check_file(
    path: str | Path, source: str, root: str
) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]] | None:
    """LSP diagnostics for the on-disk file at ``path``.

    Returns ``(errors, warnings)`` — 1-based ``(line, col, message)`` tuples,
    severity Error vs Warning — when the file was actually checked (both lists
    empty = checked clean), or ``None`` when it could not be checked and the
    caller should fall back to tree-sitter: LSP disabled, unsupported language,
    file outside ``root``, disk content differing from ``source`` (the server
    reads disk, so we would validate the wrong bytes), server missing/booting/
    failed, or no diagnostics arriving within the timeout.
    """
    try:
        if not source:
            return None
        if len(source.encode("utf-8", errors="replace")) > _MAX_SOURCE_BYTES:
            return None
        language = supported_language(path, source)
        if language is None:
            return None
        abs_path = os.path.realpath(str(path))
        root_real = os.path.realpath(root)
        try:
            if not Path(abs_path).is_relative_to(root_real):
                return None
        except Exception:  # noqa: BLE001 - defensive (odd paths)
            return None
        try:
            if Path(abs_path).read_bytes() != source.encode("utf-8", errors="replace"):
                return None
        except OSError:
            return None
        handle = _get_or_start(language, root_real)
        if handle.status != "ready":
            return None  # booting (this round falls back) or failed (cached)
        return _run_check(handle, abs_path, root_real, _drop_untyped_js_type_errors(path, source))
    except Exception:  # noqa: BLE001 - diagnostics are best-effort
        return None


def get_ready_server(language: str, root: str, wait: float = 15.0) -> _ServerHandle:
    """The language's server handle, waiting up to ``wait`` seconds for a boot.

    Unlike the diagnostics path (which never waits — a booting server just
    falls back to tree-sitter), interactive query tools prefer to block a
    bounded moment on first use rather than waste a whole model round-trip.
    Always returns the handle; the caller branches on ``handle.status``
    ("ready" | "starting" | "failed" — failures are cached for the session).
    Never raises.
    """
    handle = _get_or_start(language, root)
    deadline = time.monotonic() + max(0.0, wait)
    while handle.status == "starting" and time.monotonic() < deadline:
        time.sleep(0.1)
    return handle


def shutdown_all() -> None:
    """Stop every running language server. Idempotent; never raises."""
    with _SERVERS_GUARD:
        handles = list(_SERVERS.values())
        _SERVERS.clear()
    for handle in handles:
        try:
            if handle.ctx is not None:
                handle.ctx.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass


# --- internals ----------------------------------------------------------------


def _get_or_start(language: str, root: str) -> _ServerHandle:
    """Existing handle for ``language``, or a new one booting on a daemon thread."""
    with _SERVERS_GUARD:
        handle = _SERVERS.get(language)
        if handle is not None:
            return handle
        handle = _ServerHandle(language)
        _SERVERS[language] = handle
    threading.Thread(
        target=_start_server_thread,
        args=(handle, root),
        name=f"qw35-lsp-{language}",
        daemon=True,
    ).start()
    return handle


def _start_server_thread(handle: _ServerHandle, root: str) -> None:
    """Boot a language server; on any failure mark the handle failed (cached).

    Tries each backend lane from :func:`_server_builders` in order (Python:
    pylsp first, jedi fallback) and keeps the first one that boots.
    """
    try:
        from multilspy.multilspy_config import MultilspyConfig
        from multilspy.multilspy_logger import MultilspyLogger

        # A chatty logger writing to stderr would corrupt the Textual screen.
        logging.getLogger("multilspy").setLevel(logging.CRITICAL)
        config = MultilspyConfig.from_dict({"code_language": handle.language})
        logger = MultilspyLogger()
        inner = getattr(logger, "logger", None)
        if inner is not None:
            try:
                inner.setLevel(logging.CRITICAL)
            except Exception:  # noqa: BLE001 - logger shape varies by version
                pass
        for build in _server_builders(handle.language):
            if _attach(handle, build, config, logger, root):
                _register_atexit()
                return
        handle.status = "failed"
    except Exception:  # noqa: BLE001 - missing package, unexpected failure
        handle.status = "failed"


def _server_builders(language: str) -> list:
    """Backend constructor lanes for ``language``, preferred first.

    Python prefers pylsp when its binary is installed: jedi-language-server
    publishes nothing beyond syntax errors (raw payload verified empty for
    undefined names), while pylsp's pylint+pyflakes report the semantic
    findings (E0602 undefined-variable, W0102 dangerous-default-value, …).
    Every language keeps multilspy's stock server as the (last) lane.
    """
    builders = []
    if language == "python" and shutil.which("pylsp") is not None:
        builders.append(_build_pylsp)
    builders.append(_build_default)
    return builders


def _build_pylsp(config: Any, logger: Any, root: str) -> Any:
    from multilspy import SyncLanguageServer

    from .pylsp_server import PylspServer

    return SyncLanguageServer(PylspServer(config, logger, root), timeout=SYNC_REQUEST_TIMEOUT)


# pylint lints slower than jedi parses; give the lane a higher first-publish
# ceiling (read by _attach). The stock lanes keep FIRST_DIAG_TIMEOUT.
_build_pylsp.first_diag_timeout = PYLSP_FIRST_DIAG_TIMEOUT


def _build_default(config: Any, logger: Any, root: str) -> Any:
    from multilspy import SyncLanguageServer

    # timeout bounds the sync request_* wrappers (the query tool); the
    # diagnostics mailbox has its own first_diag_timeout and is unaffected.
    return SyncLanguageServer.create(config, logger, root, timeout=SYNC_REQUEST_TIMEOUT)


def _attach(handle: _ServerHandle, build: Any, config: Any, logger: Any, root: str) -> bool:
    """Boot one backend lane and wire the diagnostics mailbox to it."""
    try:
        sync_ls = build(config, logger, root)
        ctx = sync_ls.start_server()
        ctx.__enter__()
    except Exception:  # noqa: BLE001 - missing binary/startup failure: next lane
        return False
    # Register AFTER __enter__: the server subclass's start_server installed
    # a do_nothing handler for publishDiagnostics; ours must overwrite it.
    proto = getattr(getattr(sync_ls, "language_server", None), "server", None)
    register = getattr(proto, "on_notification", None)
    if not callable(register):
        try:
            ctx.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass
        return False
    register("textDocument/publishDiagnostics", _make_handler(handle))
    handle.first_diag_timeout = getattr(build, "first_diag_timeout", FIRST_DIAG_TIMEOUT)
    handle.sync_ls = sync_ls
    handle.ctx = ctx
    handle.status = "ready"
    return True


def _register_atexit() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        _ATEXIT_REGISTERED = True
        atexit.register(shutdown_all)


def _make_handler(handle: _ServerHandle):
    """Async publishDiagnostics callback filling ``handle``'s armed mailbox.

    Runs on multilspy's private event-loop thread; the dispatcher awaits it.
    """

    async def _on_diags(params: Any) -> None:
        try:
            if not isinstance(params, dict) or handle.target_uri is None:
                return
            if _uri_matches(str(params.get("uri", "")), handle.target_uri):
                handle.payloads.append(params)
                handle.event.set()
        except Exception:  # noqa: BLE001 - never break the protocol loop
            return

    return _on_diags


def _run_check(
    handle: _ServerHandle, abs_path: str, root: str, drop_ts_type_errors: bool = False
) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]] | None:
    """One serialized open → wait → settle → convert cycle against ``handle``."""
    relpath = os.path.relpath(abs_path, root)
    with handle.lock:
        handle.payloads = []
        handle.event.clear()
        handle.target_uri = abs_path
        try:
            with handle.sync_ls.open_file(relpath):
                if not handle.event.wait(handle.first_diag_timeout):
                    return None
                # Servers may push several full replacement sets (e.g. a quick
                # syntax pass then compiler results); keep the last one.
                while True:
                    handle.event.clear()
                    if not handle.event.wait(SETTLE_WINDOW):
                        break
                payloads = list(handle.payloads)
        finally:
            # The open_file exit above sent didClose; swallow the clearing
            # empty set that clear-on-close servers publish for it BEFORE
            # disarming, still under the lock — otherwise a back-to-back
            # check re-arms first and mistakes the stale clear for its own
            # clean answer (false OK; the mailbox race).
            handle.event.clear()
            handle.event.wait(CLOSE_CLEAR_WINDOW)
            handle.target_uri = None
            handle.payloads = []
    if not payloads:
        return None
    return _convert(payloads[-1], drop_ts_type_errors)


def _uri_matches(uri: str, target_abs_path: str) -> bool:
    """Whether an LSP ``file://`` URI refers to ``target_abs_path``.

    Servers percent-encode differently and macOS aliases ``/var`` to
    ``/private/var``, so compare realpath-normalised, case-folded paths.
    """
    try:
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme and parsed.scheme != "file":
            return False
        uri_path = urllib.parse.unquote(parsed.path or uri)
        return os.path.normcase(os.path.realpath(uri_path)) == os.path.normcase(
            os.path.realpath(target_abs_path)
        )
    except Exception:  # noqa: BLE001 - defensive
        return False


def _is_module_no_member(diag: dict, text: str) -> bool:
    """Whether ``diag`` is pylint's no-member (E1101) on a MODULE attribute.

    pylint cannot introspect C-extension modules (pygame, cv2, lxml, …), so
    ``Module 'pygame' has no 'init' member`` fires on perfectly valid code —
    a whole class of false positives that must never flag a file as broken and
    send the model off "fixing" correct lines. These are demoted to warnings:
    still visible (a genuine module typo shows as a non-blocking row; pyflakes
    still ERRORS on real undefined names), never blocking. Instance/class
    no-member stays an error — there pylint sees the actual class.
    """
    if str(diag.get("source") or "").lower() != "pylint":
        return False
    code = str(diag.get("code") or "")
    if "no-member" not in code and "E1101" not in code and "[no-member]" not in text:
        return False
    body = text.removeprefix("[no-member]").lstrip()
    return body.startswith("Module '")


def _drop_untyped_js_type_errors(path: str | Path, source: str | None) -> bool:
    """Whether the TypeScript server's *type* diagnostics on ``path`` are noise.

    Plain JavaScript is untyped, so tsserver's semantic checks (``Property 'x'
    does not exist``, ``not assignable``, ``possibly null`` …) fire on valid,
    runtime-correct code — the exact false positives editors suppress by
    defaulting ``checkJs`` off. True for .js/.jsx/.mjs/.cjs files that have NOT
    opted into checking via a ``@ts-check`` pragma; .ts/.tsx are genuinely
    typed and never suppressed. Syntax errors (TS 1xxx) are kept regardless —
    they are true in any dialect (see :func:`_is_ts_semantic_code`).
    """
    try:
        if Path(path).suffix.lower() not in _JS_FAMILY_EXTS:
            return False
        # `@ts-check` (line/block comment) is the file-level opt-in to JS type
        # checking; when present the type diagnostics are intended, so keep them.
        return "@ts-check" not in (source or "")[:2000]
    except Exception:  # noqa: BLE001 - never raise to callers
        return False


def _is_ts_semantic_code(code: Any) -> bool:
    """Whether a TypeScript diagnostic ``code`` is semantic (type) vs syntactic.

    TS numbers syntax/parser errors in 1000–1999 (``';' expected`` …) — true in
    any dialect, always kept — and type/semantic errors from 2000 up (2339
    property-does-not-exist, 2322/2345 not-assignable, 18047 possibly-null …),
    which are the untyped-JS false positives. Non-numeric/unknown codes are not
    treated as semantic, so nothing is over-suppressed.
    """
    try:
        return int(code) >= 2000
    except (TypeError, ValueError):
        return False


def _convert(
    params: dict,
    drop_ts_type_errors: bool = False,
) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]:
    """Split an LSP publishDiagnostics payload into (errors, warnings).

    Positions are converted 0-based → 1-based and messages formatted like the
    tree-sitter checker's (``line N, col M: …``) so both layers render
    identically. Severity 1 (or missing — servers may omit it) is an error,
    2 a warning; hints/infos (3-4) are dropped. One targeted demotion:
    pylint's module-attribute no-member (see :func:`_is_module_no_member`)
    lands as a warning regardless of the server's severity. When
    ``drop_ts_type_errors`` is set (untyped JS, see
    :func:`_drop_untyped_js_type_errors`), TypeScript *semantic* diagnostics are
    discarded entirely — they are false on untyped code — while its syntax
    errors still pass through.
    """
    errors: list[tuple[int, int, str]] = []
    warnings: list[tuple[int, int, str]] = []
    for diag in params.get("diagnostics", []) or []:
        try:
            source = diag.get("source")
            if (
                drop_ts_type_errors
                and str(source or "").lower() == "typescript"
                and _is_ts_semantic_code(diag.get("code"))
            ):
                continue
            start = diag.get("range", {}).get("start", {})
            line = int(start.get("line", 0)) + 1
            col = int(start.get("character", 0)) + 1
            text = " ".join(str(diag.get("message", "")).split())
            message = f"line {line}, col {col}: {text}"
            if source:
                message = f"{message} ({source})"
            severity = int(diag.get("severity", 1) or 1)
            if severity <= 1 and _is_module_no_member(diag, text):
                severity = 2
        except Exception:  # noqa: BLE001 - skip malformed entries
            continue
        if severity <= 1:
            errors.append((line, col, message))
        elif severity == 2:
            warnings.append((line, col, message))
    return errors, warnings
