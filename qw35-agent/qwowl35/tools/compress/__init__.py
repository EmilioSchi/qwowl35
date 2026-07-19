"""Tool-output compression: shrink results before they become prompt tokens.

Prefill on the local model is the expensive step, and tool results are the
bulk of it. Each result is compressed exactly ONCE, inside the registry, at
the moment it is produced — never as a later rewrite of history, which would
break the server's checkpoint-prefix KV reuse (see Agent._compact_completed_history).

Strategies (tools.compress.strategies) are adapted from Headroom
(github.com/headroomlabs-ai/headroom, Apache-2.0). Recovery is per-call: the
big tools' schemas take an optional ``compress`` boolean (default true), and a
compressed result ends with a marker telling the model to re-call with
``compress:false`` when it truly needs everything.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.diagnostics import join_section, split_trailing_section
from tools.files.adapter import TOOL_ATTENTION_MARKER

from .strategies import compress_code, compress_grep, compress_log, compress_web

# Tools whose output is worth compressing. Everything else (ls, glob, plan
# tools, `edit`, the hashline file tools whose anchors edits depend on)
# passes through verbatim.
COMPRESSIBLE_TOOLS = frozenset(
    {"run_shell_command", "bash", "inspect_file", "grep_search", "web_fetch"}
)

# ~500 tokens ≈ 5s of prefill at 96 tok/s; below that the marker overhead and
# elision risk outweigh the saving. inspect_file is exempt (see
# compress_tool_result): comment pruning's marker cost is now a single fixed
# footer rather than one per elided run, so it's worth attempting on files of
# any size — the net-savings check below still protects against a marker
# that costs more than a small file's comments save.
MIN_CHARS_TO_COMPRESS = 2_000

# A compression that saves less than this isn't worth the marker: keep the
# original. Both bounds must be met. inspect_file uses a plain saved > 0
# check instead (see compress_tool_result) — these floors exist to protect
# against per-run marker pileup, which comment pruning no longer has.
MIN_SAVED_CHARS = 400
MIN_SAVED_RATIO = 0.15

# Rendered at the end of every compressed result; doubles as the idempotency
# guard — text already carrying it is never compressed again.
MARKER_PREFIX = "\n[compressed: "

_STRATEGIES = {
    "run_shell_command": compress_log,
    "bash": compress_log,
    "inspect_file": compress_code,
    "grep_search": compress_grep,
    "web_fetch": compress_web,
}


@dataclass(frozen=True)
class CompressResult:
    text: str  # final text (marker appended when compressed)
    original_chars: int
    saved_chars: int  # original_chars - len(text); 0 when not compressed
    was_compressed: bool


def compress_requested(arguments: dict) -> bool:
    """The per-call ``compress`` flag; missing means yes.

    The XML tool-call path delivers booleans as strings, so "false"/"true"
    are honored alongside real booleans (cf. BashTool's is_background).
    """
    value = arguments.get("compress") if isinstance(arguments, dict) else None
    if value is False:
        return False
    if isinstance(value, str) and value.strip().lower() == "false":
        return False
    return True


def strip_compress_arg(arguments: dict) -> dict:
    """A copy of ``arguments`` without the ``compress`` key (never mutates:
    the caller reuses the original dict for dedup signatures and describe_call)."""
    if not isinstance(arguments, dict) or "compress" not in arguments:
        return arguments
    return {k: v for k, v in arguments.items() if k != "compress"}


def _web_query(arguments: dict) -> str:
    """web_fetch's required `prompt` argument — the rerank lane's query."""
    prompt = arguments.get("prompt") if isinstance(arguments, dict) else None
    return prompt.strip() if isinstance(prompt, str) else ""


def _code_path(arguments: dict) -> str:
    """inspect_file's file_path argument — drives language detection's extension fallback."""
    path = arguments.get("file_path") if isinstance(arguments, dict) else None
    return path.strip() if isinstance(path, str) else ""


def compress_tool_result(
    tool_name: str, arguments: dict, text: str, rerank: bool = True
) -> CompressResult:
    """Compress one tool result, routed by tool name; conservative by design.

    Returns the input unchanged when the tool isn't compressible, it is an
    error/attention result, it already carries the marker, the strategy was a
    no-op, or the saving is too small to be worth the marker. The small-text
    size gate (``MIN_CHARS_TO_COMPRESS``) and the larger saved-chars/ratio
    floor (``MIN_SAVED_CHARS``/``MIN_SAVED_RATIO``) both apply to every tool
    except ``inspect_file``, which instead always attempts comment pruning and
    keeps it whenever it's a net positive save — its marker cost is now a
    single fixed footer, not one per elided comment run, so there's no
    per-run pileup left to guard against. ``rerank`` gates the query-aware
    semantic lane (web results only).
    """
    original_chars = len(text)
    untouched = CompressResult(text, original_chars, 0, False)
    strategy = _STRATEGIES.get(tool_name)
    if strategy is None or tool_name not in COMPRESSIBLE_TOOLS:
        return untouched
    is_code = strategy is compress_code
    if not is_code and original_chars < MIN_CHARS_TO_COMPRESS:
        return untouched
    if text.startswith("Error:") or TOOL_ATTENTION_MARKER in text:
        return untouched
    if MARKER_PREFIX in text:
        return untouched

    # Diagnostics are never compression input. A trailing ``Syntax check (…)``/
    # ``LSP diagnostics (…)`` section is carved off BEFORE any strategy runs —
    # no code-cutting logic (comment pruning, repeat collapsing, language
    # detection, future lanes) ever sees it — and re-attached verbatim after
    # the marker, so the final text still ends with a canonical section.
    body, diagnostics = split_trailing_section(text)
    body_chars = len(body)

    comment_count = 0
    if strategy is compress_web:
        compressed = compress_web(body, _web_query(arguments) if rerank else "")
    elif is_code:
        compressed, comment_count = compress_code(body, _code_path(arguments))
    else:
        compressed = strategy(body)
    if compressed == body:
        return untouched

    comment_note = (
        f" ({comment_count} comment lines blanked, numbering kept)"
        if comment_count
        else ""
    )
    marker = (
        f"{MARKER_PREFIX}{body_chars - len(compressed)} of {body_chars} "
        f"chars elided{comment_note}; re-call with compress:false for the full output]"
    )
    final = join_section(compressed + marker, diagnostics)
    saved = original_chars - len(final)
    if is_code:
        if saved <= 0:
            return untouched
    elif saved < MIN_SAVED_CHARS or saved < original_chars * MIN_SAVED_RATIO:
        return untouched
    return CompressResult(final, original_chars, saved, True)
