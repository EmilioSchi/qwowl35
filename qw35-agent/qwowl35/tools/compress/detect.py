"""Content-type detection routing compression's language-aware passes.

Adopts Headroom's detector policy (headroom/compression/detector.py): Magika —
Google's ML content-type classifier — is preferred but OPTIONAL; its import is
guarded like tree-sitter-language-pack's (see requirements.txt), so a missing
install degrades to extension-based detection via the syntax checker's
extension map, and from there to "no detection" (which downstream means "no
language-aware pass", never an error).

Precedence is Magika-first: a confident verdict wins even over a known file
extension, and a confident-but-unmapped label (markdown, json, plain text…)
deliberately returns None rather than second-guessing via the extension — a
data file misnamed `.py` should not be parsed as Python.

Compression calls this on the event-loop thread (inside compress_tool_result),
but the LSP diagnostics layer (tools/lsp) also calls :func:`magika_label` from
worker threads, so the lazy module-global Magika singleton is guarded by a lock.
"""

from __future__ import annotations

import threading
from typing import Any

from tools.syntax.checker import language_for_path

# Below this Magika score we do not trust the label and fall back to the file
# extension (mirrors Headroom's detector threshold).
MAGIKA_MIN_SCORE = 0.5

# Magika content-type label → tree-sitter-language-pack grammar name. Only
# languages whose grammars the pack ships and whose comment nodes are worth
# pruning. Unmapped labels mean "no per-language pass".
_MAGIKA_TO_GRAMMAR: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "c": "c",
    "cpp": "cpp",
    "rust": "rust",
    "go": "go",
    "java": "java",
    "ruby": "ruby",
    "shell": "bash",  # magika's label vs the grammar's name
    "yaml": "yaml",
    "toml": "toml",
}

# Lazy singleton state. Constructing Magika() loads an ONNX model (~100ms,
# once); a failed construction is cached so a missing install is not retried
# on every tool result.
_MAGIKA: Any | None = None
_MAGIKA_FAILED: bool = False
# Serialises the lazy construction and each identify call. Compression runs on
# the event-loop thread, but the LSP layer calls magika_label from worker
# threads, so construction/inference must not race.
_MAGIKA_LOCK = threading.Lock()


def detect_language(text: str, file_path: str = "") -> str | None:
    """Tree-sitter grammar name for ``text``, or None when no pass applies.

    Magika first (when installed and confident), extension fallback otherwise.
    Never raises.
    """
    try:
        label, score = _magika_identify(text.encode("utf-8", errors="replace"))
        if label is not None and score >= MAGIKA_MIN_SCORE:
            # A confident unmapped label (markdown, json, txt…) means "not a
            # prunable code type" — trust it over the extension.
            return _MAGIKA_TO_GRAMMAR.get(label)
        if file_path:
            return language_for_path(file_path)
        return None
    except Exception:  # noqa: BLE001 - detection is best-effort, never raises
        return None


def magika_label(text: str) -> str | None:
    """Confident Magika content-type label for ``text``, or None.

    The raw label (e.g. "python", "javascript", "typescript", "cs") when Magika
    is installed and scores at/above ``MAGIKA_MIN_SCORE``; None otherwise
    (Magika absent, low confidence, or any error). Unlike :func:`detect_language`
    this returns Magika's own label rather than a grammar name, leaving each
    caller to apply its own label→target map — the LSP layer maps to multilspy
    languages. Never raises.
    """
    try:
        label, score = _magika_identify(text.encode("utf-8", errors="replace"))
        if label is not None and score >= MAGIKA_MIN_SCORE:
            return label
        return None
    except Exception:  # noqa: BLE001 - detection is best-effort, never raises
        return None


def _magika_identify(data: bytes) -> tuple[str | None, float]:
    """``(label, score)`` from Magika, or ``(None, 0.0)`` when unavailable.

    Reads the module globals at call time (tests stub them). Result access is
    version-defensive — Magika 0.5.x exposes ``result.output.ct_label`` /
    ``result.output.score``, 0.6+ ``result.prediction.output.label`` /
    ``result.prediction.score`` — probing chains the way checker._parse probes
    the tree-sitter binding.
    """
    global _MAGIKA, _MAGIKA_FAILED
    try:
        with _MAGIKA_LOCK:
            if _MAGIKA_FAILED:
                return None, 0.0
            if _MAGIKA is None:
                try:
                    from magika import Magika

                    _MAGIKA = Magika()
                except Exception:  # noqa: BLE001 - optional dependency
                    _MAGIKA_FAILED = True
                    return None, 0.0
            result = _MAGIKA.identify_bytes(data)
        label = _probe(
            result, ("prediction", "output", "label"), ("output", "ct_label"), ("output", "label")
        )
        score = _probe(result, ("prediction", "score"), ("output", "score"), ("score",))
        if label is None or score is None:
            return None, 0.0
        return str(label).lower(), float(score)
    except Exception:  # noqa: BLE001 - never raise out of detection
        return None, 0.0


def _probe(obj: Any, *chains: tuple[str, ...]) -> Any | None:
    """First non-None value reached by walking any of ``chains`` off ``obj``."""
    for chain in chains:
        value = obj
        for name in chain:
            value = getattr(value, name, None)
            if value is None:
                break
        if value is not None:
            return value
    return None
