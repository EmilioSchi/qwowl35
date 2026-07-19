"""Query-aware chunk rerank for prose tool results ("Kompress-style").

Where the statistical passes in ``strategies`` shrink by structure (repeats,
boilerplate), this lane shrinks by RELEVANCE: split the prose into paragraph
chunks, score each against the query, keep the best chunks byte-verbatim in
their original order, and mark every dropped run explicitly. Chunk-keep, not
token-delete — a 9B consumer must never see rewritten text.

Scoring backends:
- ``CrossEncoderScorer`` (the default) — a TRAINED cross-encoder
  (Qwen3-Reranker-0.6B) served natively by qw35-server's ``POST /v1/rerank``
  (the server auto-loads the reranker GGUF when present). Joint query×chunk
  attention, no lexical blend needed: any transport/HTTP failure — including
  the 501 a reranker-less server answers — latches the scorer dead and
  degrades to BM25 for the process, never raises.
- ``BM25Scorer`` — pure-Python Okapi BM25, always available, deterministic;
  the fail-open floor.
"""

from __future__ import annotations

import math
import re
import time

# Merge paragraphs up to ~this many chars per chunk (never splitting one).
CHUNK_TARGET_CHARS = 600
# Fewer chunks than this isn't worth reranking — decline instead.
MIN_CHUNKS = 4
# Cumulative kept-chunk budget; the chunk that crosses it is the last kept.
KEEP_BUDGET_CHARS = 8_000
# Latency guards: above this many chunks skip the neural path entirely, and
# one scoring call slower than the budget disables neural for the process.
MAX_NEURAL_CHUNKS = 64
NEURAL_TIME_BUDGET_S = 10.0

BM25_K1 = 1.5
BM25_B = 0.75

# Scorer modes for the default scorer: "cross-encoder" (the default) scores
# each (query, chunk) pair jointly on the server's native reranker
# (/v1/rerank), "bm25" is lexical-only and never touches the server.
SCORER_MODES = ("cross-encoder", "bm25")

# Server reranker endpoint (rides the same qw35-server as chat; the base URL
# is injected once at startup via set_rerank_base_url).
RERANK_SERVER_PATH = "/v1/rerank"
# Blocking-call transport timeouts. The total sits above NEURAL_TIME_BUDGET_S
# so in steady state the budget latch (which keeps a slow call's result)
# governs, not a mid-flight transport abort.
RERANK_CONNECT_TIMEOUT_S = 2.0
RERANK_TOTAL_TIMEOUT_S = 15.0

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Scorer:
    """Okapi BM25 with the chunk set as its own corpus. Zero dependencies."""

    name = "bm25"

    def score(self, query: str, chunks: list[str]) -> list[float]:
        docs = [_tokenize(chunk) for chunk in chunks]
        n = len(docs)
        if n == 0:
            return []
        avg_len = sum(len(d) for d in docs) / n or 1.0
        query_terms = _tokenize(query)
        idf = {}
        for term in set(query_terms):
            df = sum(1 for d in docs if term in d)
            idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        scores: list[float] = []
        for doc in docs:
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            norm = BM25_K1 * (1 - BM25_B + BM25_B * len(doc) / avg_len)
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf:
                    score += idf[term] * tf * (BM25_K1 + 1) / (tf + norm)
            scores.append(score)
        return scores


class CrossEncoderScorer:
    """Trained query×chunk relevance via the server's /v1/rerank endpoint.

    Fail-open contract (the tree-sitter pattern): BM25 is always computed as
    the floor, and any transport/HTTP failure (including the 501 the server
    answers when no reranker is loaded) latches ``_dead`` and degrades to
    BM25 for the rest of the process — never raises. The call path is
    synchronous, so a lazy blocking ``httpx.Client`` is used — never the
    app's async client.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._bm25 = BM25Scorer()
        self._base_url = base_url
        self._client = None
        self._dead = False
        self.last_used = "bm25"

    @property
    def name(self) -> str:
        return self.last_used

    def score(self, query: str, chunks: list[str]) -> list[float]:
        bm25 = self._bm25.score(query, chunks)
        self.last_used = "bm25"
        if self._dead or len(chunks) > MAX_NEURAL_CHUNKS:
            return bm25
        try:
            if self._client is None:
                import httpx

                self._client = httpx.Client(
                    base_url=self._base_url or _RERANK_BASE_URL,
                    timeout=httpx.Timeout(
                        RERANK_TOTAL_TIMEOUT_S, connect=RERANK_CONNECT_TIMEOUT_S
                    ),
                )
            started = time.monotonic()
            response = self._client.post(
                RERANK_SERVER_PATH,
                json={"query": query, "documents": list(chunks)},
            )
            response.raise_for_status()
            scores = [float(s) for s in response.json()["scores"]]
            if len(scores) != len(chunks):
                raise ValueError("scores/documents length mismatch")
            if time.monotonic() - started > NEURAL_TIME_BUDGET_S:
                # Use this result, but steady-state is too slow: BM25 next time.
                self._dead = True
        except Exception:  # noqa: BLE001 - optional backend, fail open
            self._dead = True
            return bm25
        self.last_used = "cross-encoder"
        return scores


# Shared scorer, created lazily so the module import stays free. Tests may
# overwrite this with a fake (restore to None afterwards). Cross-encoder is
# the default: the server ships the reranker, and an absent one fails open to
# BM25 within one fast local connect — never worse than the lexical floor.
_DEFAULT_SCORER = None
_SCORER_MODE = "cross-encoder"
# Base URL of the qw35 server hosting /v1/rerank; injected once at startup
# (set_rerank_base_url) from the same --base-url the chat client uses.
_RERANK_BASE_URL = "http://127.0.0.1:8080"


def set_default_scorer_mode(mode: str) -> None:
    """Pick how the default scorer works (SCORER_MODES); invalid = no-op."""
    global _SCORER_MODE, _DEFAULT_SCORER
    if mode in SCORER_MODES and mode != _SCORER_MODE:
        _SCORER_MODE = mode
        _DEFAULT_SCORER = None


def set_rerank_base_url(url: str) -> None:
    """Point the cross-encoder scorer at the qw35 server; invalid = no-op."""
    global _RERANK_BASE_URL, _DEFAULT_SCORER
    url = (url or "").rstrip("/")
    if url and url != _RERANK_BASE_URL:
        _RERANK_BASE_URL = url
        _DEFAULT_SCORER = None


def default_scorer():
    global _DEFAULT_SCORER
    if _DEFAULT_SCORER is None:
        if _SCORER_MODE == "bm25":
            _DEFAULT_SCORER = BM25Scorer()
        else:
            _DEFAULT_SCORER = CrossEncoderScorer()
    return _DEFAULT_SCORER


def chunk_paragraphs(paragraphs: list[str], target: int = CHUNK_TARGET_CHARS) -> list[str]:
    """Greedy forward merge of consecutive paragraphs into ~target-char chunks.

    Paragraphs are never split, so every chunk is a contiguous verbatim slice:
    ``"\\n\\n".join(chunk_paragraphs(ps)) == "\\n\\n".join(ps)``.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        current.append(paragraph)
        size += len(paragraph) + 2
        if size >= target:
            chunks.append("\n\n".join(current))
            current, size = [], 0
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def rerank_prose(
    paragraphs: list[str],
    query: str,
    budget: int = KEEP_BUDGET_CHARS,
    scorer=None,
) -> list[str] | None:
    """Keep the query-relevant chunks verbatim, in order; None declines.

    Declines (returns None) when there is no query, too few chunks, nothing
    would be dropped, or anything raises — the caller then falls back to the
    statistical path. Dropped runs become explicit count markers so nothing
    disappears silently.
    """
    try:
        if not query.strip():
            return None
        chunks = chunk_paragraphs(paragraphs)
        if len(chunks) < MIN_CHUNKS:
            return None
        if scorer is None:
            scorer = default_scorer()
        scores = scorer.score(query, chunks)
        # Chunk 0 (title/lead context) is always kept; add the rest by score
        # until the budget is crossed.
        keep = {0}
        kept_chars = len(chunks[0])
        for index in sorted(range(1, len(chunks)), key=lambda i: scores[i], reverse=True):
            if kept_chars >= budget:
                break
            keep.add(index)
            kept_chars += len(chunks[index])
        if len(keep) == len(chunks):
            return None
        out: list[str] = []
        dropped_run: list[str] = []

        def flush_run() -> None:
            if dropped_run:
                chars = sum(len(c) for c in dropped_run)
                out.append(
                    f"[… {len(dropped_run)} chunks ({chars} chars) not relevant "
                    "to the query, elided]"
                )
                dropped_run.clear()

        for index, chunk in enumerate(chunks):
            if index in keep:
                flush_run()
                out.append(chunk)
            else:
                dropped_run.append(chunk)
        flush_run()
        out.append(
            f"[query-relevance rerank: kept {len(keep)} of {len(chunks)} chunks, "
            f"scorer={scorer.name}]"
        )
        return out
    except Exception:  # noqa: BLE001 - rerank must never damage a tool result
        return None
