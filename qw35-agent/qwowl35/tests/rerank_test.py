"""Tests for tools/compress/rerank — the query-aware chunk rerank lane.

Run directly: ``python qwowl35/tests/rerank_test.py``. Hermetic: no network,
no filesystem. The CrossEncoderScorer tests stub the HTTP layer with
httpx.MockTransport; the live path needs a running qw35 with its reranker
(auto-loaded when the GGUF is present).

Manual live smoke (server running):
  python -c "import sys; sys.path.insert(0, 'qwowl35'); \
from tools.compress.rerank import CrossEncoderScorer; s = CrossEncoderScorer(); \
print(s.score('what is BM25 k1', ['BM25 uses k1=1.5', 'cats are fluffy'])); \
print(s.name)"
Expect scorer name "cross-encoder" and the first chunk scored on top.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.compress.rerank as rerank_mod  # noqa: E402
from tools.compress.rerank import (  # noqa: E402
    MIN_CHUNKS,
    BM25Scorer,
    CrossEncoderScorer,
    chunk_paragraphs,
    default_scorer,
    rerank_prose,
    set_default_scorer_mode,
    set_rerank_base_url,
)


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


class FakeScorer:
    name = "fake"

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def score(self, query: str, chunks: list[str]) -> list[float]:
        return list(self._scores[: len(chunks)])


class RaisingScorer:
    name = "boom"

    def score(self, query: str, chunks: list[str]) -> list[float]:
        raise RuntimeError("scorer exploded")


# --- chunker ------------------------------------------------------------------


def test_chunker_merges_to_target_and_reassembles() -> None:
    paragraphs = [f"paragraph {i} " + "x" * 90 for i in range(10)]
    chunks = chunk_paragraphs(paragraphs, target=300)
    assert_true(len(chunks) < len(paragraphs), "paragraphs merged")
    assert_true(all(len(c) >= 300 for c in chunks[:-1]), "chunks reach target")
    assert_equal(
        "\n\n".join(chunks), "\n\n".join(paragraphs), "reassembly is byte-identical"
    )


def test_chunker_never_splits_a_paragraph() -> None:
    big = "y" * 5000
    chunks = chunk_paragraphs(["small one", big, "small two"], target=300)
    assert_true(any(big in c for c in chunks), "oversized paragraph intact in one chunk")


def test_chunker_empty_input() -> None:
    assert_equal(chunk_paragraphs([]), [], "empty in, empty out")


# --- BM25 ---------------------------------------------------------------------


def test_bm25_ranks_query_terms_above_noise() -> None:
    chunks = [
        "the okapi bm25 ranking function uses k1 and b parameters",
        "cats are fluffy and sleep most of the day",
        "bm25 term saturation is controlled by the k1 parameter",
        "the weather tomorrow looks rainy with some wind",
    ]
    scores = BM25Scorer().score("bm25 k1 parameter", chunks)
    assert_equal(len(scores), len(chunks), "one score per chunk")
    top = max(range(len(scores)), key=lambda i: scores[i])
    assert_true(top in (0, 2), f"a bm25 chunk ranks first: {scores}")
    assert_true(min(scores[0], scores[2]) > max(scores[1], scores[3]),
                f"both matching chunks beat both noise chunks: {scores}")


def test_bm25_everywhere_term_contributes_nothing() -> None:
    chunks = ["alpha common", "beta common", "gamma common"]
    scores = BM25Scorer().score("common", chunks)
    assert_true(all(abs(s) < 0.2 for s in scores), f"df=N term ~0 idf: {scores}")
    scores_hit = BM25Scorer().score("beta", chunks)
    top = max(range(3), key=lambda i: scores_hit[i])
    assert_equal(top, 1, "rare term dominates")


# --- rerank_prose with a fake scorer -------------------------------------------


def _paragraphs(n: int, size: int = 320) -> list[str]:
    return [f"chunk {i} " + f"word{i} " * (size // 7) for i in range(n)]


def test_rerank_keeps_top_scored_in_original_order() -> None:
    paragraphs = _paragraphs(10)
    # chunk sizes ~320 -> chunk_paragraphs(600) merges pairs -> 5 chunks.
    scores = [0.0, 0.9, 0.1, 0.8, 0.2]
    out = rerank_prose(paragraphs, "some query", budget=1500, scorer=FakeScorer(scores))
    assert_true(out is not None, "rerank ran")
    text = "\n\n".join(out)
    assert_true("chunk 0" in text, "chunk 0 always kept")
    assert_true("chunk 2 " in text and "chunk 3 " in text, "top-scored chunk kept")
    assert_true("not relevant to the query, elided]" in text, "gap marker present")
    assert_true("scorer=fake]" in text, "summary names the scorer")
    # Original order: chunk 0 text precedes chunk 2 text precedes chunk 6 text.
    if "chunk 6 " in text:
        assert_true(text.index("chunk 2 ") < text.index("chunk 6 "), "original order kept")


def test_rerank_budget_is_respected() -> None:
    paragraphs = _paragraphs(20)
    scores = [float(i % 7) for i in range(10)]
    budget = 2000
    out = rerank_prose(paragraphs, "q", budget=budget, scorer=FakeScorer(scores))
    assert_true(out is not None, "rerank ran")
    kept_chars = sum(len(p) for p in out if not p.startswith("["))
    assert_true(kept_chars <= budget + 700, f"kept {kept_chars} <= budget + one chunk")


def test_rerank_gap_marker_counts() -> None:
    paragraphs = _paragraphs(12)
    total = len(chunk_paragraphs(paragraphs))
    # Tiny budget: keep only chunk 0; everything after is one dropped run.
    out = rerank_prose(paragraphs, "q", budget=1, scorer=FakeScorer([0.0] * total))
    assert_true(out is not None, "rerank ran")
    markers = [p for p in out if p.startswith("[… ")]
    assert_true(len(markers) == 1, f"one gap marker for one dropped run: {out}")
    assert_true(f"{total - 1} chunks" in markers[0], f"dropped count correct: {markers[0]}")


def test_rerank_declines_cleanly() -> None:
    paragraphs = _paragraphs(10)
    assert_equal(rerank_prose(paragraphs, "   ", scorer=FakeScorer([1.0] * 9)), None, "no query")
    assert_equal(
        rerank_prose(_paragraphs(2), "q", scorer=FakeScorer([1.0] * 2)), None, "too few chunks"
    )
    assert_equal(
        rerank_prose(paragraphs, "q", budget=10**9, scorer=FakeScorer([1.0] * 9)),
        None,
        "nothing dropped",
    )
    assert_equal(rerank_prose(paragraphs, "q", scorer=RaisingScorer()), None, "fail-open")


def test_min_chunks_constant_sane() -> None:
    assert_true(MIN_CHUNKS >= 2, "MIN_CHUNKS sane")


def test_scorer_mode_selection() -> None:
    try:
        set_default_scorer_mode("bm25")
        assert_true(isinstance(default_scorer(), BM25Scorer), "bm25 mode -> BM25Scorer")
        set_default_scorer_mode("cross-encoder")
        scorer = default_scorer()
        assert_true(
            isinstance(scorer, CrossEncoderScorer),
            "cross-encoder mode -> CrossEncoderScorer",
        )
        # Retired modes ("hybrid"/"neural") and nonsense are no-ops.
        set_default_scorer_mode("hybrid")
        assert_true(default_scorer() is scorer, "retired mode is a no-op")
        set_default_scorer_mode("nonsense")
        assert_true(default_scorer() is scorer, "invalid mode is a no-op")
    finally:
        set_default_scorer_mode("cross-encoder")
        rerank_mod._DEFAULT_SCORER = None


def test_default_mode_is_cross_encoder() -> None:
    assert_equal(rerank_mod._SCORER_MODE, "cross-encoder", "module default mode")
    assert_true(
        "cross-encoder" in rerank_mod.SCORER_MODES and "bm25" in rerank_mod.SCORER_MODES,
        "both surviving modes registered",
    )


# --- CrossEncoderScorer (server /v1/rerank, stubbed transport) -------------------


def _cross_encoder_with_transport(handler):
    """A CrossEncoderScorer whose HTTP layer is an httpx.MockTransport."""
    import httpx

    scorer = CrossEncoderScorer(base_url="http://stub")
    scorer._client = httpx.Client(
        base_url="http://stub", transport=httpx.MockTransport(handler)
    )
    return scorer


def test_cross_encoder_returns_server_scores_in_order() -> None:
    import httpx

    calls = []

    def handler(request):
        import json

        calls.append(json.loads(request.read().decode()))
        return httpx.Response(200, json={"scores": [0.1, 0.9, 0.5]})

    scorer = _cross_encoder_with_transport(handler)
    scores = scorer.score("query", ["a", "b", "c"])
    assert_equal(scores, [0.1, 0.9, 0.5], "server scores passed through in order")
    assert_equal(scorer.name, "cross-encoder", "scorer reports the neural path")
    assert_equal(calls[0]["documents"], ["a", "b", "c"], "chunks sent as documents")
    assert_equal(calls[0]["query"], "query", "query sent verbatim")


def test_cross_encoder_501_latches_dead_and_degrades_to_bm25() -> None:
    import httpx

    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(
            501, json={"error": {"code": "inference_unavailable"}}
        )

    scorer = _cross_encoder_with_transport(handler)
    chunks = ["bm25 k1 parameter details", "unrelated fluffy cats"]
    scores = scorer.score("bm25 k1", chunks)
    assert_equal(scores, BM25Scorer().score("bm25 k1", chunks), "501 degrades to BM25")
    assert_equal(scorer.name, "bm25", "fallback reported honestly")
    scorer.score("bm25 k1", chunks)
    assert_equal(len(calls), 1, "dead latch: no second HTTP call")


def test_cross_encoder_transport_error_fails_open() -> None:
    import httpx

    def handler(request):
        raise httpx.ConnectError("server down")

    scorer = _cross_encoder_with_transport(handler)
    chunks = ["bm25 k1 parameter details", "unrelated fluffy cats"]
    scores = scorer.score("bm25 k1", chunks)
    assert_equal(scores, BM25Scorer().score("bm25 k1", chunks), "connect error -> BM25")
    assert_true(scorer._dead, "transport failure latches dead")


def test_cross_encoder_mismatched_scores_fail_open() -> None:
    import httpx

    def handler(request):
        return httpx.Response(200, json={"scores": [0.5]})  # wrong length

    scorer = _cross_encoder_with_transport(handler)
    chunks = ["bm25 k1 parameter details", "unrelated fluffy cats"]
    scores = scorer.score("bm25 k1", chunks)
    assert_equal(scores, BM25Scorer().score("bm25 k1", chunks), "bad payload -> BM25")


def test_cross_encoder_chunk_cap_skips_http() -> None:
    def handler(request):
        raise AssertionError("HTTP must not be called above MAX_NEURAL_CHUNKS")

    scorer = _cross_encoder_with_transport(handler)
    chunks = [f"chunk {i}" for i in range(rerank_mod.MAX_NEURAL_CHUNKS + 1)]
    scores = scorer.score("query", chunks)
    assert_equal(len(scores), len(chunks), "BM25 covers the oversized set")
    assert_equal(scorer.name, "bm25", "neural path skipped")
    assert_true(not scorer._dead, "cap skip is not a failure")


def test_set_rerank_base_url_resets_default_scorer() -> None:
    try:
        set_default_scorer_mode("cross-encoder")
        first = default_scorer()
        set_rerank_base_url("http://127.0.0.1:9999")
        assert_true(default_scorer() is not first, "base URL change rebuilds scorer")
        again = default_scorer()
        set_rerank_base_url("")  # invalid -> no-op
        assert_true(default_scorer() is again, "empty URL is a no-op")
        set_rerank_base_url("http://127.0.0.1:9999/")  # same after rstrip -> no-op
        assert_true(default_scorer() is again, "unchanged URL is a no-op")
    finally:
        set_rerank_base_url("http://127.0.0.1:8080")
        set_default_scorer_mode("cross-encoder")
        rerank_mod._DEFAULT_SCORER = None


def main() -> None:
    test_chunker_merges_to_target_and_reassembles()
    test_chunker_never_splits_a_paragraph()
    test_chunker_empty_input()
    test_bm25_ranks_query_terms_above_noise()
    test_bm25_everywhere_term_contributes_nothing()
    test_rerank_keeps_top_scored_in_original_order()
    test_rerank_budget_is_respected()
    test_rerank_gap_marker_counts()
    test_rerank_declines_cleanly()
    test_min_chunks_constant_sane()
    test_scorer_mode_selection()
    test_default_mode_is_cross_encoder()
    test_cross_encoder_returns_server_scores_in_order()
    test_cross_encoder_501_latches_dead_and_degrades_to_bm25()
    test_cross_encoder_transport_error_fails_open()
    test_cross_encoder_mismatched_scores_fail_open()
    test_cross_encoder_chunk_cap_skips_http()
    test_set_rerank_base_url_resets_default_scorer()
    print("rerank tests passed")


if __name__ == "__main__":
    main()
