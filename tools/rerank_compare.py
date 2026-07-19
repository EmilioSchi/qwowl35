#!/usr/bin/env python3
"""Reranker speed + quality comparison: llama-cpp-python vs qw35-server.

Three-way benchmark over a rerank corpus (tools/rerank_corpus.py):
  (a) llama-cpp-python, RANK pooling, on the RAW q8_0 reranker (reference)
  (b) qw35-server /v1/rerank serving the raw q8_0 file
  (c) qw35-server /v1/rerank serving the cooked GF4+AWQ Qwowl3-Reranker

Speed: per-request wall latency p50/p90/p99 bucketed by document count,
per-doc ms, and the server's own prompt_eval_tps / prefix-reuse counters.
Quality: per-query Spearman + Pearson between engines, top-k overlap
(k in {1,3,5}), mean |score delta|. Spearman is reported twice: over ALL
documents, and over the RELEVANT subset (docs where either engine scores
> 0.05) — rank order among the many near-zero irrelevant docs is numeric
noise and never affects what the compress lane keeps, so the gates read the
relevant-subset figure.

Gates (exit code):
  (b) vs (a): relevant Spearman > 0.995  — server math is right
  (c) vs (b): relevant Spearman >= 0.97, top-3 overlap >= 0.90,
              top-1 agreement >= 0.90    — the GF4 cook holds quality

Run each engine PASS SEPARATELY against a freshly-started server (one model
loaded at a time); results accumulate in --state so the final report compares
whatever passes have been recorded:

  python3 tools/rerank_corpus.py --out rerank-corpus.jsonl --docs-per-query 16
  # llama reference (no server needed):
  python3 tools/rerank_compare.py --corpus rerank-corpus.jsonl --state cmp.json \
      run-llama --model .gguf/qwen3-reranker-0.6b-q8_0.gguf
  # server passes (start qw35 --reranker-model <file> first):
  python3 tools/rerank_compare.py --corpus rerank-corpus.jsonl --state cmp.json \
      run-server --label server-q8_0 --server http://127.0.0.1:8091
  python3 tools/rerank_compare.py --corpus rerank-corpus.jsonl --state cmp.json \
      run-server --label server-gf4 --server http://127.0.0.1:8091
  # report + gates:
  python3 tools/rerank_compare.py --state cmp.json report
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.request

RERANK_TEMPLATE = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, retrieve "
    "relevant passages that answer the query\n<Query>: {query}\n<Document>: {document}"
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def load_corpus(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_state(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"passes": {}}


def save_state(path: str, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f)


def run_llama(args, corpus: list[dict]) -> dict:
    import llama_cpp

    # n_batch/n_ubatch must cover the LONGEST prompt: llama.cpp RANK pooling
    # silently returns garbage when a prompt spans multiple ubatches (measured:
    # a 592-token prompt scores 0.006 at n_batch=512 vs 0.999 at 4096).
    llm = llama_cpp.Llama(
        model_path=args.model,
        embedding=True,
        pooling_type=llama_cpp.LLAMA_POOLING_TYPE_RANK,
        n_ctx=2048,
        n_batch=2048,
        n_ubatch=2048,
        n_gpu_layers=args.gpu_layers,
        verbose=False,
    )
    # Llama.embed() tokenizes with special=False, which spells the template's
    # <|im_start|>/<think> markers out as plain text — NOT the framing the
    # reranker was trained on (and not what qw35 or llama.cpp's own /rerank
    # server do). Force special-token parsing for the embed path.
    original_tokenize = llm.tokenize
    llm.tokenize = lambda text, add_bos=True, special=False: original_tokenize(
        text, add_bos=add_bos, special=True
    )
    results = []
    for case in corpus:
        started = time.monotonic()
        scores = []
        for doc in case["documents"]:
            emb = llm.embed(RERANK_TEMPLATE.format(query=case["query"], document=doc))
            scores.append(float(emb[0]))  # softmax P("yes")
        wall_ms = (time.monotonic() - started) * 1000.0
        results.append({"docs": len(case["documents"]), "scores": scores, "wall_ms": wall_ms})
        print(f"  llama {len(results)}/{len(corpus)} ({wall_ms:.0f}ms)", file=sys.stderr, flush=True)
    return {"engine": "llama-cpp-python", "results": results}


def run_server(args, corpus: list[dict]) -> dict:
    results = []
    for case in corpus:
        body = json.dumps({"query": case["query"], "documents": case["documents"]}).encode()
        req = urllib.request.Request(
            f"{args.server}/v1/rerank",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read())
        wall_ms = (time.monotonic() - started) * 1000.0
        results.append(
            {
                "docs": len(case["documents"]),
                "scores": [float(s) for s in payload["scores"]],
                "wall_ms": wall_ms,
                "qw35_timings": payload.get("qw35_timings", {}),
            }
        )
        print(f"  {args.label} {len(results)}/{len(corpus)} ({wall_ms:.0f}ms)", file=sys.stderr, flush=True)
    return {"engine": args.label, "model": payload.get("model", ""), "results": results}


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        for rank, idx in enumerate(order):
            r[idx] = float(rank)
        return r

    return pearson(ranks(xs), ranks(ys))


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def top_k(scores: list[float], k: int) -> set[int]:
    return set(sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k])


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(pct / 100.0 * (len(values) - 1)))))
    return values[idx]


def speed_report(name: str, results: list[dict]) -> None:
    by_docs: dict[int, list[float]] = {}
    for r in results:
        by_docs.setdefault(r["docs"], []).append(r["wall_ms"])
    print(f"\n[{name}] latency by document count:")
    for docs in sorted(by_docs):
        lat = by_docs[docs]
        per_doc = statistics.mean(lat) / docs
        print(
            f"  {docs:3d} docs: p50={percentile(lat, 50):8.1f}ms "
            f"p90={percentile(lat, 90):8.1f}ms p99={percentile(lat, 99):8.1f}ms "
            f"({per_doc:6.1f} ms/doc, n={len(lat)})"
        )
    tps = [r["qw35_timings"].get("prompt_eval_tps", 0.0) for r in results if "qw35_timings" in r]
    reused = [r["qw35_timings"].get("prefix_reused_tokens", 0) for r in results if "qw35_timings" in r]
    if tps and any(tps):
        print(f"  server prompt_eval_tps: mean={statistics.mean(tps):.0f} "
              f"p50={percentile(tps, 50):.0f}; prefix_reused/req p50={percentile([float(x) for x in reused], 50):.0f}")


RELEVANT_SCORE_FLOOR = 0.05


def quality_pair(a: list[dict], b: list[dict], label: str) -> dict:
    spearmans, rel_spearmans, pearsons, deltas = [], [], [], []
    top1 = top3 = top5_overlap = 0.0
    n = 0
    for ra, rb in zip(a, b):
        xa, xb = ra["scores"], rb["scores"]
        if len(xa) != len(xb) or len(xa) < 2:
            continue
        n += 1
        spearmans.append(spearman(xa, xb))
        pearsons.append(pearson(xa, xb))
        deltas.extend(abs(x - y) for x, y in zip(xa, xb))
        relevant = [i for i in range(len(xa)) if max(xa[i], xb[i]) > RELEVANT_SCORE_FLOOR]
        if len(relevant) >= 2:
            rel_spearmans.append(
                spearman([xa[i] for i in relevant], [xb[i] for i in relevant])
            )
        top1 += 1.0 if top_k(xa, 1) == top_k(xb, 1) else 0.0
        top3 += len(top_k(xa, 3) & top_k(xb, 3)) / 3.0
        top5_overlap += len(top_k(xa, 5) & top_k(xb, 5)) / 5.0
    out = {
        "label": label,
        "queries": n,
        "spearman_mean_all": statistics.mean(spearmans) if spearmans else float("nan"),
        "spearman_mean_relevant": statistics.mean(rel_spearmans) if rel_spearmans else float("nan"),
        "pearson_mean": statistics.mean(pearsons) if pearsons else float("nan"),
        "top1_agreement": top1 / n if n else float("nan"),
        "top3_overlap": top3 / n if n else float("nan"),
        "top5_overlap": top5_overlap / n if n else float("nan"),
        "mean_abs_delta": statistics.mean(deltas) if deltas else float("nan"),
    }
    print(f"\n[{label}] queries={n}")
    for key in (
        "spearman_mean_all",
        "spearman_mean_relevant",
        "pearson_mean",
        "top1_agreement",
        "top3_overlap",
        "top5_overlap",
        "mean_abs_delta",
    ):
        print(f"  {key} = {out[key]:.4f}")
    return out


def report(args, state: dict) -> int:
    passes = state["passes"]
    for name, data in passes.items():
        speed_report(name, data["results"])

    ok = True
    if "llama" in passes and "server-q8_0" in passes:
        q = quality_pair(passes["server-q8_0"]["results"], passes["llama"]["results"], "server-q8_0 vs llama")
        if not q["spearman_mean_relevant"] > 0.995:
            print("  GATE FAIL: server-q8_0 vs llama relevant spearman <= 0.995")
            ok = False
    if "server-q8_0" in passes and "server-gf4" in passes:
        q = quality_pair(passes["server-gf4"]["results"], passes["server-q8_0"]["results"], "server-gf4 vs server-q8_0")
        gates = [
            (q["spearman_mean_relevant"] >= 0.97, "spearman_mean_relevant >= 0.97"),
            (q["top3_overlap"] >= 0.90, "top3_overlap >= 0.90"),
            (q["top1_agreement"] >= 0.90, "top1_agreement >= 0.90"),
        ]
        for passed, label in gates:
            if not passed:
                print(f"  GATE FAIL: server-gf4 vs server-q8_0 {label}")
                ok = False
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="rerank-corpus.jsonl")
    parser.add_argument("--state", default="rerank-compare.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_llama = sub.add_parser("run-llama", help="reference pass via llama-cpp-python")
    p_llama.add_argument("--model", default=".gguf/qwen3-reranker-0.6b-q8_0.gguf")
    p_llama.add_argument("--gpu-layers", type=int, default=-1)

    p_server = sub.add_parser("run-server", help="pass via a running qw35 /v1/rerank")
    p_server.add_argument("--server", default="http://127.0.0.1:8091")
    p_server.add_argument("--label", required=True, choices=["server-q8_0", "server-gf4"])

    sub.add_parser("report", help="print speed + quality report and evaluate gates")

    args = parser.parse_args()
    state = load_state(args.state)

    if args.cmd == "run-llama":
        state["passes"]["llama"] = run_llama(args, load_corpus(args.corpus))
        save_state(args.state, state)
        return 0
    if args.cmd == "run-server":
        state["passes"][args.label] = run_server(args, load_corpus(args.corpus))
        save_state(args.state, state)
        return 0
    return report(args, state)


if __name__ == "__main__":
    sys.exit(main())
