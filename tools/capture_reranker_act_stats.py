#!/usr/bin/env python3
"""Drive rerank traffic through /v1/rerank to capture QW35ACT act-stats.

The server's Metal runtime accumulates per-channel mean-abs FFN activations
when QW35_CAPTURE_ACT_OUT is set (the same mechanism used for the 9B AWQ
cook); the reranker engine forces --prefill-chunk 1 in capture mode because
the capture hooks only the single-token eval path. This script simply replays
a rerank corpus so the accumulated statistics reflect REAL rerank prompts.

Recipe (two terminals):

  # 1. start the server in capture mode with the RAW q8_0 reranker; do NOT
  #    send any chat traffic while capturing (the 9B runtime writes nothing
  #    at 0 tokens, so the shared output path stays reranker-only)
  QW35_CAPTURE_ACT_OUT=.gguf/reranker-act-stats.bin \
    ./target/release/qw35 --port 8091 \
    --reranker-model .gguf/qwen3-reranker-0.6b-q8_0.gguf

  # 2. build the corpus once, then replay it (repeat until >= min tokens)
  python3 tools/rerank_corpus.py --out rerank-corpus.jsonl
  python3 tools/capture_reranker_act_stats.py --corpus rerank-corpus.jsonl \
    --server http://127.0.0.1:8091 --min-tokens 100000

The stats file is flushed every 16 captured tokens, so a Ctrl+C of the server
after this script finishes still leaves a complete file. Verify with:
  python3 -c "from qw35_cook_common import read_act_stats; import json; \
    s = read_act_stats('.gguf/reranker-act-stats.bin'); \
    print(s['layers'], s['gu_dim'], s['dn_dim'], s['tokens'])"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def rerank(base_url: str, query: str, documents: list[str]) -> dict:
    body = json.dumps({"query": query, "documents": documents}).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/rerank",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="rerank-corpus.jsonl")
    parser.add_argument("--server", default="http://127.0.0.1:8091")
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=100_000,
        help="Keep replaying the corpus until this many prompt tokens were evaluated",
    )
    args = parser.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if not cases:
        print("empty corpus", file=sys.stderr)
        return 1

    started = time.monotonic()
    total_tokens = 0
    requests = 0
    epoch = 0
    while total_tokens < args.min_tokens:
        epoch += 1
        for case in cases:
            payload = rerank(args.server, case["query"], case["documents"])
            timings = payload.get("qw35_timings", {})
            total_tokens += int(timings.get("prompt_eval_count", 0))
            requests += 1
            if requests % 10 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"epoch {epoch} req {requests}: {total_tokens} tokens "
                    f"({total_tokens / max(elapsed, 1e-9):.0f} tok/s)",
                    file=sys.stderr,
                    flush=True,
                )
            if total_tokens >= args.min_tokens:
                break

    elapsed = time.monotonic() - started
    print(
        f"done: {requests} requests, {total_tokens} prompt tokens in {elapsed:.0f}s",
        file=sys.stderr,
    )
    print(
        "note: token counts above exclude prefix-reused tokens; the capture file "
        "counts every evaluated token. Stop the server (or just leave it; the file "
        "flushes every 16 tokens) and verify with read_act_stats.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
