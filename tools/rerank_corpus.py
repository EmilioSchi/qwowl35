#!/usr/bin/env python3
"""Build a deterministic rerank calibration/benchmark corpus from real repo text.

Emits JSONL lines of `{"query": str, "documents": [str, ...]}` shaped like the
agent's compress lane feeds the reranker: ~600-char chunks of real technical
material (docs, source code, build scripts), a mix of relevant and irrelevant
per query. Used both to capture QW35ACT activation stats for the AWQ cook
(tools/capture_reranker_act_stats.py) and as the speed/quality benchmark input
(tools/rerank_compare.py).

Deterministic: fixed seed, sorted file walks — the same tree yields the same
corpus.

Usage:
  python3 tools/rerank_corpus.py [--out rerank-corpus.jsonl] [--docs-per-query 16]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Text sources: (path glob root, patterns). Real prose + code + logs-ish
# content, matching what the compress lane actually reranks.
SOURCES = [
    ("", ["README.md", "MODEL_CARD.md", "AGENTS.md"]),
    ("qw35-agent/qwowl35", ["**/*.py"]),
    ("qw35-agent/qwowl35/tools/compress", ["*.md"]),
    ("qw35-server/src", ["*.rs", "metal/*.metal", "metal/bridge/*.m"]),
    ("tools", ["*.py"]),
]

# Realistic agent-style queries. Each maps to keywords used to pick the
# "relevant" chunks; the rest of the documents are drawn from elsewhere.
QUERIES = [
    ("how is the KV cache grown and what are slabs", ["slab", "kv"]),
    ("where does the server validate the GGUF model", ["validate", "gguf"]),
    ("how does the GF4 quantization pack weights", ["gf4", "scale"]),
    ("what does the AWQ fold do to the norms", ["awq", "norm"]),
    ("how are tool results compressed before reaching the model", ["compress", "chunk"]),
    ("how does BM25 scoring work in the rerank lane", ["bm25", "score"]),
    ("where is the chat template rendered for the 9B", ["prompt", "im_start"]),
    ("how does the Metal attention kernel apply RoPE", ["rope", "attention"]),
    ("what CLI flags does the qw35 server accept", ["--", "flag"]),
    ("how is the residency set used to pin weights", ["residency", "pin"]),
    ("what is the checkpoint stack and when does it rewind", ["checkpoint", "rewind"]),
    ("how does the SSM recurrent state get exported", ["ssm", "state"]),
    ("come viene calcolato il punteggio di rilevanza yes/no", ["yes", "score"]),
    ("what quantization types does the Metal backend support", ["q4_k", "q8_0"]),
    ("how are prefill chunks split at slab boundaries", ["prefill", "chunk"]),
    ("where does the tokenizer handle byte fallback and merges", ["merge", "token"]),
    ("how does the fail-open guard fall back to bm25", ["fail", "fallback"]),
    ("what timings are reported per generation request", ["timings", "duration"]),
    ("how do I run the download script to fetch the model", ["download", "model"]),
    ("what does the sampler do with repetition penalties", ["penalt", "sampl"]),
]

CHUNK_TARGET_CHARS = 600


def iter_source_files() -> list[Path]:
    files: list[Path] = []
    for root, patterns in SOURCES:
        base = REPO / root if root else REPO
        for pattern in patterns:
            files.extend(sorted(base.glob(pattern)))
    seen = set()
    out = []
    for f in files:
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        out.append(f)
    return out


def chunk_text(text: str) -> list[str]:
    """Greedy ~600-char chunks on paragraph boundaries (the compress lane's
    chunking shape), never splitting a paragraph."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip("\n")
        if current and len(current) + len(para) + 2 > CHUNK_TARGET_CHARS:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    # Drop tiny fragments and cap pathological chunks.
    return [c[:2000] for c in chunks if len(c) >= 120]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="rerank-corpus.jsonl")
    parser.add_argument("--docs-per-query", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1,
                        help="Emit the query set this many times with different "
                        "document samples (calibration wants volume)")
    args = parser.parse_args()

    rng = random.Random(0x9735)
    all_chunks: list[str] = []
    for path in iter_source_files():
        try:
            all_chunks.extend(chunk_text(path.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    if len(all_chunks) < args.docs_per_query * 2:
        print(f"not enough source chunks ({len(all_chunks)})", file=sys.stderr)
        return 1

    lines = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for _ in range(max(1, args.repeats)):
            for query, keywords in QUERIES:
                relevant = [
                    c for c in all_chunks
                    if all(k.lower() in c.lower() for k in keywords)
                ]
                rng.shuffle(relevant)
                docs = relevant[: max(2, args.docs_per_query // 4)]
                while len(docs) < args.docs_per_query:
                    candidate = all_chunks[rng.randrange(len(all_chunks))]
                    if candidate not in docs:
                        docs.append(candidate)
                rng.shuffle(docs)
                out.write(json.dumps({"query": query, "documents": docs}) + "\n")
                lines += 1

    total_chars = sum(len(c) for c in all_chunks)
    print(
        f"wrote {lines} queries x {args.docs_per_query} docs to {args.out} "
        f"(pool: {len(all_chunks)} chunks, {total_chars/1e6:.1f} MB)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
