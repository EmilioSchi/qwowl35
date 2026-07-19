#!/usr/bin/env python3
"""Numerical parity gate: qw35-server /v1/rerank vs llama-cpp-python.

Drives the SAME (query, document) pairs through:
  (a) llama-cpp-python with RANK pooling on a reranker GGUF (reference);
      llama.cpp emits softmax([yes, no]) — its out[0] equals qw35's
      sigmoid(z_yes - z_no).
  (b) a running qw35-server started with --reranker-model.

Pass criteria (printed, exit code 0/1):
  * Pearson r > 0.999 across all pairs
  * identical top-1 document for every query
  * max |delta| < 2e-2 — measured cross-implementation numerics: token ids are
    byte-identical (verified via real_reranker_tokenization_dump), but the two
    q8_0 matmul implementations accumulate differently (qw35 dequant-to-f32,
    llama.cpp int8-dot), which shows up as up to ~1e-2 score drift at
    sigmoid-saturated extremes. Each engine is internally stable to ~1e-3.

The pair set deliberately stresses the tokenizer (digits, punctuation runs,
contractions, CJK, code, paths) so a qwen2-vs-qwen35 pretokenizer divergence
would surface as a score mismatch.

Usage:
  python3 tools/rerank_parity.py [--model .gguf/qwen3-reranker-0.6b-q8_0.gguf]
                                 [--server http://127.0.0.1:8080]
                                 [--gpu-layers 0]
"""

import argparse
import json
import math
import sys
import urllib.request

RERANK_TEMPLATE = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, retrieve "
    "relevant passages that answer the query\n<Query>: {query}\n<Document>: {document}"
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# (query, [documents...]) — first document is the intended winner.
CASES = [
    (
        "What is the capital of France?",
        [
            "Paris is the capital and largest city of France.",
            "The mitochondria is the powerhouse of the cell.",
            "Bananas are rich in potassium.",
        ],
    ),
    (
        "How does Rust's borrow checker work?",
        [
            "The borrow checker tracks ownership and lifetimes of references at compile time.",
            "Iron oxidizes into rust when exposed to moisture.",
            "Python uses reference counting with a cyclic garbage collector.",
        ],
    ),
    (
        "fix Metal shader compile error 'function_constant(754)' not set",
        [
            "Pipeline creation fails when a referenced function constant (e.g. index 754) has no value; set it via MTLFunctionConstantValues.",
            "Vulkan validation layers report unbound descriptor sets differently.",
            "Use spotlight to search files on macOS.",
        ],
    ),
    (
        "distanza Terra-Luna in km, valori numerici precisi 384400",
        [
            "La distanza media Terra-Luna e' 384400 km (perigeo 363300 km, apogeo 405500 km).",
            "Il Monte Bianco e' alto 4808 metri.",
            "Nel 1969 l'Apollo 11 allunò; don't confuse it with Apollo 13's 1970 flight!!!",
        ],
    ),
    (
        "什么是量子纠缠?",
        [
            "量子纠缠是两个或多个粒子之间的非经典关联，测量其中一个会立即影响另一个的状态。",
            "长城是中国古代的军事防御工程。",
            "The stock market closed higher on Friday.",
        ],
    ),
    (
        "python function to parse /etc/hosts entries",
        [
            "def parse_hosts(path='/etc/hosts'):\n    with open(path) as f:\n        return [l.split() for l in f if l.strip() and not l.startswith('#')]",
            "The /etc/passwd file stores user account metadata on Unix systems.",
            "JavaScript's fetch() returns a Promise resolving to a Response.",
        ],
    ),
    (
        "who's coming to the 3:30pm stand-up? it's re-scheduled",
        [
            "Reminder: the stand-up moved to 3:30pm today; Ana, Luis and I'll join, Marta can't.",
            "Standing desks reduce back strain according to some studies.",
            "The 5:00am train to Milan doesn't run on Sundays.",
        ],
    ),
]


def llama_scores(model_path: str, gpu_layers: int):
    import llama_cpp

    # n_batch/n_ubatch must cover the longest prompt: llama.cpp RANK pooling
    # silently returns garbage when a prompt spans multiple ubatches.
    llm = llama_cpp.Llama(
        model_path=model_path,
        embedding=True,
        pooling_type=llama_cpp.LLAMA_POOLING_TYPE_RANK,
        n_ctx=2048,
        n_batch=2048,
        n_ubatch=2048,
        n_gpu_layers=gpu_layers,
        verbose=False,
    )
    # Llama.embed() tokenizes with special=False, spelling the template's
    # <|im_start|>/<think> markers out as plain text; force special parsing so
    # the reference sees the framing the model was trained on (qw35 does).
    original_tokenize = llm.tokenize
    llm.tokenize = lambda text, add_bos=True, special=False: original_tokenize(
        text, add_bos=add_bos, special=True
    )
    out = []
    for query, docs in CASES:
        scores = []
        for doc in docs:
            emb = llm.embed(RERANK_TEMPLATE.format(query=query, document=doc))
            scores.append(float(emb[0]))  # softmax P("yes")
        out.append(scores)
    return out


def server_scores(base_url: str):
    out = []
    for query, docs in CASES:
        body = json.dumps({"query": query, "documents": docs}).encode()
        req = urllib.request.Request(
            f"{base_url}/v1/rerank",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read())
        out.append([float(s) for s in payload["scores"]])
    return out


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".gguf/qwen3-reranker-0.6b-q8_0.gguf")
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--gpu-layers", type=int, default=0)
    args = parser.parse_args()

    ref = llama_scores(args.model, args.gpu_layers)
    got = server_scores(args.server)

    flat_ref = [s for case in ref for s in case]
    flat_got = [s for case in got for s in case]
    r = pearson(flat_ref, flat_got)
    max_delta = max(abs(a - b) for a, b in zip(flat_ref, flat_got))
    top1_ok = all(
        max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
        for a, b in zip(ref, got)
    )

    print(f"pairs={len(flat_ref)}  pearson_r={r:.6f}  max|delta|={max_delta:.6f}  top1_match={top1_ok}")
    for (query, _), a, b in zip(CASES, ref, got):
        print(f"  {query[:48]!r:50} llama={['%.4f' % s for s in a]} qw35={['%.4f' % s for s in b]}")

    ok = r > 0.999 and top1_ok and max_delta < 2e-2
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
