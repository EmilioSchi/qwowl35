#!/usr/bin/env python3
"""Cross-engine code-quality and repetition-loop comparison harness.

Runs a fixed prompt ladder against any OpenAI-compatible chat endpoint and
scores each response: fenced Python blocks are compile-checked with
py_compile, and the tail of the text is scanned for verbatim repetition
loops. Results land in JSONL, one record per generation, so batteries run
one engine at a time (16 GB machine) and are compared afterwards:

    python3 engine_compare.py --url http://127.0.0.1:8089 --engine-tag qw35-base \
        --out /tmp/engine-compare/qw35-base.jsonl
    python3 engine_compare.py --summarize /tmp/engine-compare/*.jsonl

The greedy L0 cell is deterministic, so --summarize also reports the first
character where each engine pair diverges on it: with two engines serving
the same GGUF, an early divergence point is the sharpest locator for an
inference-chain bug.
"""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROMPTS = [
    {
        "id": "L0-greedy",
        "kind": "loop",
        "temperature": 0.0,
        "max_tokens": 400,
        "runs": 1,  # deterministic
        "text": "make a reverse count from 218 to 156, jumping by 2 one time and other by 4",
    },
    {
        "id": "L0-sampled",
        "kind": "loop",
        "temperature": 0.3,
        "max_tokens": 400,
        "text": "make a reverse count from 218 to 156, jumping by 2 one time and other by 4",
    },
    {
        "id": "L1-easy",
        "kind": "code",
        "temperature": 0.35,
        "max_tokens": 800,
        "text": "Write a python function fizzbuzz(n) that returns the classic FizzBuzz output as a list of strings.",
    },
    {
        "id": "L2-medium",
        "kind": "code",
        "temperature": 0.35,
        "max_tokens": 1200,
        "text": "Write a python script that reads a log file path from argv, counts lines per log level (INFO/WARN/ERROR), handles missing files with try/except, and prints a sorted summary.",
    },
    {
        "id": "L3-hard",
        "kind": "code",
        "temperature": 0.35,
        "max_tokens": 2000,
        "text": "Write a python script with a class FileScanner that recursively lists files, filters by extension, and prints a summary table. Include if/else branches, a loop, and error handling.",
    },
]

LOOP_RE = re.compile(r"(.{12,}?)\1{4,}", re.S)
FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def detect_loop(text: str) -> str | None:
    """Verbatim loop in the response tail; markdown rules are not loops."""
    match = LOOP_RE.search(text[-3000:])
    if not match:
        return None
    fragment = match.group(1)
    if set(fragment.strip()) <= set("=-#*_~ .|"):
        return None
    return fragment


def score_code(text: str) -> dict[str, Any]:
    blocks = FENCE_RE.findall(text)
    compiled = 0
    errors: list[str] = []
    for block in blocks:
        handle = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        handle.write(block)
        handle.close()
        try:
            py_compile.compile(handle.name, doraise=True)
            compiled += 1
        except Exception as exc:  # py_compile raises several types
            errors.append(str(exc)[:160])
        finally:
            os.unlink(handle.name)
    return {"blocks": len(blocks), "compiled": compiled, "errors": errors}


def chat(url: str, prompt: dict[str, Any], extra: dict[str, Any], timeout: float) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt["text"]}],
        "temperature": prompt["temperature"],
        "max_tokens": prompt["max_tokens"],
        "top_p": 0.95,
        "top_k": 20,
        # Sampling parity: qw35's agentic defaults and llama.cpp's own
        # defaults must not leak into the comparison.
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "repeat_penalty": 1.0,  # llama.cpp's native name
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    payload.update(extra)
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run_battery(args: argparse.Namespace) -> int:
    extra: dict[str, Any] = json.loads(args.extra) if args.extra else {}
    if args.model:
        extra["model"] = args.model
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    with out_path.open("w") as out:
        for prompt in PROMPTS:
            runs = prompt.get("runs", args.runs)
            for run in range(1, runs + 1):
                started = time.time()
                try:
                    response = chat(args.url, prompt, extra, args.timeout)
                except (urllib.error.URLError, OSError) as exc:
                    print(f"{args.engine_tag} {prompt['id']} run{run}: REQUEST FAILED: {exc}", file=sys.stderr)
                    return 1
                choice = response["choices"][0]
                message = choice["message"]
                text = (message.get("reasoning_content") or "") + (message.get("content") or "")
                visible = THINK_RE.sub("", text)
                loop = detect_loop(text)
                record: dict[str, Any] = {
                    "engine": args.engine_tag,
                    "prompt": prompt["id"],
                    "kind": prompt["kind"],
                    "run": run,
                    "temperature": prompt["temperature"],
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": response.get("usage", {}).get("completion_tokens"),
                    "elapsed_s": round(time.time() - started, 1),
                    "loop": loop,
                    "text": text,
                }
                if prompt["kind"] == "code":
                    record["code"] = score_code(visible)
                out.write(json.dumps(record) + "\n")
                out.flush()
                records += 1
                summary = f"loop={'YES ' + repr(loop)[:40] if loop else 'none'}"
                if prompt["kind"] == "code":
                    code = record["code"]
                    summary = f"blocks={code['blocks']} compiled={code['compiled']}"
                print(
                    f"{args.engine_tag} {prompt['id']} run{run}: finish={record['finish_reason']}"
                    f" tokens={record['completion_tokens']} {summary} ({record['elapsed_s']}s)"
                )
    print(f"{records} records -> {out_path}")
    return 0


def summarize(paths: list[str]) -> int:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path) as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    engines = sorted({row["engine"] for row in rows})
    prompts = [prompt["id"] for prompt in PROMPTS]

    print(f"{'prompt':<12}" + "".join(f"{engine:>22}" for engine in engines))
    for prompt_id in prompts:
        cells = []
        for engine in engines:
            sub = [row for row in rows if row["engine"] == engine and row["prompt"] == prompt_id]
            if not sub:
                cells.append("-")
            elif sub[0]["kind"] == "code":
                ok = sum(1 for row in sub if row["code"]["blocks"] > 0 and row["code"]["compiled"] == row["code"]["blocks"])
                cells.append(f"compile {ok}/{len(sub)}")
            else:
                loops = sum(1 for row in sub if row["loop"])
                cells.append(f"loops {loops}/{len(sub)}")
        print(f"{prompt_id:<12}" + "".join(f"{cell:>22}" for cell in cells))

    greedy = {row["engine"]: row for row in rows if row["prompt"] == "L0-greedy"}
    names = sorted(greedy)
    if len(names) > 1:
        print("\nL0-greedy first divergence (same-GGUF engines should track closely):")
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                ta, tb = greedy[a]["text"], greedy[b]["text"]
                pos = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
                same = "IDENTICAL" if ta == tb else f"diverge at char {pos}/{max(len(ta), len(tb))}"
                print(f"  {a} vs {b}: {same}")
                if ta != tb:
                    print(f"    {a}: ...{ta[max(0, pos - 40):pos + 40]!r}")
                    print(f"    {b}: ...{tb[max(0, pos - 40):pos + 40]!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="OpenAI-compatible server base URL")
    parser.add_argument("--engine-tag", help="Label stored with each record")
    parser.add_argument("--model", default=None, help="Model id to send (some servers require it)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--extra", default=None, help="JSON merged into every request payload")
    parser.add_argument("--summarize", nargs="+", default=None, metavar="JSONL")
    args = parser.parse_args()

    if args.summarize:
        return summarize(args.summarize)
    if not (args.url and args.engine_tag and args.out):
        parser.error("--url, --engine-tag and --out are required to run a battery")
    return run_battery(args)


if __name__ == "__main__":
    sys.exit(main())
