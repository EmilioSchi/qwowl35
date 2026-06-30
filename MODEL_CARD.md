---
tags:
  - qwen3.5
  - qwowl3.5
  - reasoning
  - long-context
  - function-calling
  - tool-use
  - agentic
library_name: qwowl35
license: apache-2.0
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3.5-9B
---

# Qwowl3.5-9B

> **Built with [qwowl35](https://github.com/EmilioSchi/qwowl35)** to optimize
> inference on Apple Silicon. The **qwowl35** cooking tool bakes a base Qwen3.5
> GGUF (Q4_K_M / Q5_K / Q6_K FFN) into a single self-contained `.gguf` whose FFN
> gate/up/down tensors are stored as GF4 (eight 3-bit quants plus an fp8 scale
> per 32-bit word, GGUF type-id 100) and whose post-attention norms carry an AWQ
> per-channel scale fold. Every other tensor is copied verbatim. The result is
> what the server loads by default — `Qwowl3.5-9B.gguf`, cooked from
> `Qwen3.5-9B-Q4_K_M.gguf`.

**Qwowl3.5-9B** is the dense, text path of Qwen3.5-9B re-packed for local
inference with [qw35](https://github.com/EmilioSchi/qwowl35), a Metal-backed
engine built for agentic loops on a MacBook. It is the *same* model as the base
GGUF — same weights, same tokenizer, same chat template — nothing is retrained.
The only change is how the FFN weights are stored (GF4, a uniform 4-bit grouped
floating-point format) and the folding of a one-time AWQ calibration scale into
the post-attention norms, so the engine serves both prefill and decode from one
mmap. It is a thinking-capable, tool-calling model: a turn may open with a hidden
`<think> … </think>` reasoning block, and tool calls are emitted as compact
Qwen3 XML inside `<tool_call>`. It keeps Qwen3.5's 262K native context (the
server defaults to a 128K window for the throughput sweet spot).

> [!Note]
> This repository contains the model weights as a custom GGUF file.
>
> `Qwowl3.5-9B.gguf` is a **qw35-specific** artifact: its FFN tensors use GF4
> (GGUF type-id 100), a custom quantization that standard GGUF loaders
> (e.g. stock llama.cpp) do not understand. It is built for qw35's **Metal**
> runtime and therefore runs **only on Apple Silicon** (M1 or newer, macOS 14+).
> There is no CUDA, CPU, or cross-platform path.

## Cooking details

The cook touches only two families of tensors and copies everything else
through untouched:

| Tensors | Base `Qwen3.5-9B-Q4_K_M.gguf` | Unified `Qwowl3.5-9B.gguf` |
|---|---|---|
| FFN `ffn_gate` / `ffn_up` / `ffn_down` (96 tensors) | Q4_K / Q6_K | **GF4** (type-id 100): eight 3-bit quants + one fp8 (e5m2) scale per 32-bit word, 4 bpw |
| `post_attention_norm` (32 tensors) | as shipped | **AWQ-folded**: the inverse of the per-input-channel AWQ scale is baked into the norm weights |
| Everything else (attention, Gated DeltaNet, embeddings, other norms, KV metadata, chat template) | — | **copied verbatim** |

GF4 drives both the single-token decode matvec and the tiled multi-token prefill
matmul, so there is no separate sidecar and no duplicate Q4_K FFN. AWQ scales the
*salient* input channels up before GF4 quantization (giving them finer
resolution) and folds the inverse into the preceding norm, so the layer's
full-precision output is unchanged while the quantized weights spend their bits
where the activations are largest.

**Quality** is measured by teacher-forcing the calibration corpus through both
the base GGUF and the cooked model and comparing logits
(`real_model_unified_quality_report`, M2 16 GB). AWQ-GF4 wins on every axis at
identical decode speed, which is why it is the canonical build:

| Cooked variant | top-1 agreement vs base | mean KL | argmax flips |
|---|---|---:|---:|
| **AWQ-GF4 (`Qwowl3.5-9B.gguf`)** | **89.8%** | **0.038** | **5/49** |
| plain GF4 (no AWQ) | 85.7% | 0.062 | 7/49 |

| Property | Value |
|---|---|
| Predominant precision | 4-bit (GF4 FFN, uniform 4 bpw) |
| AWQ scale | `s_c = act_c^0.6`, geomean-normalised, folded into post-attention norms |
| KV cache | `q8_0` (default) or `f16` |
| Decode throughput | ~17 tok/s (M2, 128K ctx) |
| Greedy parity | byte-identical to llama.cpp on the same base GGUF |
| Base size | 5,680,522,464 B (≈ 5.29 GiB) |
| Unified size | 5,170,914,624 B (≈ 4.82 GiB), ~0.47 GiB smaller |

The unified file is smaller because GF4 is a flat 4 bpw versus Q4_K's ~4.5 bpw,
and the AWQ fold adds no bytes (it reuses the existing norm tensors).

## Usage

Grab the cooked weights from this Hugging Face repo (or
[cook them yourself](#cook-by-yourself-optional)) and drop the file in `.gguf/`
so the server's default path resolves:

```bash
huggingface-cli download EmilioSchi/Qwowl3.5-9B Qwowl3.5-9B.gguf --local-dir .gguf
```

Then load it with `qw35` and use it as usual. Build and run the server:

```bash
make run                                    # or: cargo run --release -p qw35-server --bin qw35
```

or compile once and run the binary directly:

```bash
cargo build --release
./target/release/qw35
```

The server seeds its sampling and think/no-think defaults from an official
Qwen3.5 profile, selectable with `--mode` (default: `thinking-coding`):

```bash
qw35 --mode thinking-coding
```

| Profile (`--mode`) | thinking | temperature | top_p | top_k | presence | repeat |
|---|---|---:|---:|---:|---:|---:|
| `thinking-general`   | on  | 1.0 | 0.95 | 20 | 1.5 | 1.1 |
| `thinking-coding`    | on  | 0.6 | 0.95 | 20 | 0.0 | 1.1 |
| `instruct-general`   | off | 0.7 | 0.80 | 20 | 1.5 | 1.0 |
| `instruct-reasoning` | off | 1.0 | 0.95 | 20 | 1.5 | 1.0 |

Pick the profile by mode and task type (per-request parameters override them):

- **Thinking mode, general tasks** → `thinking-general`
- **Thinking mode, precise coding (e.g. WebDev)** → `thinking-coding`
- **Instruct (non-thinking) mode, general tasks** → `instruct-general`
- **Instruct (non-thinking) mode, reasoning tasks** → `instruct-reasoning`

In another terminal, launch the **qwowl35** TUI agent (it talks to the server on
`127.0.0.1:8080`):

```bash
pip install -r qw35-agent/requirements.txt    # textual, rich, httpx, xxhash
cd qw35-agent && python -m qwowl35
```

## Cook by yourself (optional)

The cooked `Qwowl3.5-9B.gguf` is published on Hugging Face, so most users can
just download it (see [Usage](#usage)). Cooking it yourself is optional — useful
if you want to re-quantize from a different base GGUF or reproduce the build. The
downloader script wraps the whole flow:

```bash
./download_model.sh         # base GGUF -> .gguf/Qwen3.5-9B-Q4_K_M.gguf (~5.3 GB)
./download_model.sh cook    # cook the unified Qwowl3.5-9B.gguf (CPU-heavy)
./download_model.sh all     # both
```

`cook` runs the AWQ-GF4 cooker, which needs the AWQ activation statistics
(`act-stats.bin`, per-channel mean-abs FFN activations captured once from the
base model over a calibration corpus):

```bash
# capture activation stats first (if act-stats.bin is missing)
cargo test -p qw35-server --lib real_model_capture_activations -- --ignored

# then cook (this is what `download_model.sh cook` invokes)
python3 tools/cook_qw35_awq_gf4.py \
    .gguf/Qwen3.5-9B-Q4_K_M.gguf .gguf/Qwowl3.5-9B.gguf --awq .gguf/act-stats.bin
```

Per FFN tensor, the cooker dequantizes the Q4_K/Q5_K/Q6_K source → multiplies
each input-channel column by its AWQ scale → re-packs as GF4 → folds `1/s_c` into
that layer's `post_attention_norm`. Every non-FFN tensor is streamed through
untouched, one tensor at a time, so peak RAM is a single tensor. It is CPU-heavy
and takes a few minutes. Requires `python3` + `numpy` + `gguf`.

## Licensing

The model weights are **Qwowl3.5-9B** by Emilio Schininà, cooked from the Qwen
Team's model and released under the **Apache-2.0** license. qw35 is an
independent inference engine for the model.

### Citation

```bibtex
@misc{qwen3.5,
    title  = {{Qwen3.5}: Towards Native Multimodal Agents},
    author = {{Qwen Team}},
    month  = {February},
    year   = {2026},
    url    = {https://qwen.ai/blog?id=qwen3.5}
}
```
