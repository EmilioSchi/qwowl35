# Qwen3.5-9B model card synopsis

This document extracts the information about the **model** that matters for
local inference with qw35 — architecture, the weight artifacts served, the
reasoning modes, the chat encoding, and the sampling the model expects. It is
about the model, not the engine; for how to build and run the server see
[`README.md`](README.md).

Source: https://huggingface.co/Qwen/Qwen3.5-9B
GGUF source: https://huggingface.co/unsloth/Qwen3.5-9B-GGUF

## Model Family

Qwen3.5-9B is the model qw35 serves. It is the dense, text path of the
Qwen3.5 family (the base model also ships a vision encoder, which qw35 does
not serve).

| Field | Value |
|---|---|
| Parameters | 9B |
| Hidden dimension | 4096 |
| Layers | 32 |
| Token embedding (padded) | 248,320 |
| Context length | 262,144 tokens native (extensible to ~1,010,000 via YaRN) |
| License | Apache-2.0 |

## Architecture

Qwen3.5 is a hybrid that interleaves linear attention with full attention
instead of using full attention at every layer. The 32 layers follow the
repeating block:

```
8 × ( 3 × (Gated DeltaNet → FFN)  →  1 × (Gated Attention → FFN) )
```

So three out of every four layers use **Gated DeltaNet** (a linear-attention
/ state-space style mixer with a constant-size recurrent state), and one in
four uses **Gated Attention** (standard softmax attention with RoPE). This is
why the model can expose a 262K window without a full per-token KV cache in
every layer — only the gated-attention layers carry a growing KV cache.

| Component | Detail |
|---|---|
| Gated DeltaNet | linear-attention heads 32 (V) / 16 (QK), head dim 128 |
| Gated Attention | heads 16 (Q) / 4 (KV), head dim 256, RoPE over 64 dims |
| Feed-forward (FFN) | intermediate dimension 12,288 |

qw35 implements the hybrid directly in Metal. The DeltaNet recurrent state is
kept and checkpointed at history boundaries for the session cache; QK-Norm on
the attention layers keeps activations bounded (measured ≤306 even at 17K
context, ~213× under the fp16 range), so the engine runs in fp16 without
overflow.

## Precision And Weights

qw35 serves the model from a single base GGUF plus an optional decode
sidecar:

| Artifact | Precision / format | Used for |
|---|---|---|
| `Qwen3.5-9B-Q4_K_M.gguf` | Q4_K / Q5_K / Q6_K mixed (Unsloth dynamic) | prefill always; decode without sidecar |
| `Qwen3.5-9B-Q4_K_M.gf4.bin` | GF4 — eight 3-bit quants + fp8 scale per word | single-token decode (default when present) |
| KV cache | `q8_0` (default) or `f16` | attention-layer key/value store |

The GF4 sidecar is a qw35-specific, locally-cooked artifact (not a Hugging
Face download): it trades a small amount of decode-path precision for speed,
while prefill stays on the exact base weights. The `q8_0` KV cache is
byte-identical to `f16` at fp16-parity speed for roughly half the memory.

No benchmark scores from the official card are reproduced here; the one
locally-measured fact worth recording is parity: qw35's greedy (temperature 0)
output is byte-identical to llama.cpp serving the same GGUF.

## Reasoning Modes

Qwen3.5 is a **thinking-capable** model and reasons by default. A turn may
begin with a hidden reasoning block delimited by `<think> … </think>` before
the visible answer.

| Mode | How it is requested | Output shape |
|---|---|---|
| Thinking | default for `thinking-*` profiles; or `enable_thinking: true` / `reasoning_effort` | `<think> … </think>` then answer |
| Non-thinking | `enable_thinking: false` (required to suppress) | answer only |

Because the model thinks by default, a client must explicitly send
`enable_thinking: false` to get a non-thinking response. qw35 also enforces a
**think-token budget** backstop (`reasoning_effort` low/med/high ≈ 4% / 10% /
16% of the output budget) with a `</think>` logit ramp and a grace window, to
keep long coding turns from looping on the close tag.

## Chat Template And Encoding

The source of truth is the **Jinja chat template embedded in the GGUF**
(ChatML-style). The relevant special tokens are:

| Purpose | Token |
|---|---|
| Turn start | `<\|im_start\|>` |
| Turn end | `<\|im_end\|>` |
| End of text | `<\|endoftext\|>` |
| Thinking start / end | `<think>` / `</think>` |
| Tool call block | `<tool_call>` / `</tool_call>` |

Roles are `system`, `user`, `assistant`, and `tool`. A turn is opened with
`<|im_start|>{role}\n` and closed with `<|im_end|>`. Tool definitions are
rendered into the system prompt as a canonical `# Tools` / `<tools>` JSON
block (tools-first), kept inside the stable prefix so the session prefix
cache keeps hitting across agent turns.

Tool **calls** are emitted by the model as compact Qwen3 XML attributes
inside `<tool_call>`, for example:

```text
<tool_call><bash command="pwd"/></tool_call>
```

JSON inside `<tool_call>` is treated as model-format drift and rejected.

## Local Running Notes

The official sampling recommendations, seeded by qw35's `--mode` profiles
(per-request parameters override them):

| Profile (`--mode`) | thinking | temperature | top_p | top_k | presence | repeat |
|---|---|---:|---:|---:|---:|---:|
| `thinking-general`   | on  | 1.0 | 0.95 | 20 | 1.5 | 1.1 |
| `thinking-coding`    | on  | 0.6 | 0.95 | 20 | 0.0 | 1.1 |
| `instruct-general`   | off | 0.7 | 0.80 | 20 | 1.5 | 1.0 |
| `instruct-reasoning` | off | 1.0 | 0.95 | 20 | 1.5 | 1.0 |

When no `--mode` is given, qw35 defaults to an **agentic-coding** profile
(temperature 0.8, presence 0.3, repeat 1.1, thinking off). Use temperature 0
for pure greedy argmax.

Output length: ~8K tokens covers a typical agent turn; raise to 32K+ for
complex math/coding. The model card recommends keeping at least ~128K of
context available to preserve thinking quality.

## Licensing

The model weights are **Qwen3.5-9B** by the Qwen Team, released under the
**Apache-2.0** license. qw35 is an independent inference engine for the model.

## Citation

```bibtex
@misc{qwen3.5,
    title  = {{Qwen3.5}: Towards Native Multimodal Agents},
    author = {{Qwen Team}},
    month  = {February},
    year   = {2026},
    url    = {https://qwen.ai/blog?id=qwen3.5}
}
```
