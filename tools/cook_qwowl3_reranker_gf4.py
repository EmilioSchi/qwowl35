#!/usr/bin/env python3
"""Cook the unified qw35 reranker: Qwen3-Reranker GGUF -> Qwowl3-Reranker.

The reranker twin of `cook_qw35_awq_gf4.py`, sharing all machinery through
`qw35_cook_common.py` but with the dense-Qwen3 recipe:

  * FFN gate/up/down baked as GF4 (type-id 100) from a Q8_0 (or K-quant)
    source — the same 256-element super-block layout the server's GF4
    decode/tiled kernels already execute;
  * AWQ per-channel scales on the gate/up columns, inverse folded into the
    pre-FFN norm — which a dense Qwen3 names `ffn_norm.weight` (the 9B hybrid
    calls it `post_attention_norm.weight`);
  * `--scale-search` ON by default: a 0.6B has less redundancy than the 9B,
    so the per-group min-MSE e5m2 clip is worth the cook-time cost;
  * everything else (attention, embeddings, norms, the yes/no `cls.output`
    head, tokenizer/metadata) copied verbatim — the arch stays an honest
    `qwen3`; only `general.name` is re-badged and `qwowl.cook.*` provenance
    keys are added.

The 9B cooker and its output are untouched by this script.

Usage:
  cook_qwowl3_reranker_gf4.py .gguf/qwen3-reranker-0.6b-q8_0.gguf \
      .gguf/Qwowl3-Reranker-0.6B.gguf --awq .gguf/reranker-act-stats.bin
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf import GGUFReader  # noqa: E402

from qw35_cook_common import (  # noqa: E402
    F32_GGUF_TYPE,
    FFN_RE,
    GF4_GGUF_TYPE,
    Q4_K_TYPE,
    Q5_K_TYPE,
    Q6_K_TYPE,
    Q8_0_TYPE,
    awq_col_scale,
    copy_metadata,
    pack_ffn_gf4,
    read_act_stats,
    verify_unified,
)

COOKED_NAME = "Qwowl3-Reranker-0.6B"
PRE_FFN_NORM = "ffn_norm.weight"  # dense Qwen3 naming
FFN_SRC_TYPES = (Q8_0_TYPE, Q4_K_TYPE, Q5_K_TYPE, Q6_K_TYPE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cook the unified qw35 reranker: dense Qwen3 GGUF -> FFN baked as GF4 + AWQ norm fold."
    )
    parser.add_argument("model", help="Source Qwen3-Reranker GGUF (rank conversion; Q8_0 or K-quant)")
    parser.add_argument("output", help="Output unified reranker .gguf path")
    parser.add_argument(
        "--awq",
        default=None,
        metavar="ACT_STATS",
        help="QW35ACT stats captured from REAL rerank prompts (serve the raw"
        " reranker with QW35_CAPTURE_ACT_OUT and replay a corpus, see"
        " capture_reranker_act_stats.py). Scales salient gate/up input channels"
        " and folds the inverse into ffn_norm.",
    )
    parser.add_argument(
        "--awq-alpha",
        type=float,
        default=0.6,
        help="AWQ scale exponent s_c = act_c^alpha (geomean-normalised). Default 0.6.",
    )
    parser.add_argument(
        "--no-scale-search",
        action="store_true",
        help="Disable the per-group min-MSE e5m2 clip search (ON by default for"
        " the 0.6B; the 9B cooker defaults the other way).",
    )
    parser.add_argument(
        "--name",
        default=COOKED_NAME,
        help=f"general.name stamped into the cooked file. Default: {COOKED_NAME}",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=64,
        help="Rows to dequantize and quantize at a time",
    )
    return parser.parse_args()


def cook_reranker_unified(args: argparse.Namespace) -> int:
    from gguf import GGUFValueType, GGUFWriter

    model = Path(args.model)
    out_path = Path(args.output)
    scale_search = not args.no_scale_search
    reader = GGUFReader(str(model))

    arch = str(reader.fields["general.architecture"].contents())
    if arch != "qwen3":
        raise SystemExit(f"{model}: general.architecture is {arch!r}, expected qwen3 (dense reranker)")
    if "qwen3.pooling_type" not in reader.fields:
        raise SystemExit(f"{model}: missing qwen3.pooling_type — not a rank-converted reranker GGUF")

    # AWQ pre-pass: derive the per-input-channel gate/up scale per layer and
    # fold base_norm/s into ffn_norm (the dense pre-FFN norm), mirroring the
    # 9B recipe (down stays plain GF4).
    awq_scales: dict[int, np.ndarray] = {}
    awq_norms: dict[str, np.ndarray] = {}
    if args.awq:
        stats = read_act_stats(args.awq)
        norm_by_name = {t.name: t for t in reader.tensors}
        gate_re = re.compile(r"^blk\.(\d+)\.ffn_gate\.weight$")
        layers = sorted(int(gate_re.match(t.name).group(1)) for t in reader.tensors if gate_re.match(t.name))
        for layer in layers:
            if layer >= stats["layers"]:
                continue
            norm_t = norm_by_name.get(f"blk.{layer}.{PRE_FFN_NORM}")
            if norm_t is None:
                continue
            base_norm = np.asarray(norm_t.data, dtype=np.float32).reshape(-1)
            if base_norm.shape[0] != stats["gu_dim"]:
                continue
            s = awq_col_scale(stats["gateup"][layer], args.awq_alpha)
            awq_scales[layer] = s
            awq_norms[f"blk.{layer}.{PRE_FFN_NORM}"] = (base_norm / s).astype(np.float32)
        print(f"AWQ: alpha={args.awq_alpha} layers={len(awq_scales)} folded into ffn_norm", file=sys.stderr)

    align_field = reader.fields.get("general.alignment")
    alignment = int(align_field.contents()) if align_field is not None else 32

    gate_up_re = re.compile(r"^blk\.(\d+)\.ffn_(?:gate|up)\.weight$")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(path=str(out_path), arch=arch)
    writer.data_alignment = alignment
    kv_count = copy_metadata(
        writer,
        reader,
        overrides={"general.name": args.name},
        extra={
            "qwowl.cook.recipe": ("awq-gf4" if args.awq else "gf4", GGUFValueType.STRING),
            "qwowl.cook.source": (model.name, GGUFValueType.STRING),
            "qwowl.cook.awq_alpha": (float(args.awq_alpha if args.awq else 0.0), GGUFValueType.FLOAT32),
            "qwowl.cook.scale_search": (bool(scale_search), GGUFValueType.BOOL),
        },
    )

    def classify(t: Any) -> tuple[int, int]:
        src_type = int(t.tensor_type)
        if FFN_RE.match(t.name) and src_type in FFN_SRC_TYPES:
            cols, rows = int(t.shape[0]), int(t.shape[1])
            if cols % 256 != 0:
                raise SystemExit(
                    f"{t.name}: GF4 tiled prefill needs k % 256 == 0, got {cols}"
                )
            return GF4_GGUF_TYPE, rows * (cols // 8) * 4
        if t.name in awq_norms:
            return F32_GGUF_TYPE, int(awq_norms[t.name].size) * 4
        return src_type, int(np.asarray(t.data).nbytes)

    # Pass 1 — declare every tensor (see the 9B cooker for the writer-shape
    # and raw_dtype conventions; identical here).
    n_gf4 = 0
    for t in reader.tensors:
        out_type, nbytes = classify(t)
        if out_type == GF4_GGUF_TYPE:
            n_gf4 += 1
        writer_shape = tuple(reversed(tuple(int(d) for d in t.shape)))
        writer.add_tensor_info(t.name, writer_shape, np.dtype(np.uint32), nbytes, raw_dtype=out_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # Pass 2 — stream tensor bytes in order, one at a time.
    for idx, t in enumerate(reader.tensors, 1):
        src_type = int(t.tensor_type)
        if FFN_RE.match(t.name) and src_type in FFN_SRC_TYPES:
            m = gate_up_re.match(t.name)
            col_scale = awq_scales.get(int(m.group(1))) if m is not None else None
            arr = np.frombuffer(pack_ffn_gf4(t, col_scale, scale_search, args.chunk_rows), dtype=np.uint8)
        elif t.name in awq_norms:
            arr = np.ascontiguousarray(awq_norms[t.name], dtype="<f4").view(np.uint8).reshape(-1)
        else:
            arr = np.asarray(t.data).view(np.uint8).reshape(-1)
        writer.write_tensor_data(arr)
        if idx % 50 == 0 or idx == len(reader.tensors):
            print(f"  wrote {idx}/{len(reader.tensors)} tensors", file=sys.stderr, flush=True)
    writer.close()

    verify_unified(out_path, expected_tensors=len(reader.tensors), expected_kv=kv_count)

    size = os.path.getsize(out_path)
    print(json.dumps({
        "output": str(out_path),
        "name": args.name,
        "tensor_count": len(reader.tensors),
        "gf4_ffn_tensors": n_gf4,
        "awq_norm_folds": len(awq_norms),
        "scale_search": scale_search,
        "data_bytes": size,
        "data_gib": size / 1024**3,
    }, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    return cook_reranker_unified(args)


if __name__ == "__main__":
    raise SystemExit(main())
