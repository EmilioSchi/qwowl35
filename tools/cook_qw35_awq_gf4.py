#!/usr/bin/env python3
"""Cook the canonical unified qw35 model.

Bakes a base Qwen3.5 GGUF (Q4_K_M / Q5_K / Q6_K FFN) into a single
self-contained `.gguf` whose FFN gate/up/down tensors are stored as GF4
(eight 3-bit quants plus an fp8 scale per 32-bit word, GGUF type-id 100) and
whose post-attention norms carry an AWQ per-channel scale fold. Every other
tensor is copied verbatim. The result is what the server loads by default —
`Qwowl3.5-9B.gguf` cooked from `Qwen3.5-9B-Q4_K_M.gguf`.

The packing/AWQ/metadata machinery lives in `qw35_cook_common.py` (shared with
the reranker cooker `cook_qwowl3_reranker_gf4.py`); this script keeps only the
9B recipe. Behavior and output are byte-identical to the pre-extraction
cooker (gate: re-cook sha256 matches the shipped Qwowl3.5-9B.gguf).

Usage:
  cook_qw35_awq_gf4.py MODEL.gguf OUT.gguf --awq act-stats.bin
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
    awq_col_scale,
    copy_metadata,
    pack_ffn_gf4,
    read_act_stats,
    verify_unified,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cook the unified qw35 model: base GGUF -> FFN baked as GF4 + AWQ norm fold."
    )
    parser.add_argument("model", help="Source Qw35 base GGUF (Q4_K_M / Q5_K / Q6_K)")
    parser.add_argument("output", help="Output unified .gguf path")
    parser.add_argument(
        "--awq",
        default=None,
        metavar="ACT_STATS",
        help="Activation-aware FFN gate/up quantization. Path to a QW35ACT stats"
        " file (cook with QW35_CAPTURE_ACT_OUT). Scales salient input-channel"
        " weights up and folds the inverse into the post_attention_norm so prefill"
        " and decode share the folded norm. Same 4 bpw.",
    )
    parser.add_argument(
        "--awq-alpha",
        type=float,
        default=0.6,
        help="AWQ scale exponent s_c = act_c^alpha (geomean-normalised). Default 0.6.",
    )
    parser.add_argument(
        "--scale-search",
        action="store_true",
        help="Per group, search a few e5m2 clip ratios and keep the min-MSE scale"
        " instead of plain max-abs. Same 4 bpw / on-disk layout; cooker-only cost.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=16,
        help="Rows to dequantize and quantize at a time",
    )
    return parser.parse_args()


def cook_qw35_unified(args: argparse.Namespace) -> int:
    from gguf import GGUFWriter

    model = Path(args.model)
    out_path = Path(args.output)
    reader = GGUFReader(str(model))

    # AWQ pre-pass: per layer, derive the per-input-channel scale s (applied to
    # gate/up columns) and bake base_norm/s into post_attention_norm so prefill
    # AND decode share the folded norm (no decode-only override needed).
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
            norm_t = norm_by_name.get(f"blk.{layer}.post_attention_norm.weight")
            if norm_t is None:
                continue
            base_norm = np.asarray(norm_t.data, dtype=np.float32).reshape(-1)
            if base_norm.shape[0] != stats["gu_dim"]:
                continue
            s = awq_col_scale(stats["gateup"][layer], args.awq_alpha)
            awq_scales[layer] = s
            awq_norms[f"blk.{layer}.post_attention_norm.weight"] = (base_norm / s).astype(np.float32)
        print(f"AWQ: alpha={args.awq_alpha} layers={len(awq_scales)} folded into norms", file=sys.stderr)

    arch = str(reader.fields["general.architecture"].contents())
    align_field = reader.fields.get("general.alignment")
    alignment = int(align_field.contents()) if align_field is not None else 32

    gate_up_re = re.compile(r"^blk\.(\d+)\.ffn_(?:gate|up)\.weight$")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(path=str(out_path), arch=arch)
    writer.data_alignment = alignment
    kv_count = copy_metadata(writer, reader)

    def classify(t: Any) -> tuple[int, int]:
        """Return (out_type_id, out_nbytes) for a source tensor — cheap, no
        dequant/pack (FFN size is a formula), so pass 1 stays light on RAM."""
        src_type = int(t.tensor_type)
        if FFN_RE.match(t.name) and src_type in (Q4_K_TYPE, Q5_K_TYPE, Q6_K_TYPE):
            cols, rows = int(t.shape[0]), int(t.shape[1])
            return GF4_GGUF_TYPE, rows * (cols // 8) * 4
        if t.name in awq_norms:
            return F32_GGUF_TYPE, int(awq_norms[t.name].size) * 4
        return src_type, int(np.asarray(t.data).nbytes)

    # Pass 1 — declare every tensor (writer_shape is reversed(file dims); the
    # writer re-reverses on write, reproducing the source dim order exactly).
    # A non-uint8 dummy dtype skips GGUFWriter's byte-shape massaging; the int
    # type_id is stored verbatim and packed as the GGUF tensor type.
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

    # Pass 2 — produce and stream each tensor's bytes in order, one at a time
    # (FFN packed exactly once here), so peak RAM is a single tensor.
    for idx, t in enumerate(reader.tensors, 1):
        src_type = int(t.tensor_type)
        if FFN_RE.match(t.name) and src_type in (Q4_K_TYPE, Q5_K_TYPE, Q6_K_TYPE):
            m = gate_up_re.match(t.name)
            col_scale = awq_scales.get(int(m.group(1))) if m is not None else None
            arr = np.frombuffer(pack_ffn_gf4(t, col_scale, args.scale_search, args.chunk_rows), dtype=np.uint8)
        elif t.name in awq_norms:
            arr = np.ascontiguousarray(awq_norms[t.name], dtype="<f4").view(np.uint8).reshape(-1)
        else:
            # Stream verbatim straight from the source memmap (no big copy).
            arr = np.asarray(t.data).view(np.uint8).reshape(-1)
        writer.write_tensor_data(arr)
        if idx % 50 == 0 or idx == len(reader.tensors):
            print(f"  wrote {idx}/{len(reader.tensors)} tensors", file=sys.stderr, flush=True)
    writer.close()

    verify_unified(out_path, expected_tensors=len(reader.tensors), expected_kv=kv_count)

    size = os.path.getsize(out_path)
    print(json.dumps({
        "output": str(out_path),
        "tensor_count": len(reader.tensors),
        "gf4_ffn_tensors": n_gf4,
        "awq_norm_folds": len(awq_norms),
        "data_bytes": size,
        "data_gib": size / 1024**3,
    }, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    return cook_qw35_unified(args)


if __name__ == "__main__":
    raise SystemExit(main())
