#!/usr/bin/env python3
"""Cook the canonical unified qw35 model.

Bakes a base Qwen3.5 GGUF (Q4_K_M / Q5_K / Q6_K FFN) into a single
self-contained `.gguf` whose FFN gate/up/down tensors are stored as GF4
(eight 3-bit quants plus an fp8 scale per 32-bit word, GGUF type-id 100) and
whose post-attention norms carry an AWQ per-channel scale fold. Every other
tensor is copied verbatim. The result is what the server loads by default —
`Qwowl3.5-9B.gguf` cooked from `Qwen3.5-9B-Q4_K_M.gguf`.

Usage:
  cook_qw35_awq_gf4.py MODEL.gguf OUT.gguf --awq act-stats.bin
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import Q4_K, Q5_K, Q6_K  # noqa: E402


FFN_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)\.weight$")
Q4_K_TYPE = 12
Q5_K_TYPE = 13
Q6_K_TYPE = 14

GF4_GGUF_TYPE = 100  # qw35 GF4 weight type-id inside a unified .gguf
F32_GGUF_TYPE = 0


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


def fp32_to_e5m2_byte(values: np.ndarray) -> np.ndarray:
    half_bits = values.astype(np.float16).view(np.uint16).astype(np.uint32)
    rounded = half_bits + np.uint32(0x80)
    return ((rounded >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)


def e5m2_byte_to_fp32(values: np.ndarray) -> np.ndarray:
    half_bits = (values.astype(np.uint16) << np.uint16(8)).astype(np.uint16)
    return half_bits.view(np.float16).astype(np.float32)


# Clip ratios tried by --scale-search: the group scale is gmax*f rounded to
# e5m2. f<1 sacrifices the (already-saturated) max element for finer resolution
# on the rest of the group; the min-MSE choice is taken per group.
_GF4_SEARCH_FACTORS = np.array(
    [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70], dtype=np.float32
)


def _gf4_quantize_with_scale(
    groups: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize (R,G,8) groups to GF4 codes for a given signed per-group scale.

    Returns (q_raw uint32 in [0,7], reconstruction x_hat). The 8 representable
    levels are scale*{1,.75,.5,.25,0,-.25,-.5,-.75} (q_raw 0..7), matching the
    Metal dequant `out = (4 - q_raw)*0.25*scale`.
    """
    norm = np.zeros_like(groups, dtype=np.float32)
    np.divide(groups, scale[..., None], out=norm, where=scale[..., None] != 0.0)
    q_raw = np.clip(np.rint(norm * -4.0 + 4.0), 0.0, 7.0)
    x_hat = (4.0 - q_raw) * 0.25 * scale[..., None]
    return q_raw.astype(np.uint32), x_hat


def pack_gf4_prepared(
    rows: np.ndarray, scale_search: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if rows.dtype != np.float32:
        rows = rows.astype(np.float32, copy=False)
    if rows.shape[1] % 8 != 0:
        raise ValueError(f"GF4 requires columns divisible by 8, got {rows.shape[1]}")

    groups = rows.reshape(rows.shape[0], rows.shape[1] // 8, 8)
    max_idx = np.abs(groups).argmax(axis=2)
    gmax = np.take_along_axis(groups, max_idx[..., None], axis=2).squeeze(axis=2)

    if scale_search:
        best_err: np.ndarray | None = None
        scale_byte = fp32_to_e5m2_byte(gmax)
        for factor in _GF4_SEARCH_FACTORS:
            cand_byte = fp32_to_e5m2_byte(gmax * factor)
            cand_scale = e5m2_byte_to_fp32(cand_byte)
            _, x_hat = _gf4_quantize_with_scale(groups, cand_scale)
            err = np.sum((groups - x_hat) ** 2, axis=2)
            if best_err is None:
                best_err, scale_byte = err, cand_byte
            else:
                better = err < best_err
                best_err = np.where(better, err, best_err)
                scale_byte = np.where(better, cand_byte, scale_byte).astype(np.uint8)
    else:
        scale_byte = fp32_to_e5m2_byte(gmax)

    scale = e5m2_byte_to_fp32(scale_byte)
    q_raw, _ = _gf4_quantize_with_scale(groups, scale)

    q_prepared = q_raw ^ np.uint32(4)
    words = scale_byte.astype(np.uint32)
    for i in range(8):
        words |= q_prepared[:, :, i] << np.uint32(8 + i * 3)
    return words.astype("<u4", copy=False), scale


def read_act_stats(path: str) -> dict[str, Any]:
    """Read a QW35ACT capture file (per-channel mean-abs FFN activations)."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"QW35ACT\0":
            raise ValueError(f"{path}: not a QW35ACT stats file")
        version, layers, gu_dim, dn_dim = struct.unpack("<IIII", f.read(16))
        (tokens,) = struct.unpack("<Q", f.read(8))
        gateup = np.fromfile(f, dtype="<f4", count=layers * gu_dim).reshape(layers, gu_dim)
        down = np.fromfile(f, dtype="<f4", count=layers * dn_dim).reshape(layers, dn_dim)
    return {
        "version": version,
        "layers": layers,
        "gu_dim": gu_dim,
        "dn_dim": dn_dim,
        "tokens": tokens,
        "gateup": gateup,
        "down": down,
    }


def awq_col_scale(act: np.ndarray, alpha: float) -> np.ndarray:
    """Per-input-channel AWQ scale s_c = act_c^alpha, geomean-normalised to 1 and
    clipped. Salient channels (act above geomean) get s>1: their weight columns
    are multiplied by s for finer GF4 resolution, and the inverse is folded into
    the pre-FFN norm so the layer output is unchanged in full precision."""
    act = np.maximum(act.astype(np.float64), 1e-8)
    log_s = alpha * np.log(act)
    log_s -= log_s.mean()  # geomean(s) = 1
    s = np.exp(log_s)
    return np.clip(s, 0.2, 8.0).astype(np.float32)


def dequantize_chunk(tensor_data: np.ndarray, tensor_type: int, start: int, stop: int) -> np.ndarray:
    rows = tensor_data[start:stop]
    if tensor_type == Q4_K_TYPE:
        block_size = 144
        blocks = rows.reshape(-1, block_size)
        cols = rows.shape[1] // block_size * 256
        return Q4_K.dequantize_blocks(blocks).reshape(stop - start, cols)
    if tensor_type == Q5_K_TYPE:
        block_size = 176
        blocks = rows.reshape(-1, block_size)
        cols = rows.shape[1] // block_size * 256
        return Q5_K.dequantize_blocks(blocks).reshape(stop - start, cols)
    if tensor_type == Q6_K_TYPE:
        block_size = 210
        blocks = rows.reshape(-1, block_size)
        cols = rows.shape[1] // block_size * 256
        return Q6_K.dequantize_blocks(blocks).reshape(stop - start, cols)
    raise ValueError(f"unsupported source tensor type {tensor_type}")


def _pack_ffn_gf4(
    tensor: Any, col_scale: np.ndarray | None, scale_search: bool, chunk_rows: int
) -> bytes:
    """Dequantize an FFN tensor, optionally AWQ-scale its columns, and return the
    GF4 prepared-word bytes (row-major rows x groups uint32), matching the on-disk
    layout the decode/tiled kernels expect."""
    src_type = int(tensor.tensor_type)
    cols = int(tensor.shape[0])
    rows = int(tensor.shape[1])
    if cols % 8 != 0:
        raise ValueError(f"{tensor.name}: GF4 needs columns divisible by 8, got {cols}")
    parts: list[np.ndarray] = []
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        src = dequantize_chunk(tensor.data, src_type, start, stop)
        if col_scale is not None:
            src = src * col_scale[None, :]
        words, _ = pack_gf4_prepared(src, scale_search=scale_search)
        parts.append(words)
    arr = np.concatenate(parts, axis=0)
    expected = rows * (cols // 8) * 4
    if arr.nbytes != expected:
        raise ValueError(f"{tensor.name}: GF4 nbytes {arr.nbytes} != expected {expected}")
    return arr.tobytes(order="C")


def _copy_metadata(writer: Any, reader: GGUFReader) -> int:
    """Copy every KV metadata field from the source GGUF into the writer. Returns
    the number of distinct keys present in the output (incl. general.architecture,
    which the writer's constructor already added)."""
    from gguf import GGUFValueType

    added = {"general.architecture"}  # set by GGUFWriter.__init__
    for key, field in reader.fields.items():
        if key == "general.architecture":
            continue
        types = list(field.types)
        if not types:
            continue
        try:
            if types[0] == GGUFValueType.ARRAY:
                writer.add_key_value(key, field.contents(), GGUFValueType.ARRAY, sub_type=types[1])
            else:
                writer.add_key_value(key, field.contents(), types[0])
            added.add(key)
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: skipped metadata key {key!r}: {exc}", file=sys.stderr)
    return len(added)


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
    kv_count = _copy_metadata(writer, reader)

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
            arr = np.frombuffer(_pack_ffn_gf4(t, col_scale, args.scale_search, args.chunk_rows), dtype=np.uint8)
        elif t.name in awq_norms:
            arr = np.ascontiguousarray(awq_norms[t.name], dtype="<f4").view(np.uint8).reshape(-1)
        else:
            # Stream verbatim straight from the source memmap (no big copy).
            arr = np.asarray(t.data).view(np.uint8).reshape(-1)
        writer.write_tensor_data(arr)
        if idx % 50 == 0 or idx == len(reader.tensors):
            print(f"  wrote {idx}/{len(reader.tensors)} tensors", file=sys.stderr, flush=True)
    writer.close()

    _verify_unified(out_path, expected_tensors=len(reader.tensors), expected_kv=kv_count)

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


def _verify_unified(path: Path, expected_tensors: int, expected_kv: int) -> None:
    """Self-check the written unified .gguf header (gguf-py can't re-read GF4 type-100,
    so this validates the magic/version/counts; `qw35-server --check` is the
    authoritative full parse)."""
    with open(path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
    if magic != 0x46554747:
        raise ValueError(f"{path}: bad GGUF magic {magic:#x}")
    if version != 3:
        raise ValueError(f"{path}: unexpected GGUF version {version}")
    if n_tensors != expected_tensors:
        raise ValueError(f"{path}: tensor count {n_tensors} != {expected_tensors}")
    if n_kv != expected_kv:
        raise ValueError(f"{path}: kv count {n_kv} != {expected_kv}")
    print(f"  self-check OK: {n_tensors} tensors, {n_kv} kv entries", file=sys.stderr)


def main() -> int:
    args = parse_args()
    return cook_qw35_unified(args)


if __name__ == "__main__":
    raise SystemExit(main())
