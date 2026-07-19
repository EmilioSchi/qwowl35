#!/usr/bin/env python3
"""Shared helpers for the qw35 model cookers.

Everything the GF4/AWQ bake needs, extracted VERBATIM from
`cook_qw35_awq_gf4.py` (the 9B cooker) so the reranker cooker
(`cook_qwowl3_reranker_gf4.py`) can reuse it without forking:

  * e5m2 scale byte round-trip (`fp32_to_e5m2_byte` / `e5m2_byte_to_fp32`)
  * the GF4 packer (`pack_gf4_prepared`, `_gf4_quantize_with_scale`) —
    eight 3-bit quants + fp8/e5m2 scale per 32-bit word, GGUF type-id 100
  * QW35ACT activation-stats reading (`read_act_stats`) and the AWQ
    per-channel scale (`awq_col_scale`)
  * source-row dequantization (`dequantize_chunk`; Q4_K/Q5_K/Q6_K for the 9B
    plus a Q8_0 branch for the reranker source)
  * the FFN pack driver (`pack_ffn_gf4`), metadata copy (`copy_metadata`)
    and the written-header self-check (`verify_unified`)

The 9B cooker's behavior is unchanged by the extraction: the functions are
byte-identical, and the new parameters (`Q8_0` support, metadata overrides)
default to the old behavior.
"""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import Q4_K, Q5_K, Q6_K, Q8_0  # noqa: E402


FFN_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)\.weight$")
Q8_0_TYPE = 8
Q4_K_TYPE = 12
Q5_K_TYPE = 13
Q6_K_TYPE = 14

GF4_GGUF_TYPE = 100  # qw35 GF4 weight type-id inside a unified .gguf
F32_GGUF_TYPE = 0


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
    if tensor_type == Q8_0_TYPE:
        # 34-byte blocks of 32 elements (fp16 scale + 32 int8). Used by the
        # reranker source GGUF; the 9B sources are K-quants only.
        block_size = 34
        blocks = rows.reshape(-1, block_size)
        cols = rows.shape[1] // block_size * 32
        return Q8_0.dequantize_blocks(blocks).reshape(stop - start, cols)
    raise ValueError(f"unsupported source tensor type {tensor_type}")


def pack_ffn_gf4(
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


def copy_metadata(
    writer: Any,
    reader: GGUFReader,
    overrides: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> int:
    """Copy every KV metadata field from the source GGUF into the writer.
    `overrides` replaces the copied value for matching keys (e.g. a cooked
    general.name); `extra` appends brand-new keys (e.g. qwowl.cook.* recipe
    provenance). Returns the number of distinct keys present in the output
    (incl. general.architecture, which the writer's constructor already
    added). With both left at None this is exactly the 9B cooker's historical
    metadata copy."""
    from gguf import GGUFValueType

    overrides = overrides or {}
    added = {"general.architecture"}  # set by GGUFWriter.__init__
    for key, field in reader.fields.items():
        if key == "general.architecture":
            continue
        types = list(field.types)
        if not types:
            continue
        try:
            if key in overrides:
                writer.add_key_value(key, overrides[key], types[0])
            elif types[0] == GGUFValueType.ARRAY:
                writer.add_key_value(key, field.contents(), GGUFValueType.ARRAY, sub_type=types[1])
            else:
                writer.add_key_value(key, field.contents(), types[0])
            added.add(key)
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: skipped metadata key {key!r}: {exc}", file=sys.stderr)
    for key, (value, value_type) in (extra or {}).items():
        if key in added:
            continue
        writer.add_key_value(key, value, value_type)
        added.add(key)
    return len(added)


def verify_unified(path: Path, expected_tensors: int, expected_kv: int) -> None:
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
