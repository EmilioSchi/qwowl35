#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import io
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import Q4_K, Q5_K, Q6_K  # noqa: E402


MAGIC = b"QW35GF4\0"
VERSION = 1
FFN_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)\.weight$")
# Full decode tensor set: everything the single-token decode path streams.
FULL_RE = re.compile(
    r"^blk\.(\d+)\.(ffn_(?:gate|up|down)|attn_qkv|attn_gate|attn_output|attn_q|attn_k|attn_v|ssm_out)\.weight$"
)
Q4_K_TYPE = 12
Q5_K_TYPE = 13
Q6_K_TYPE = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cook fixed Qw35 FFN tensors into a calm-style prepared GF4 sidecar."
    )
    parser.add_argument("model", help="Source Qw35 GGUF model")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .bin path. Default: <model>.ffn-gf4.bin",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Output metadata JSON path. Default: <output>.json",
    )
    parser.add_argument(
        "--only",
        choices=["all", "gate-up", "down", "full", "full-no-head"],
        default="all",
        help="Tensor subset to cook; `full` covers every decode-path matvec weight plus output.weight,"
        " `full-no-head` the same without output.weight (the head feeds logits directly, so its"
        " quantization noise flips sampled tokens with no compounding speedup)",
    )
    parser.add_argument(
        "--limit-tensors",
        type=int,
        default=0,
        help="Cook only the first N matching tensors for smoke tests",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=16,
        help="Rows to dequantize and quantize at a time",
    )
    parser.add_argument(
        "--metrics-sample-rows",
        type=int,
        default=2,
        help="Rows per tensor to reconstruct for quantization-error metrics",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected tensors and size estimate without writing data",
    )
    return parser.parse_args()


def align_file(f: BinaryIO, alignment: int = 64) -> int:
    pos = f.tell()
    pad = (-pos) % alignment
    if pad:
        f.write(b"\0" * pad)
    return f.tell()


def fp32_to_e5m2_byte(values: np.ndarray) -> np.ndarray:
    half_bits = values.astype(np.float16).view(np.uint16).astype(np.uint32)
    rounded = half_bits + np.uint32(0x80)
    return ((rounded >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)


def e5m2_byte_to_fp32(values: np.ndarray) -> np.ndarray:
    half_bits = (values.astype(np.uint16) << np.uint16(8)).astype(np.uint16)
    return half_bits.view(np.float16).astype(np.float32)


def pack_gf4_prepared(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rows.dtype != np.float32:
        rows = rows.astype(np.float32, copy=False)
    if rows.shape[1] % 8 != 0:
        raise ValueError(f"GF4 requires columns divisible by 8, got {rows.shape[1]}")

    groups = rows.reshape(rows.shape[0], rows.shape[1] // 8, 8)
    max_idx = np.abs(groups).argmax(axis=2)
    gmax = np.take_along_axis(groups, max_idx[..., None], axis=2).squeeze(axis=2)
    scale_byte = fp32_to_e5m2_byte(gmax)
    scale = e5m2_byte_to_fp32(scale_byte)

    norm = np.zeros_like(groups, dtype=np.float32)
    np.divide(groups, scale[..., None], out=norm, where=scale[..., None] != 0.0)
    q_raw = np.rint(norm * -4.0 + 4.0)
    q_raw = np.clip(q_raw, 0.0, 7.0).astype(np.uint32)

    q_prepared = q_raw ^ np.uint32(4)
    words = scale_byte.astype(np.uint32)
    for i in range(8):
        words |= q_prepared[:, :, i] << np.uint32(8 + i * 3)
    return words.astype("<u4", copy=False), scale


def unpack_gf4_prepared(words: np.ndarray) -> np.ndarray:
    words = words.astype(np.uint32, copy=False)
    scale = e5m2_byte_to_fp32((words & np.uint32(0xFF)).astype(np.uint8))
    out = np.empty(words.shape + (8,), dtype=np.float32)
    mul = scale * -0.25
    for i in range(8):
        q_prepared = (words >> np.uint32(8 + i * 3)) & np.uint32(7)
        q_raw = q_prepared ^ np.uint32(4)
        out[:, :, i] = (q_raw.astype(np.int32) - 4).astype(np.float32) * mul
    return out.reshape(words.shape[0], words.shape[1] * 8)


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


def selected_tensors(reader: GGUFReader, only: str) -> list[Any]:
    supported = (Q4_K_TYPE, Q5_K_TYPE, Q6_K_TYPE)
    result = []
    for tensor in reader.tensors:
        if only in ("full", "full-no-head"):
            if tensor.name == "output.weight":
                if only == "full" and int(tensor.tensor_type) in supported:
                    result.append(tensor)
                continue
            match = FULL_RE.match(tensor.name)
            if match and int(tensor.tensor_type) in supported:
                result.append(tensor)
            continue
        match = FFN_RE.match(tensor.name)
        if not match:
            continue
        kind = match.group(2)
        if only == "gate-up" and kind == "down":
            continue
        if only == "down" and kind != "down":
            continue
        if int(tensor.tensor_type) not in (Q4_K_TYPE, Q6_K_TYPE):
            continue
        result.append(tensor)
    result.sort(key=lambda t: t.name)
    return result


def estimate_bytes(tensors: list[Any]) -> int:
    total = 0
    for tensor in tensors:
        cols = int(tensor.shape[0])
        rows = int(tensor.shape[1])
        total += rows * (cols // 8) * 4
    return total


def tensor_metrics(src_rows: np.ndarray, gf4_words: np.ndarray) -> dict[str, float]:
    recon = unpack_gf4_prepared(gf4_words)
    diff = recon - src_rows
    return {
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "src_abs_mean": float(np.mean(np.abs(src_rows))),
    }


def cook_tensor(out, tensor: Any, chunk_rows: int, sample_rows: int) -> dict[str, Any]:
    source_type = int(tensor.tensor_type)
    cols = int(tensor.shape[0])
    rows = int(tensor.shape[1])
    if cols % 8 != 0:
        raise ValueError(f"{tensor.name}: columns must be divisible by 8, got {cols}")
    if tensor.data.shape[0] != rows:
        raise ValueError(f"{tensor.name}: unexpected tensor data shape {tensor.data.shape}")

    data_offset = align_file(out)
    written = 0
    metrics_acc: list[dict[str, float]] = []

    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        src = dequantize_chunk(tensor.data, source_type, start, stop)
        words, _ = pack_gf4_prepared(src)
        out.write(words.tobytes(order="C"))
        written += words.nbytes
        if sample_rows > 0 and len(metrics_acc) < sample_rows:
            take = min(sample_rows - len(metrics_acc), stop - start)
            metrics_acc.append(tensor_metrics(src[:take], words[:take]))

    metrics: dict[str, float] = {}
    if metrics_acc:
        keys = metrics_acc[0].keys()
        metrics = {key: float(sum(item[key] for item in metrics_acc) / len(metrics_acc)) for key in keys}

    return {
        "name": tensor.name,
        "source_type": source_type,
        "shape": [cols, rows],
        "rows": rows,
        "cols": cols,
        "gf4_groups_per_row": cols // 8,
        "data_offset": data_offset,
        "data_nbytes": written,
        "prepared_codes": True,
        "metrics": metrics,
    }


def write_table(out: BinaryIO, records: list[dict[str, Any]]) -> int:
    table_offset = align_file(out)
    out.write(struct.pack("<I", len(records)))
    for record in records:
        name = record["name"].encode("utf-8")
        if len(name) > 0xFFFF:
            raise ValueError(f"tensor name too long: {record['name']}")
        out.write(struct.pack("<H", len(name)))
        out.write(name)
        out.write(
            struct.pack(
                "<HIIQQQI?",
                int(record["source_type"]),
                int(record["rows"]),
                int(record["cols"]),
                int(record["data_offset"]),
                int(record["data_nbytes"]),
                int(record["gf4_groups_per_row"]),
                0,
                bool(record["prepared_codes"]),
            )
        )
    return table_offset


def main() -> int:
    args = parse_args()
    model = Path(args.model)
    if args.output:
        output = Path(args.output)
    elif args.only == "full":
        output = model.parent / (model.stem + ".gf4.bin")
    else:
        output = model.parent / (model.stem + ".ffn-gf4.bin")
    meta_path = Path(args.json) if args.json else output.with_suffix(output.suffix + ".json")

    reader = GGUFReader(str(model))
    tensors = selected_tensors(reader, args.only)
    if args.limit_tensors:
        tensors = tensors[: args.limit_tensors]
    if not tensors:
        raise SystemExit("no matching FFN tensors found")

    estimated = estimate_bytes(tensors)
    if args.dry_run:
        print(json.dumps({
            "source_model": str(model),
            "tensor_count": len(tensors),
            "estimated_data_bytes": estimated,
            "estimated_data_gib": estimated / 1024**3,
            "tensors": [
                {
                    "name": t.name,
                    "source_type": int(t.tensor_type),
                    "shape": [int(t.shape[0]), int(t.shape[1])],
                    "source_nbytes": int(t.n_bytes),
                    "gf4_nbytes": int(t.shape[0]) * int(t.shape[1]) // 2,
                }
                for t in tensors
            ],
        }, indent=2))
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output.open("wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<IIQ", VERSION, len(tensors), 0))
        for index, tensor in enumerate(tensors, 1):
            print(f"[{index}/{len(tensors)}] {tensor.name}", file=sys.stderr, flush=True)
            records.append(cook_tensor(out, tensor, args.chunk_rows, args.metrics_sample_rows))
        table_offset = write_table(out, records)
        out.seek(len(MAGIC) + 8)
        out.write(struct.pack("<Q", table_offset))

    meta = {
        "magic": MAGIC.decode("ascii", errors="replace").rstrip("\0"),
        "version": VERSION,
        "source_model": str(model),
        "output": str(output),
        "tensor_count": len(records),
        "data_bytes": os.path.getsize(output),
        "tensors": records,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "metadata": str(meta_path),
        "tensor_count": len(records),
        "data_bytes": meta["data_bytes"],
        "data_gib": meta["data_bytes"] / 1024**3,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
