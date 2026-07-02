#!/usr/bin/env python3
"""Register qw35's custom GGUF tensor types with the installed gguf package.

The unified qw35 .gguf stores FFN tensors with custom GGUF type-ids that stock
gguf-py rejects in `GGUFReader._build_tensors` ("100 is not a valid
GGMLQuantizationType"):

  100  GF4 — eight 3-bit codes + fp8(e5m2) scale per uint32 word; 8 elems / 4 B
  101  GF2 — interleaved 256-elem super-block: 16 uint32 code words (2-bit
       codes) followed by 16 fp8(e5m2) scale bytes; 256 elems / 80 B (= 16/5)

`gguf.constants.GGMLQuantizationType` and `gguf.constants.GGML_QUANT_SIZES`
are process-wide singletons that gguf_reader/quants alias by reference, so
injecting the members once — before any GGUFReader is constructed — is enough
for reading, size math (n_bytes, BPW), and `tensor_type.name` rendering.
Custom-typed tensors surface as raw uint8 blobs (the reader's quantized-blob
fallback), which is correct: gguf-py has no dequant for them.

Usage (from any tools/ script, before creating a GGUFReader):

    import gguf_custom_types
    gguf_custom_types.register_custom_types()
"""
from __future__ import annotations

QW35_CUSTOM_TYPES = {
    # name: (type_id, block_elems, block_bytes)
    "GF4": (100, 8, 4),
    "GF2": (101, 16, 5),
}


def register_custom_types() -> None:
    """Idempotently inject GF4/GF2 into gguf-py's type enum and size table."""
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

    for name, (type_id, block_elems, block_bytes) in QW35_CUSTOM_TYPES.items():
        if type_id in GGMLQuantizationType._value2member_map_:
            continue
        # Stdlib IntEnum has no public extension API (and aenum is not a
        # dependency), so build the member the way EnumMeta would.
        member = int.__new__(GGMLQuantizationType, type_id)
        member._name_ = name
        member._value_ = type_id
        GGMLQuantizationType._value2member_map_[type_id] = member
        # EnumMeta forbids setattr on the class; _member_map_ is what
        # EnumMeta.__getattr__ consults, so GGMLQuantizationType.GF4 resolves.
        GGMLQuantizationType._member_map_[name] = member
        GGMLQuantizationType._member_names_.append(name)
        GGML_QUANT_SIZES[member] = (block_elems, block_bytes)
