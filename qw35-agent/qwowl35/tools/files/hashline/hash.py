"""Line hashing, mirroring hashline's ``hash.rs``."""

from __future__ import annotations

import xxhash

ShortHash = int

_P1 = 0x9E3779B1
_P2 = 0x85EBCA77
_P3 = 0xC2B2AE3D
_P4 = 0x27D4EB2F
_P5 = 0x165667B1
_MASK = 0xFFFFFFFF


def _rotl32(value: int, bits: int) -> int:
    return ((value << bits) | (value >> (32 - bits))) & _MASK


def _round32(acc: int, lane: int) -> int:
    acc = (acc + lane * _P2) & _MASK
    acc = _rotl32(acc, 13)
    return (acc * _P1) & _MASK


def xxh32(data: bytes, seed: int = 0) -> int:
    """Return xxHash32 with seed 0, matching hashline's Rust dependency."""
    length = len(data)
    offset = 0
    if length >= 16:
        v1 = (seed + _P1 + _P2) & _MASK
        v2 = (seed + _P2) & _MASK
        v3 = seed & _MASK
        v4 = (seed - _P1) & _MASK
        limit = length - 16
        while offset <= limit:
            v1 = _round32(v1, int.from_bytes(data[offset : offset + 4], "little"))
            offset += 4
            v2 = _round32(v2, int.from_bytes(data[offset : offset + 4], "little"))
            offset += 4
            v3 = _round32(v3, int.from_bytes(data[offset : offset + 4], "little"))
            offset += 4
            v4 = _round32(v4, int.from_bytes(data[offset : offset + 4], "little"))
            offset += 4
        h32 = (_rotl32(v1, 1) + _rotl32(v2, 7) + _rotl32(v3, 12) + _rotl32(v4, 18)) & _MASK
    else:
        h32 = (seed + _P5) & _MASK

    h32 = (h32 + length) & _MASK
    while offset + 4 <= length:
        lane = int.from_bytes(data[offset : offset + 4], "little")
        h32 = (h32 + lane * _P3) & _MASK
        h32 = (_rotl32(h32, 17) * _P4) & _MASK
        offset += 4
    while offset < length:
        h32 = (h32 + data[offset] * _P5) & _MASK
        h32 = (_rotl32(h32, 11) * _P1) & _MASK
        offset += 1

    h32 ^= h32 >> 15
    h32 = (h32 * _P2) & _MASK
    h32 ^= h32 >> 13
    h32 = (h32 * _P3) & _MASK
    h32 ^= h32 >> 16
    return h32 & _MASK


def full_hash(line: str) -> int:
    return full_hash_bytes(line.rstrip().encode("utf-8"))


def full_hash_bytes(bytes_: bytes) -> int:
    return xxh32(bytes_, 0)


def full_hash64(line: str) -> int:
    return full_hash_bytes64(line.rstrip().encode("utf-8"))


def full_hash_bytes64(bytes_: bytes) -> int:
    return xxhash.xxh3_64_intdigest(bytes_)


def short_hash(line: str) -> str:
    return format_short_hash(short_hash_value(line))


def short_hash_value(line: str) -> ShortHash:
    return short_from_full(full_hash(line))


def short_from_full(full: int) -> ShortHash:
    return full & 0xFF


def format_short_hash(short: ShortHash) -> str:
    return f"{short & 0xFF:02x}"


def format_line_ref(line_no: int, short: ShortHash) -> str:
    """The copyable line locator shown in read output: line number followed by
    the 2-hex content hash with NO separator (e.g. ``12af``).

    Single choke point for the anchor grammar. Dropping the former ``:`` measured
    ~-4.8% read-output tokens against the model's own BPE vocab (see
    ``tools/anchor_encoding_decision.md``) at zero cost to the cross-check: the
    hash is still the low byte, so the fixed trailing 2 chars parse back to it.
    """
    return f"{line_no}{format_short_hash(short)}"


def write_short_hash_bytes(buf: bytearray, short: ShortHash) -> None:
    rendered = format_short_hash(short).encode("ascii")
    buf[0:2] = rendered


def collides(a: str, b: str) -> bool:
    return short_hash_value(a) == short_hash_value(b)
