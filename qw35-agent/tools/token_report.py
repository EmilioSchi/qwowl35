#!/usr/bin/env python3
"""Ground-truth BPE token-cost report for the file-tool anchor grammar.

Ports the qw35 tokenizer (byte-level BPE + the `qwen35` pre-tokenizer) from
`qw35-server/src/tokenizer.rs` into Python, driven by the model's own
vocab+merges read from the GGUF. It is validated against the vocab-research
`camelcase_single_after_function.tsv` oracle before any number is trusted.

Then it ranks candidate encodings for:
  1. tool names after `<function=`
  2. concept/param names after `<parameter=`
  3. per-line read prefixes measured IN CONTEXT over a real source corpus
  4. the copied-anchor cost the model emits back in a tool call

Usage:  python3 tools/token_report.py [--gguf PATH]
Reads only; writes nothing. Uses the base GGUF (standard quant) by default —
the cooked Qwowl GGUF carries custom tensor types the stock `gguf` lib rejects,
but its tokenizer is identical.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_GGUF = Path("/Users/emilioschinina/Documents/qw35/.gguf/Qwen3.5-9B-Q4_K_M.gguf")
VOCAB_RESEARCH = Path("/Users/emilioschinina/Documents/qw35/vocab-research")


# --------------------------------------------------------------------------- #
# GPT-2 byte<->unicode map (mirrors tokenizer.rs::byte_maps)
# --------------------------------------------------------------------------- #
def build_byte_maps() -> tuple[list[str], dict[str, int]]:
    byte_to_unicode = [""] * 256
    unicode_to_byte: dict[str, int] = {}
    used = [False] * 256

    def put(byte: int, codepoint: int) -> None:
        ch = chr(codepoint)
        used[byte] = True
        byte_to_unicode[byte] = ch
        unicode_to_byte[ch] = byte

    for b in range(0x21, 0x7E + 1):
        put(b, b)
    for b in range(0xA1, 0xAC + 1):
        put(b, b)
    for b in range(0xAE, 0xFF + 1):
        put(b, b)
    nxt = 0
    for b in range(256):
        if used[b]:
            continue
        put(b, 256 + nxt)
        nxt += 1
    return byte_to_unicode, unicode_to_byte


# --------------------------------------------------------------------------- #
# qwen35 pre-tokenizer (mirrors tokenizer.rs::qwen35_pretokenize)
# --------------------------------------------------------------------------- #
def _is_combining_mark(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x0300 <= cp <= 0x036F
        or 0x1AB0 <= cp <= 0x1AFF
        or 0x1DC0 <= cp <= 0x1DFF
        or 0x20D0 <= cp <= 0x20FF
        or 0xFE20 <= cp <= 0xFE2F
    )


def _is_letter_or_mark(ch: str) -> bool:
    return ch.isalpha() or _is_combining_mark(ch)


def _is_number(ch: str) -> bool:
    return ch.isnumeric()


def _is_whitespace(ch: str) -> bool:
    return ch.isspace()


def _contraction_end(chars: list[str], pos: int) -> int | None:
    n = len(chars)
    if pos + 1 >= n:
        return None
    c1 = chars[pos + 1].lower()
    if c1 in ("s", "t", "m", "d"):
        return pos + 2
    if pos + 2 >= n:
        return None
    c2 = chars[pos + 2].lower()
    if (c1, c2) in (("r", "e"), ("v", "e"), ("l", "l")):
        return pos + 3
    return None


def qwen35_pretokenize(text: str) -> list[str]:
    chars = list(text)
    n = len(chars)
    pieces: list[str] = []
    pos = 0
    while pos < n:
        start = pos
        c = chars[pos]

        if c == "'":
            end = _contraction_end(chars, pos)
            if end is not None:
                pieces.append("".join(chars[start:end]))
                pos = end
                continue

        nextc = chars[pos + 1] if pos + 1 < n else None
        if (
            c != "\r"
            and c != "\n"
            and not _is_number(c)
            and (_is_letter_or_mark(c) or (nextc is not None and _is_letter_or_mark(nextc)))
        ):
            pos += 1
            while pos < n and _is_letter_or_mark(chars[pos]):
                pos += 1
            pieces.append("".join(chars[start:pos]))
            continue

        if _is_number(c):
            pos += 1
            pieces.append("".join(chars[start:pos]))
            continue

        ch2 = (chars[pos + 1] if pos + 1 < n else None) if c == " " else c
        if ch2 is not None and not _is_whitespace(ch2) and not _is_letter_or_mark(ch2) and not _is_number(ch2):
            if c == " ":
                pos += 1
            while (
                pos < n
                and not _is_whitespace(chars[pos])
                and not _is_letter_or_mark(chars[pos])
                and not _is_number(chars[pos])
            ):
                pos += 1
            while pos < n and (chars[pos] == "\r" or chars[pos] == "\n"):
                pos += 1
            pieces.append("".join(chars[start:pos]))
            continue

        if _is_whitespace(c):
            scan = pos
            last_rn = None
            while scan < n and _is_whitespace(chars[scan]):
                ch = chars[scan]
                scan += 1
                if ch == "\r" or ch == "\n":
                    last_rn = scan
            if last_rn is not None:
                pieces.append("".join(chars[start:last_rn]))
                pos = last_rn
                continue
            if scan - pos > 1 and scan < n:
                end = scan - 1
                pieces.append("".join(chars[start:end]))
                pos = end
                continue
            pieces.append("".join(chars[start:scan]))
            pos = scan
            continue

        pos += 1
        pieces.append("".join(chars[start:pos]))
    return pieces


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
class Qw35Tokenizer:
    def __init__(self, tokens: list[str], merges: list[str]) -> None:
        self.tokens = tokens
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.byte_to_unicode, self.unicode_to_byte = build_byte_maps()
        self.merge_ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(merges):
            # split on the first space at char-index >= 1 (mirrors tokenizer.rs)
            idx = merge.find(" ", 1)
            if idx == -1:
                continue
            self.merge_ranks[(merge[:idx], merge[idx + 1 :])] = rank

    @classmethod
    def load(cls, gguf_path: Path) -> "Qw35Tokenizer":
        from gguf import GGUFReader

        reader = GGUFReader(str(gguf_path), "r")

        def field(name: str):
            f = reader.get_field(name)
            if f is None:
                raise SystemExit(f"missing GGUF metadata: {name}")
            return f

        model = field("tokenizer.ggml.model").contents()
        pre = field("tokenizer.ggml.pre").contents()
        if model != "gpt2" or pre != "qwen35":
            raise SystemExit(f"unexpected tokenizer model={model!r} pre={pre!r}")
        tokens = list(field("tokenizer.ggml.tokens").contents())
        merges = list(field("tokenizer.ggml.merges").contents())
        return cls(tokens, merges)

    def _bpe(self, encoded: str) -> list[str]:
        symbols = list(encoded)
        if len(symbols) <= 1:
            return symbols
        ranks = self.merge_ranks
        while True:
            best_idx = -1
            best_rank = None
            for i in range(len(symbols) - 1):
                r = ranks.get((symbols[i], symbols[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_idx = i
            if best_idx == -1:
                break
            symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1]
            del symbols[best_idx + 1]
        return symbols

    def _byte_encode(self, piece: str) -> str:
        b2u = self.byte_to_unicode
        return "".join(b2u[b] for b in piece.encode("utf-8"))

    def encode_tokens(self, text: str) -> list[str]:
        """Return the list of vocab-token surface forms (byte-encoded)."""
        out: list[str] = []
        for piece in qwen35_pretokenize(text):
            out.extend(self._bpe(self._byte_encode(piece)))
        return out

    def count(self, text: str) -> int:
        return len(self.encode_tokens(text))

    def decode_token(self, token: str) -> str:
        u2b = self.unicode_to_byte
        return bytes(u2b[ch] for ch in token if ch in u2b).decode("utf-8", "replace")

    def readable(self, text: str) -> str:
        """`a | b | c` view of the token boundaries (for validation/inspection)."""
        return " | ".join(self.decode_token(t) for t in self.encode_tokens(text))

    def name_is_single_token(self, prefix: str, name: str, suffix: str = ">") -> bool:
        """True iff `name` is covered by exactly one token in prefix+name+suffix."""
        toks = self.encode_tokens(prefix + name + suffix)
        decoded = [self.decode_token(t) for t in toks]
        return name in decoded


# --------------------------------------------------------------------------- #
# Validation against the vocab-research oracle
# --------------------------------------------------------------------------- #
def validate(tok: Qw35Tokenizer) -> None:
    print("== VALIDATION (against vocab-research oracle) ==")
    tsv = VOCAB_RESEARCH / "camelcase_single_after_function.tsv"
    checked = passed = 0
    if tsv.exists():
        lines = tsv.read_text(encoding="utf-8").splitlines()[1:]
        # sample every ~10th row to keep it quick but broad
        for row in lines[::10]:
            parts = row.split("\t")
            if len(parts) < 4:
                continue
            name, expected = parts[0], parts[3].strip()
            got = tok.readable(f"<function={name}>")
            checked += 1
            passed += got == expected
            if got != expected and checked - passed <= 5:
                print(f"  MISMATCH {name!r}: expected {expected!r} got {got!r}")
        print(f"  TSV boundary match: {passed}/{checked} sampled `<function=NAME>` rows")
    else:
        print("  (TSV oracle not found; skipping boundary check)")

    # Hard facts from the README
    facts = {
        "beginTransaction": True,   # single token after <function=
        "getName": True,
        "getContext": False,        # splits into =get · Context
        "getRow": False,
    }
    for name, want in facts.items():
        got = tok.name_is_single_token("<function=", name)
        flag = "OK" if got == want else "FAIL"
        print(f"  [{flag}] <function={name}> single-token={got} (expected {want})")
    print()


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _table(rows: list[tuple], headers: tuple[str, ...]) -> str:
    cols = list(zip(*([headers] + rows))) if rows else [(h,) for h in headers]
    widths = [max(len(str(x)) for x in col) for col in cols]
    fmt = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths))
    out = [fmt(headers), fmt(tuple("-" * w for w in widths))]
    out += [fmt(r) for r in rows]
    return "\n".join(out)


def report_tool_names(tok: Qw35Tokenizer) -> None:
    print("== 1. TOOL NAMES after `<function=` ==")
    names = ["read", "beginTransaction", "edit", "insert", "delete", "view", "open", "patch"]
    rows = []
    for name in sorted(names, key=lambda x: tok.count(f"<function={x}>")):
        total = tok.count(f"<function={name}>")
        single = tok.name_is_single_token("<function=", name)
        name_only = tok.count(name)
        rows.append((name, total, "yes" if single else "NO", name_only))
    print(_table(rows, ("name", "<function=NAME> toks", "single?", "name-alone toks")))
    print()


def report_concept_names(tok: Qw35Tokenizer) -> None:
    print("== 2. CONCEPT / PARAM NAMES after `<parameter=` ==")
    names = ["anchor", "hash", "ref", "at", "loc", "line", "tag", "mark", "id", "pos", "key", "row"]
    rows = []
    for name in sorted(names, key=lambda x: tok.count(f"<parameter={x}>")):
        total = tok.count(f"<parameter={name}>")
        single = tok.name_is_single_token("<parameter=", name)
        rows.append((name, total, "yes" if single else "NO"))
    print(_table(rows, ("name", "<parameter=NAME> toks", "single?")))
    print()


def _load_corpus() -> list[str]:
    files = [
        REPO / "qwowl35/tools/files/hashline/tool_calling.py",
        REPO / "qwowl35/tools/files/hashline/anchor.py",
        REPO / "qwowl35/tools/files/hashline/output.py",
    ]
    lines: list[str] = []
    for f in files:
        if f.exists():
            lines.extend(f.read_text(encoding="utf-8").splitlines())
    return lines


def _build_word_table(tok: Qw35Tokenizer) -> list[str] | None:
    """256 distinct lowercase words that stay a single token after a digit."""
    words = []
    seen = set()
    for t in tok.tokens:
        w = tok.decode_token(t)
        if w.isascii() and w.isalpha() and w.islower() and 2 <= len(w) <= 6 and w not in seen:
            # must be single-token in render position: right after a line-number digit
            if tok.name_is_single_token("1", w, "|"):
                words.append(w)
                seen.add(w)
                if len(words) == 256:
                    break
    return words if len(words) == 256 else None


def report_line_prefix(tok: Qw35Tokenizer) -> None:
    from qwowl35.tools.files.hashline.hash import short_hash_value, format_short_hash

    print("== 3. PER-LINE PREFIX SCHEMES (in-context, real corpus) ==")
    corpus = _load_corpus()
    if not corpus:
        print("  (corpus not found)")
        return
    print(f"  corpus: {len(corpus)} lines")

    words = _build_word_table(tok)
    word_ok = words is not None
    if not word_ok:
        print("  (word-table: could not assemble 256 single-token words; skipping word schemes)")

    def hh(line: str) -> str:
        return format_short_hash(short_hash_value(line))

    def byte(line: str) -> int:
        return short_hash_value(line) & 0xFF

    schemes = {
        "N:hh|C   (baseline)": lambda i, c: f"{i}:{hh(c)}|{c}",
        "Nhh|C    (drop ':')": lambda i, c: f"{i}{hh(c)}|{c}",
        "N hh C   (spaces)":   lambda i, c: f"{i} {hh(c)} {c}",
        "N:hh C   (space join)": lambda i, c: f"{i}:{hh(c)} {c}",
        "Nhh C    (drop ':' +sp)": lambda i, c: f"{i}{hh(c)} {c}",
        "NxHH|C   (x, UPPER)": lambda i, c: f"{i}x{hh(c).upper()}|{c}",
        "N:HH|C   (UPPER hex)": lambda i, c: f"{i}:{hh(c).upper()}|{c}",
    }
    if word_ok:
        schemes["N:word|C (word)"] = lambda i, c: f"{i}:{words[byte(c)]}|{c}"
        schemes["Nword|C  (word,no:)"] = lambda i, c: f"{i}{words[byte(c)]}|{c}"
        schemes["N word C (word sp)"] = lambda i, c: f"{i} {words[byte(c)]} {c}"

    results = []
    for label, render in schemes.items():
        blob = "\n".join(render(i + 1, c) for i, c in enumerate(corpus))
        results.append((label, tok.count(blob)))
    base = dict(results)["N:hh|C   (baseline)"]
    rows = []
    for label, total in sorted(results, key=lambda x: x[1]):
        delta = total - base
        rows.append((label, total, f"{delta:+d}", f"{100*delta/base:+.1f}%"))
    print(_table(rows, ("scheme", "total toks", "delta", "delta%")))
    print()


def report_copied_anchor(tok: Qw35Tokenizer) -> None:
    from qwowl35.tools.files.hashline.hash import short_hash_value, format_short_hash

    print("== 4. COPIED-ANCHOR COST (what the model emits back) ==")
    samples = ["def greet(name):", "    return msg", "        name = name.upper()"]
    rows = []
    for c in samples:
        h = format_short_hash(short_hash_value(c))
        base = f"<parameter=anchor>12:{h}</parameter>"
        new = f"<parameter=ref>12{h}</parameter>"
        rows.append((repr(c[:18]), tok.count(base), tok.count(new), tok.count(base) - tok.count(new)))
    # a range too
    rng_base = "<parameter=anchor>12:af..18:9c</parameter>"
    rng_new = "<parameter=ref>12af..189c</parameter>"
    rows.append(("(range 12:af..18:9c)", tok.count(rng_base), tok.count(rng_new), tok.count(rng_base) - tok.count(rng_new)))
    print(_table(rows, ("line/anchor", "baseline toks", "new toks", "saved")))
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    args = ap.parse_args()
    if not args.gguf.exists():
        raise SystemExit(f"GGUF not found: {args.gguf}")
    print(f"Loading tokenizer from {args.gguf.name} ...")
    tok = Qw35Tokenizer.load(args.gguf)
    print(f"  vocab={len(tok.tokens)} merges={len(tok.merge_ranks)}\n")
    validate(tok)
    report_tool_names(tok)
    report_concept_names(tok)
    report_line_prefix(tok)
    report_copied_anchor(tok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
