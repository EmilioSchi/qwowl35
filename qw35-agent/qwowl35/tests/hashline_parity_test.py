"""File/function/type parity audit against a local hashline reference clone."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys


REFERENCE = Path(os.environ.get("HASHLINE_REF", "/private/tmp/hashline")) / "crates/core/src"
LOCAL = Path(__file__).resolve().parent.parent / "tools" / "files" / "hashline"

FILE_MAP = {
    "anchor.rs": "anchor.py",
    "cli.rs": "cli.py",
    "context.rs": "context.py",
    "document.rs": "document.py",
    "error.rs": "error.py",
    "fast.rs": "fast.py",
    "hash.rs": "hash.py",
    "hash_cache.rs": "hash_cache.py",
    "lib.rs": "lib.py",
    "main.rs": "main.py",
    "merge.rs": "merge.py",
    "mutation.rs": "mutation.py",
    "orchestration.rs": "orchestration.py",
    "output.rs": "output.py",
    "receipt.rs": "receipt.py",
    "risk.rs": "risk.py",
    "session_cache.rs": "session_cache.py",
    "sha256_window.rs": "sha256_window.py",
    "commands/annotate.rs": "commands/annotate.py",
    "commands/batch.rs": "commands/batch.py",
    "commands/common.rs": "commands/common.py",
    "commands/delete.rs": "commands/delete.py",
    "commands/diff_apply.rs": "commands/diff_apply.py",
    "commands/doctor.rs": "commands/doctor.py",
    "commands/edit.rs": "commands/edit.py",
    "commands/find_block.rs": "commands/find_block.py",
    "commands/grep.rs": "commands/grep.py",
    "commands/index.rs": "commands/index.py",
    "commands/indent.rs": "commands/indent.py",
    "commands/insert.rs": "commands/insert.py",
    "commands/mod.rs": "commands/__init__.py",
    "commands/move.rs": "commands/move.py",
    "commands/patch.rs": "commands/patch.py",
    "commands/read.rs": "commands/read.py",
    "commands/replace.rs": "commands/replace.py",
    "commands/serve.rs": "commands/serve.py",
    "commands/stats.rs": "commands/stats.py",
    "commands/swap.rs": "commands/swap.py",
    "commands/verify.rs": "commands/verify.py",
}

SKIPPED_TRANSPORT = {"mcp.rs"}

RUST_FN = re.compile(r"^(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.M)
RUST_TYPE = re.compile(r"^(?:pub\s+)?(?:struct|enum|type)\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.M)
PY_FN = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.M)
PY_TYPE = re.compile(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\b|^([A-Za-z_][A-Za-z0-9_]*)\s*=", re.M)


def _public_python_types(text: str) -> set[str]:
    names: set[str] = set()
    for class_name, assigned_name in PY_TYPE.findall(text):
        name = class_name or assigned_name
        if not name.startswith("_"):
            names.add(name)
    return names


def test_hashline_file_function_type_parity() -> None:
    if not REFERENCE.exists():
        print(f"skipped: hashline reference not found at {REFERENCE}")
        return

    missing_files: list[str] = []
    gaps: list[str] = []
    rust_files = sorted(REFERENCE.glob("*.rs")) + sorted((REFERENCE / "commands").glob("*.rs"))
    for rust_file in rust_files:
        rel = rust_file.relative_to(REFERENCE).as_posix()
        if rel in SKIPPED_TRANSPORT:
            continue
        py_rel = FILE_MAP.get(rel)
        if py_rel is None:
            missing_files.append(rel)
            continue
        py_file = LOCAL / py_rel
        if not py_file.exists():
            missing_files.append(f"{rel} -> {py_rel}")
            continue

        rust_text = rust_file.read_text(encoding="utf-8")
        py_text = py_file.read_text(encoding="utf-8")
        missing_fns = sorted(set(RUST_FN.findall(rust_text)) - set(PY_FN.findall(py_text)))
        missing_types = sorted(set(RUST_TYPE.findall(rust_text)) - _public_python_types(py_text))
        if missing_fns or missing_types:
            gaps.append(f"{rel}: functions={missing_fns} types={missing_types}")

    if missing_files or gaps:
        raise AssertionError(f"missing_files={missing_files}; gaps={gaps}")


def main() -> None:
    test_hashline_file_function_type_parity()
    print("hashline parity audit passed")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
