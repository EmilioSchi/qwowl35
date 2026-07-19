"""Tests for the JSONL transcript writer.

Run directly: ``python qwowl35/tests/sessions_transcript_test.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sessions.transcript import TranscriptWriter  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def test_records_round_trip_as_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "transcript.jsonl"
        writer = TranscriptWriter(path)
        writer.record("request", messages=[{"role": "user", "content": "hi"}],
                      tools=[], params={"qw35_session": "main"})
        writer.record("raw_chunk", data='{"choices":[{"delta":{"content":"h"}}]}')
        writer.record("assistant", content="hello", reasoning="",
                      tool_calls=[{"id": "c1", "name": "bash",
                                   "arguments": {"command": "ls"}}])
        writer.record("tool_result", id="c1", name="bash", result="ok",
                      is_error=False, executed=True)
        writer.close()

        records = [json.loads(line) for line in path.read_text().splitlines()]
        kinds = [rec["kind"] for rec in records]
        assert_equal(kinds, ["request", "raw_chunk", "assistant", "tool_result"],
                     "kinds in write order")
        assert_equal(records[0]["params"], {"qw35_session": "main"},
                     "request params round-trip")
        assert_equal(records[2]["tool_calls"][0]["arguments"], {"command": "ls"},
                     "nested tool-call arguments round-trip")
        assert_true(all("t" in rec for rec in records), "every record stamped")


def test_timestamps_are_monotonic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "transcript.jsonl"
        writer = TranscriptWriter(path)
        for i in range(5):
            writer.record("raw_chunk", data=str(i))
        writer.close()

        stamps = [json.loads(line)["t"] for line in path.read_text().splitlines()]
        assert_true(all(b >= a for a, b in zip(stamps, stamps[1:])),
                    "t never decreases")


def test_unwritable_path_never_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        writer = TranscriptWriter(Path(tmp))
        writer.record("meta", goal="should be dropped silently")
        writer.close()


def test_unserializable_field_degrades_to_string() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "transcript.jsonl"
        writer = TranscriptWriter(path)
        writer.record("meta", payload=object())
        writer.close()

        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert_equal(len(records), 1, "record still written")
        assert_true(isinstance(records[0]["payload"], str),
                    "unserializable value stringified")


def main() -> None:
    test_records_round_trip_as_jsonl()
    test_timestamps_are_monotonic()
    test_unwritable_path_never_raises()
    test_unserializable_field_degrades_to_string()
    print("sessions transcript tests passed")


if __name__ == "__main__":
    main()
