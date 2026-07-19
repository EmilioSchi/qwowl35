"""Tests for the web_fetch tool's offline pieces (HTML→text, validation).

Run directly: ``python qwowl35/tests/web_fetch_test.py``. No network: the
fetch path is exercised only up to argument validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.web.web_fetch import WEB_FETCH_SCHEMA, WebFetchTool  # noqa: E402
from tools.web.web_fetch.executor import html_to_text  # noqa: E402


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def test_schema_matches_qwen_wire_shape() -> None:
    assert_equal(WEB_FETCH_SCHEMA["name"], "web_fetch", "wire name")
    params = WEB_FETCH_SCHEMA["parameters"]
    assert_equal(sorted(params["required"]), ["prompt", "url"], "url+prompt required")
    assert_equal(
        params["properties"]["format"]["enum"],
        ["auto", "markdown", "html", "text"],
        "format enum",
    )


def test_html_to_text_strips_scripts_and_keeps_blocks() -> None:
    html = (
        "<html><head><title>t</title><style>p{}</style></head>"
        "<body><h1>Header</h1><script>var x=1;</script>"
        "<p>First paragraph.</p><p>Second &amp; last.</p></body></html>"
    )
    text = html_to_text(html)
    assert_true("Header" in text, text)
    assert_true("First paragraph." in text, text)
    assert_true("Second & last." in text, "entities decoded")
    assert_true("var x=1" not in text, "script dropped")
    assert_true("p{}" not in text, "style dropped")
    assert_true("\n" in text, "block structure keeps line breaks")


def test_invalid_arguments_are_rejected_offline() -> None:
    tool = WebFetchTool()
    assert_true(
        tool.execute({"prompt": "x"}).startswith("Error: 'url' is required."),
        "missing url",
    )
    assert_true(
        tool.execute({"url": "ftp://example.com", "prompt": "x"}).startswith(
            "Error: The URL must be a fully-formed valid URL"
        ),
        "non-http scheme rejected",
    )


def main() -> None:
    test_schema_matches_qwen_wire_shape()
    test_html_to_text_strips_scripts_and_keeps_blocks()
    test_invalid_arguments_are_rejected_offline()
    print("web_fetch tests passed")


if __name__ == "__main__":
    main()
