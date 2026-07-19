"""Tests for the search_engine tool's offline pieces (parser, URL cleanup,
validation).

Run directly: ``python qwowl35/tests/search_engine_test.py``. No network:
the fetch path is exercised only up to argument validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.web.search_engine import (  # noqa: E402
    SEARCH_ENGINE_SCHEMA,
    SearchEngineTool,
)
from tools.web.search_engine.executor import (  # noqa: E402
    InstantAnswer,
    ResultsPageParser,
    SearchResult,
    clean_url,
    format_results,
)


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


_RESULTS_PAGE = """
<html><body>
<div class="serp__results">
  <div class="result results_links">
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">Example <b>Title</b></a>
    <a class="result__snippet">A short <b>snippet</b> of text.</a>
    <span class="result__url"> example.com/page </span>
  </div>
  <div class="result result--ad">
    <a class="result__a" href="https://ads.example.com">Sponsored</a>
  </div>
  <div class="result results_links">
    <a class="result__a" href="https://plain.example.org/doc">Second</a>
  </div>
  <div class="nav-link">
    <form action="/html/" method="post">
      <input type="hidden" name="q" value="test query">
      <input type="hidden" name="s" value="30">
      <input type="submit" class="btn" value="Next">
    </form>
  </div>
</body></html>
"""


def test_schema_wire_shape() -> None:
    assert_equal(SEARCH_ENGINE_SCHEMA["name"], "search_engine", "wire name")
    params = SEARCH_ENGINE_SCHEMA["parameters"]
    assert_equal(params["required"], ["query"], "query required")
    assert_true("max_results" in params["properties"], "max_results advertised")


def test_parser_extracts_results_skips_ads_and_finds_next() -> None:
    parser = ResultsPageParser()
    parser.feed(_RESULTS_PAGE)
    assert_equal(len(parser.results), 2, "two organic results, ad skipped")
    first = parser.results[0]
    assert_equal(first["title"], "Example Title", "nested-tag title joined")
    assert_equal(first["description"], "A short snippet of text.", "snippet")
    assert_equal(first["domain"], "example.com/page", "domain whitespace collapsed")
    assert_true(parser.next_page is not None, "next form found")
    assert_equal(parser.next_page["q"], "test query", "hidden q carried")
    assert_equal(parser.next_page["s"], "30", "hidden offset carried")


def test_clean_url_unwraps_redirects() -> None:
    wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc"
    assert_equal(clean_url(wrapped), "https://example.com/page", "uddg unwrapped")
    assert_equal(
        clean_url("https://plain.example.org/doc"),
        "https://plain.example.org/doc",
        "plain URL untouched",
    )


def test_format_results_renders_card_and_results() -> None:
    card = InstantAnswer(
        heading="Example",
        abstract="An example thing.",
        abstract_source="Wikipedia",
        abstract_url="https://en.wikipedia.org/wiki/Example",
    )
    results = [
        SearchResult(
            title="Example Title",
            description="A snippet.",
            url="https://example.com/page",
            domain="example.com",
        )
    ]
    text = format_results(card, results, "example")
    assert_true("Example" in text, "card heading rendered")
    assert_true("An example thing." in text, "abstract rendered")
    assert_true("1. Example Title" in text, "result numbered")
    assert_true("https://example.com/page" in text, "result URL rendered")
    empty = format_results(InstantAnswer(), [], "nothing")
    assert_true("No results." in empty, "empty search reported")


def test_invalid_arguments_are_rejected_offline() -> None:
    tool = SearchEngineTool()
    assert_true(
        tool.execute({}).startswith("Error: 'query' is required."),
        "missing query",
    )
    assert_true(
        tool.execute({"query": "   "}).startswith("Error: 'query' is required."),
        "blank query",
    )


def main() -> None:
    test_schema_wire_shape()
    test_parser_extracts_results_skips_ads_and_finds_next()
    test_clean_url_unwraps_redirects()
    test_format_results_renders_card_and_results()
    test_invalid_arguments_are_rejected_offline()
    print("search_engine tests passed")


if __name__ == "__main__":
    main()
