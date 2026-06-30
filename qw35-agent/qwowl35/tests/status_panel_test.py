"""Tests for the top status panel formatting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.text import Text  # noqa: E402

from widgets.status_panel import (  # noqa: E402
    StatusState,
    compact_count,
    compose_status_bar,
    context_line,
    context_summary,
    decode_summary,
    display_path,
    effort_cap_percent,
    fmt_percent,
    fmt_tps,
    host_label,
    percent,
    rough_token_count,
    think_summary,
)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(value, label: str) -> None:
    if not value:
        raise AssertionError(label)


def test_percent_handles_unknowns_and_bounds() -> None:
    assert_equal(percent(None, 100), None, "unknown used")
    assert_equal(percent(50, None), None, "unknown total")
    assert_equal(percent(1, 0), None, "zero total")
    assert_equal(percent(25, 100), 25.0, "normal percent")
    assert_equal(percent(150, 100), 100.0, "percent capped at 100")


def test_effort_cap_mapping_matches_server() -> None:
    assert_equal(effort_cap_percent("on", "low"), 4, "low effort")
    assert_equal(effort_cap_percent("on", "medium"), 10, "medium effort")
    assert_equal(effort_cap_percent("on", "high"), 16, "high effort")
    assert_equal(effort_cap_percent("on", "xhigh"), 16, "xhigh server backstop")
    assert_equal(effort_cap_percent("off", "high"), None, "thinking off")
    assert_equal(effort_cap_percent("auto", None), None, "auto before reasoning")
    assert_equal(effort_cap_percent("auto", None, inferred_thinking=True), 16, "auto inferred")


def test_display_path_normalizes_home_and_crops_from_left() -> None:
    home_path = str(Path.home() / "projects/qw35/qw35-agent")
    assert_true(display_path(home_path).startswith("~/"), "home path uses tilde")
    path = str(Path.home() / "projects/qw35/qw35-agent/qwowl35/widgets/status_panel.py")
    cropped = display_path(path, max_len=24)
    assert_true(cropped.startswith("..."), "uses ascii crop marker")
    assert_true(cropped.endswith("status_panel.py"), "keeps useful suffix")
    assert_true(len(cropped) <= 24, "respects max length")


def test_compact_context_and_think_text() -> None:
    assert_equal(compact_count(120_000), "120k", "compact thousands")
    assert_equal(fmt_percent(4.1308, 1), "4.1%", "one decimal percent")
    assert_equal(context_summary(4_957, 120_000), "4.1%/120k", "compact context")
    assert_equal(think_summary("on", "low"), "low think", "enabled effort")
    assert_equal(think_summary("on", "low", None), "low think", "idle effort")
    assert_equal(think_summary("off", "low"), "disabled think", "disabled thinking")
    assert_equal(think_summary("auto", None), "auto think", "auto before inference")
    assert_equal(think_summary("auto", "high", 12.4, True), "high think 12%", "live estimate")


def test_decode_speed_uses_fixed_four_figure_field() -> None:
    assert_equal(fmt_tps(None), " 0.0", "unknown speed")
    assert_equal(fmt_tps(0), " 0.0", "zero speed treated as unknown")
    assert_equal(fmt_tps(9.9), " 9.9", "single digit padded to four")
    assert_equal(fmt_tps(19.65), "19.6", "two digit one decimal")
    assert_equal(fmt_tps(127.4), " 127", "three digit drops decimal")
    assert_true(len(fmt_tps(9.9)) == 4, "field width is four")
    assert_equal(decode_summary(19.65), "19.6 tok/s", "speed carries unit")
    assert_equal(decode_summary(None), " 0.0 tok/s", "unknown speed keeps width")


def test_host_label_strips_scheme() -> None:
    assert_equal(host_label("http://localhost:8080"), "localhost:8080", "http stripped")
    assert_equal(host_label("https://host:9"), "host:9", "https stripped")
    assert_equal(host_label("localhost:8080"), "localhost:8080", "no scheme untouched")


def test_context_line_combines_context_and_think() -> None:
    state = StatusState(
        base_url="http://localhost:8080",
        think="auto",
        effort=None,
        ctx_size=130_000,
        prompt_tokens=4_957,
    )
    assert_equal(context_line(state), " 0.0 tok/s  3.8%/130k  auto think", "footer-left text before speed")
    state.decode_tps = 19.65
    assert_equal(context_line(state), "19.6 tok/s  3.8%/130k  auto think", "speed leads the footer text")


def test_status_bar_stacks_when_too_narrow() -> None:
    left = Text("3.8%/130k  auto think")   # 21 cells
    right = Text("● localhost:8080 · qw35")  # 24 cells

    wide = compose_status_bar(left, right, 80)
    assert_true("\n" not in wide.plain, "wide terminal keeps one justified row")
    assert_true(wide.plain.startswith("3.8%/130k"), "left half leads the row")
    assert_true(wide.plain.rstrip().endswith("qw35"), "host half trails the row")

    narrow = compose_status_bar(left, right, 30)
    assert_true("\n" in narrow.plain, "narrow terminal stacks onto two rows")
    top, bottom = narrow.plain.split("\n", 1)
    assert_equal(top, "3.8%/130k  auto think", "context on the first row")
    assert_equal(bottom, "● localhost:8080 · qw35", "host on the second row")

    unknown = compose_status_bar(left, right, 0)
    assert_true("\n" not in unknown.plain, "unknown width falls back to one row")


def test_rough_token_count_is_monotonic() -> None:
    assert_equal(rough_token_count(""), 0, "empty text")
    assert_true(rough_token_count("hello world") > 0, "non-empty text")
    assert_true(
        rough_token_count("hello world" * 4) > rough_token_count("hello world"),
        "larger text estimates more tokens",
    )


def main() -> None:
    test_percent_handles_unknowns_and_bounds()
    test_effort_cap_mapping_matches_server()
    test_display_path_normalizes_home_and_crops_from_left()
    test_compact_context_and_think_text()
    test_decode_speed_uses_fixed_four_figure_field()
    test_host_label_strips_scheme()
    test_context_line_combines_context_and_think()
    test_status_bar_stacks_when_too_narrow()
    test_rough_token_count_is_monotonic()
    print("status panel tests passed")


if __name__ == "__main__":
    main()
