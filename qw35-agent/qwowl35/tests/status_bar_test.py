"""Tests for the top status panel formatting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.text import Text  # noqa: E402

from widgets.status_bar import (  # noqa: E402
    StatusState,
    active_ctx,
    compact_count,
    compose_status_bar,
    context_line,
    context_summary,
    context_text,
    decode_summary,
    display_path,
    effort_cap_percent,
    fmt_percent,
    fmt_tps,
    host_label,
    mode_box,
    mode_label,
    percent,
    rough_token_count,
    think_summary,
    thinking_cap_tokens,
)

import theme  # noqa: E402
from modes import MODE_COLOR_TOKENS, Mode  # noqa: E402


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


def test_thinking_cap_tokens_matches_server_basis() -> None:
    # Mirrors qw35-server `thinking_budget_for`: fixed max_tokens is the basis,
    # otherwise the server's 8192 agentic default; the cap never drops below 16.
    assert_equal(thinking_cap_tokens(None, None), None, "no cap when thinking off")
    assert_equal(thinking_cap_tokens(10, None), 819, "8192 basis when max_tokens unset")
    assert_equal(thinking_cap_tokens(16, None), 1310, "backstop on the 8192 basis")
    assert_equal(thinking_cap_tokens(16, 4096), 655, "fixed max_tokens is the basis")
    assert_equal(thinking_cap_tokens(4, 100), 16, "cap floored at 16 tokens")


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

    # Live usage scales against the server's 8192 basis, not remaining context:
    # medium cap = 819 tokens, so 410 estimated reasoning tokens ≈ 50%.
    state.think = "on"
    state.effort = "medium"
    state.reasoning_estimate = 410
    assert_equal(context_line(state), "19.6 tok/s  3.8%/130k  medium think 50%", "usage on server basis")


def test_active_session_ctx_takes_precedence() -> None:
    # The percentage is measured against the LIVE ceiling of the session that
    # is actually streaming (server-reported, grows on demand); the main size
    # from /health//props is only the fallback.
    state = StatusState(
        base_url="http://localhost:8080",
        think="off",
        effort=None,
        ctx_size=131_072,
        prompt_tokens=8_192,
    )
    assert_equal(active_ctx(state), 131_072, "main ctx is the fallback")
    assert_true("6.2%/131.1k" in context_line(state), f"fallback percent: {context_line(state)}")
    state.active_ctx_size = 16_384
    assert_equal(active_ctx(state), 16_384, "server-reported session ctx wins")
    assert_true("50.0%/16.4k" in context_line(state), f"scratch percent: {context_line(state)}")
    # A grown session moves the ceiling and the percentage follows.
    state.active_ctx_size = 32_768
    assert_true("25.0%/32.8k" in context_line(state), f"grown percent: {context_line(state)}")


def test_context_text_colors_only_speed_and_percentage() -> None:
    # Only the tok/s figure and the context percentage wear the active
    # agent's mode color (INSERT = editor = WARNING); the ctx size and think
    # summary stay plain, so the two live numbers pop against the rest.
    state = StatusState(
        base_url="http://localhost:8080",
        think="off",
        effort=None,
        ctx_size=131_072,
        active_ctx_size=16_384,
        prompt_tokens=8_192,
        decode_tps=19.65,
        mode=Mode.INSERT,
    )
    styled = context_text(state)
    assert_equal(styled.plain, context_line(state), "styled text mirrors the plain line")
    fill = getattr(theme, MODE_COLOR_TOKENS[Mode.INSERT])
    colored = [styled.plain[s.start:s.end] for s in styled.spans if str(s.style) == fill]
    assert_equal(colored, ["19.6 tok/s", "50.0%"], "speed and percentage wear the agent color")
    plain_dim = [styled.plain[s.start:s.end] for s in styled.spans if str(s.style) == theme.FG_DIM]
    assert_equal(plain_dim, ["/16.4k", "disabled think"], "size and think stay plain")
    # The color follows the active agent: the planner's PLAN fill differs
    # from the editor's INSERT fill.
    state.mode = Mode.PLAN
    plan_fill = getattr(theme, MODE_COLOR_TOKENS[Mode.PLAN])
    assert_true(plan_fill != fill, "PLAN and INSERT wear distinct fills")
    replanned = context_text(state)
    recolored = [
        replanned.plain[s.start:s.end] for s in replanned.spans if str(s.style) == plan_fill
    ]
    assert_equal(recolored, ["19.6 tok/s", "50.0%"], "color tracks the active agent")


def test_mode_box_is_permanent_inverted_and_uppercase() -> None:
    state = StatusState(base_url="http://localhost:8080", think="auto", effort=None)

    # A fresh state shows NORMAL (every conversation starts there) — the box
    # is permanent, never blank.
    assert_equal(state.mode, Mode.NORMAL, "fresh state defaults to NORMAL")
    assert_equal(mode_label(state), "NORMAL", "default label is NORMAL")
    normal = mode_box(state)
    assert_equal(normal.plain, " NORMAL ", "box pads the label so it reads as a box")

    # A pushed display mode takes over, uppercased like a vim mode.
    state.mode = Mode.VISUAL
    assert_equal(mode_label(state), "VISUAL", "live mode is uppercased")
    assert_equal(mode_box(state).plain, " VISUAL ", "box tracks the live mode")

    # Inverted highlight lives on the label span (so an appended gap stays
    # neutral): base-colored text on the mode's fill, bold.
    assert_equal(len(normal.spans), 1, "the fill is a single span, not the whole Text")
    style = str(normal.spans[0].style)
    assert_true(theme.ACCENT in style, "NORMAL fills with the accent")
    assert_true(theme.BG_BASE in style, "base tone is the box text")
    assert_true("bold" in style, "box text is bold")


def test_mode_box_colors_are_theme_derived_and_mirror_agent_roles() -> None:
    state = StatusState(base_url="http://localhost:8080", think="auto", effort=None)
    seen_fills = {}
    for mode in Mode:
        state.mode = mode
        box = mode_box(state)
        assert_equal(box.plain, f" {mode.value.upper()} ", f"{mode.value} label")
        fill = getattr(theme, MODE_COLOR_TOKENS[mode])
        style = str(box.spans[0].style)
        assert_true(fill in style, f"{mode.value} box fills with theme.{MODE_COLOR_TOKENS[mode]}")
        seen_fills[mode] = fill
    # VISUAL/INSERT mirror the sub-agent card colors (Explorer=ACCENT,
    # Editor=WARNING) so every visual reference to a role converges. That
    # makes VISUAL share NORMAL's accent deliberately — the label text
    # disambiguates. The user-selectable modes stay distinct among themselves.
    assert_equal(MODE_COLOR_TOKENS[Mode.VISUAL], "ACCENT", "VISUAL wears the Explorer color")
    assert_equal(MODE_COLOR_TOKENS[Mode.INSERT], "WARNING", "INSERT wears the Editor color")
    user_fills = [seen_fills[m] for m in (Mode.NORMAL, Mode.PLAN, Mode.WEB, Mode.CHAT)]
    assert_equal(
        len(set(user_fills)), len(user_fills), "user-selectable modes have distinct fills"
    )


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
    test_thinking_cap_tokens_matches_server_basis()
    test_display_path_normalizes_home_and_crops_from_left()
    test_compact_context_and_think_text()
    test_decode_speed_uses_fixed_four_figure_field()
    test_host_label_strips_scheme()
    test_context_line_combines_context_and_think()
    test_active_session_ctx_takes_precedence()
    test_context_text_colors_only_speed_and_percentage()
    test_mode_box_is_permanent_inverted_and_uppercase()
    test_mode_box_colors_are_theme_derived_and_mirror_agent_roles()
    test_status_bar_stacks_when_too_narrow()
    test_rough_token_count_is_monotonic()
    print("status bar tests passed")


if __name__ == "__main__":
    main()
