"""Tests for the thinking (reasoning) card."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import theme  # noqa: E402

from chat_test_helpers import _ansi, _plain, assert_true  # noqa: E402


def test_thinking_collapsed_hides_body() -> None:
    from widgets.chat.thinking_block import _thinking_card

    plain = _plain(_thinking_card("secret chain of thought", expanded=False, done=False, frame=0))
    assert_true("» Thinking ..." in plain, f"collapsed label shown: {plain}")
    assert_true("secret chain" not in plain, f"body hidden while collapsed: {plain}")


def test_thinking_expanded_shows_body() -> None:
    from widgets.chat.thinking_block import _thinking_card

    plain = _plain(_thinking_card("secret chain of thought", expanded=True, done=False, frame=0))
    assert_true("⌄ Thinking ..." in plain, f"expanded marker shown: {plain}")
    assert_true("secret chain of thought" in plain, f"body visible when expanded: {plain}")


def test_thinking_label_animates_across_frames() -> None:
    from widgets.chat.thinking_block import _thinking_card

    frame0 = _ansi(_thinking_card("", False, False, frame=0), width=40)
    frame1 = _ansi(_thinking_card("", False, False, frame=1), width=40)
    assert_true(frame0 != frame1, "consecutive frames differ (hue drift + bounce)")
    assert_true("\x1b[2;" in frame0 or "\x1b[2m" in frame0, f"bouncing dim char present: {frame0!r}")

    # The dim span's start index ping-pongs 0..n-1..0 across the label.
    def bounce_index(frame: int) -> int:
        card = _thinking_card("", False, False, frame)
        for span in card.spans:
            if "dim" in str(span.style):
                return span.start
        raise AssertionError("no dim span found")

    n = len("» Thinking ...")
    period = 2 * (n - 1)
    assert_true(bounce_index(3) == 3, "bounce advances with the frame")
    assert_true(bounce_index(period - 3) == 3, "bounce reflects back after the last char")


def test_thinking_label_freezes_when_done() -> None:
    from widgets.chat.thinking_block import _thinking_card

    frame0 = _ansi(_thinking_card("", False, True, frame=0), width=40)
    frame9 = _ansi(_thinking_card("", False, True, frame=9), width=40)
    assert_true(frame0 == frame9, "done label ignores the frame (frozen)")
    assert_true("\x1b[2;" not in frame0 and "\x1b[2m" not in frame0, "no dim bounce when done")
    faint = theme.FG_FAINT.lstrip("#")
    triplet = ";".join(str(int(faint[i:i + 2], 16)) for i in (0, 2, 4))
    assert_true(triplet in frame0, f"frozen label uses theme FG_FAINT: {frame0!r}")


def test_thinking_hue_derived_from_theme_accent() -> None:
    import colorsys
    import re as _re

    from widgets.chat.primitives import _hex_hsv, _THINK_HUE_SPAN
    from widgets.chat.thinking_block import _thinking_card

    ansi = _ansi(_thinking_card("", False, False, frame=0), width=40)
    colors = _re.findall(r"38;2;(\d+);(\d+);(\d+)", ansi)
    assert_true(colors, f"truecolor label spans emitted: {ansi!r}")
    accent_hue, accent_sat, accent_val = _hex_hsv(theme.ACCENT)
    for r, g, b in colors:
        hue, sat, val = colorsys.rgb_to_hsv(int(r) / 255, int(g) / 255, int(b) / 255)
        distance = min(abs(hue - accent_hue), 1 - abs(hue - accent_hue))
        assert_true(
            distance <= _THINK_HUE_SPAN / 2 + 0.01,
            f"label hue {hue:.2f} drifts around the accent hue {accent_hue:.2f}",
        )
        # Saturation/value are inherited from the accent, not assumed pastels
        # (0.02 slack for the 8-bit round-trip through #rrggbb).
        assert_true(abs(sat - accent_sat) <= 0.02, f"accent saturation inherited, got {sat:.2f}")
        assert_true(abs(val - accent_val) <= 0.02, f"accent value inherited, got {val:.2f}")


def test_thinking_toggle_flips_state() -> None:
    from widgets.chat import ThinkingBlock

    block = ThinkingBlock()
    block.body = "the why"
    assert_true(not block.expanded, "starts collapsed")
    block.toggle()
    assert_true(block.expanded, "toggle expands")
    plain = _plain(block.render_card())
    assert_true("⌄" in plain and "the why" in plain, f"expanded card shows body: {plain}")
    block.toggle()
    plain = _plain(block.render_card())
    assert_true("»" in plain and "the why" not in plain, f"re-collapsed hides body: {plain}")


def test_reasoning_stream_lifecycle_creates_independent_cards() -> None:
    from widgets.chat import ChatView, ThinkingBlock

    view = ChatView()
    appended: list = []
    view._append = lambda w: (appended.append(w), w)[1]
    view._bump = lambda: None

    view.add_reasoning_chunk("first ")
    view.add_reasoning_chunk("segment")
    first = view._reasoning
    assert_true(isinstance(first, ThinkingBlock), "chunk creates a ThinkingBlock")
    assert_true(first.body == "first segment", f"chunks accumulate: {first.body!r}")
    assert_true(appended == [first], "one card mounted for one segment")

    view.flush_reasoning()
    assert_true(first.done, "flush freezes the card")
    assert_true(view._reasoning is None, "flush detaches the streaming ref")

    view.add_reasoning_chunk("second segment")
    second = view._reasoning
    assert_true(second is not first, "next segment gets a fresh card")
    assert_true(not second.done and second.body == "second segment", "fresh card streams")


def test_thinking_tick_throttles_label_frames() -> None:
    from widgets.chat import ChatView
    from widgets.chat.chat_view import _THINK_FRAME_TICKS

    view = ChatView()
    view._append = lambda w: w
    view._bump = lambda: None
    view.add_reasoning_chunk("hmm")
    block = view._reasoning
    for _ in range(2 * _THINK_FRAME_TICKS):
        view._tick()
    assert_true(block.anim_frame == 2, f"label advances every {_THINK_FRAME_TICKS} ticks: {block.anim_frame}")
    view.flush_reasoning()
    for _ in range(2 * _THINK_FRAME_TICKS):
        view._tick()
    assert_true(block.anim_frame == 2, "animation stops at flush")


def test_thinking_stats_suffix_streams_only() -> None:
    from widgets.chat.thinking_block import _thinking_card

    streaming = _plain(_thinking_card("", False, False, 0, stats="12s · ↓ 34 tokens"))
    assert_true("(12s · ↓ 34 tokens)" in streaming, f"stats suffix while streaming: {streaming}")
    frozen = _plain(_thinking_card("", False, True, 0, stats="12s · ↓ 34 tokens"))
    assert_true("12s" not in frozen, f"stats dropped once done: {frozen}")


def test_thinking_tip_row_streams_only() -> None:
    from widgets.chat.thinking_block import _thinking_card

    tip = "type /clear to start a new conversation"
    streaming = _plain(_thinking_card("", False, False, 0, tip=tip))
    assert_true(f"⮑  Tip: {tip}" in streaming, f"tip row while streaming: {streaming}")
    frozen = _plain(_thinking_card("", False, True, 0, tip=tip))
    assert_true("Tip:" not in frozen, f"tip dropped once done: {frozen}")

    # Suffix and tip wear the ghost gray, and never the dim attribute — the
    # bouncing label character must stay the only dim span.
    ansi = _ansi(_thinking_card("", False, False, 0, stats="1s · ↓ 2 tokens", tip=tip), width=100)
    ghost = theme.FG_GHOST.lstrip("#")
    triplet = ";".join(str(int(ghost[i:i + 2], 16)) for i in (0, 2, 4))
    assert_true(triplet in ansi, f"decorations use theme FG_GHOST: {ansi!r}")
    card = _thinking_card("", False, False, 0, stats="1s · ↓ 2 tokens", tip=tip)
    dim_spans = [span for span in card.spans if "dim" in str(span.style)]
    assert_true(len(dim_spans) == 1, f"only the bounce char is dim: {dim_spans}")


def test_pick_tip_deterministic_with_rng() -> None:
    import random

    from widgets.chat import ThinkingBlock
    from widgets.chat.thinking_block import _pick_tip, _TIPS

    tip = _pick_tip(random.Random(7))
    assert_true(tip == _pick_tip(random.Random(7)), "same seed, same tip")
    assert_true(tip in _TIPS, f"tip comes from the pool: {tip!r}")

    block = ThinkingBlock(rng=random.Random(7))
    assert_true(block.tip == tip, "block picks its tip through the injected rng")
    block.body += "growing body must not re-roll the tip"
    assert_true(block.tip == tip, "tip picked once per segment")


def test_thinking_block_live_stats_from_body_and_clock() -> None:
    import time

    from widgets.chat import ThinkingBlock
    from widgets.status_bar import rough_token_count as rtc

    block = ThinkingBlock()
    block.body = "x" * 400
    block.started_mono = time.monotonic() - 33
    plain = _plain(block.render_card())
    assert_true("33s" in plain, f"elapsed seconds shown: {plain}")
    assert_true(f"↓ {rtc(block.body)} tokens" in plain, f"token estimate from body: {plain}")
    assert_true("⮑  Tip:" in plain, f"tip row shown while streaming: {plain}")

    block.done = True
    frozen = _plain(block.render_card())
    assert_true("s ·" not in frozen and "⮑" not in frozen,
                f"stats and tip gone once frozen: {frozen}")


def test_format_elapsed_rolls_to_minutes() -> None:
    from widgets.chat.thinking_block import _format_elapsed

    assert_true(_format_elapsed(33) == "33s", f"seconds only: {_format_elapsed(33)!r}")
    assert_true(_format_elapsed(60) == "1m 00s", f"minute boundary: {_format_elapsed(60)!r}")
    assert_true(_format_elapsed(72) == "1m 12s", f"minutes+seconds: {_format_elapsed(72)!r}")


def main() -> None:
    test_thinking_collapsed_hides_body()
    test_thinking_expanded_shows_body()
    test_thinking_label_animates_across_frames()
    test_thinking_label_freezes_when_done()
    test_thinking_hue_derived_from_theme_accent()
    test_thinking_toggle_flips_state()
    test_reasoning_stream_lifecycle_creates_independent_cards()
    test_thinking_tick_throttles_label_frames()
    test_thinking_stats_suffix_streams_only()
    test_thinking_tip_row_streams_only()
    test_pick_tip_deterministic_with_rng()
    test_thinking_block_live_stats_from_body_and_clock()
    test_format_elapsed_rolls_to_minutes()
    print("thinking block tests passed")


if __name__ == "__main__":
    main()
