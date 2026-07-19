"""Tests for the card frame chrome."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import theme  # noqa: E402

from chat_test_helpers import _ansi, _plain, _textual_content_height, assert_true  # noqa: E402


def test_card_frame_height_counts_every_row() -> None:
    # Same height contract as _FullWidthLines: Textual counts newlines, so
    # every card row — borders included — must terminate with one, or the
    # bottom border is cropped away.
    from widgets.chat.card import _CardFrame

    height = _textual_content_height(_CardFrame(Text("a\nb"), title="T", timestamp="12:00:00"))
    assert_true(height == 4, f"2 borders + 2 body rows (got {height})")

    height = _textual_content_height(_CardFrame(Text(""), title="T"))
    assert_true(height == 3, f"empty card keeps one blank body row (got {height})")

    height = _textual_content_height(_CardFrame(Text("word " * 20), title="T"), width=20)
    assert_true(height >= 6, f"wrapped inner rows all measured (got {height})")


def test_card_frame_chrome_layout() -> None:
    from widgets.chat.card import _CardFrame

    plain = _plain(
        _CardFrame(Text("hello"), title="User", chip="Explorer", timestamp="14:32:05")
    )
    lines = [line for line in plain.splitlines() if line.strip()]
    assert_true(lines[0].startswith("╭─ User "), f"title on the top-left edge: {lines[0]}")
    assert_true(lines[0].rstrip().endswith("╮"), f"top border closes: {lines[0]}")
    assert_true(lines[1].startswith("│ hello"), f"inner content framed: {lines[1]}")
    # The chip rides one cell after the ╰ corner — no ╞/╡ notch glyphs, which
    # render broken in some terminal fonts.
    assert_true(lines[-1].startswith("╰─ Explorer "), f"chip on the bottom-left edge: {lines[-1]}")
    assert_true("╞" not in plain and "╡" not in plain, f"no notch glyphs: {lines[-1]}")
    assert_true(
        lines[-1].rstrip().endswith("14:32:05 ─╯"),
        f"timestamp on the bottom-right edge: {lines[-1]}",
    )
    # Narrow terminals degrade (drop timestamp, then chip, truncate title)
    # without crashing.
    for width in (20, 10):
        _ansi(
            _CardFrame(
                Text("hello"), title="Incoming Message", chip="Explorer", timestamp="14:32:05"
            ),
            width=width,
        )


def test_card_frame_edge_takes_scope_color() -> None:
    # Each card scope colors its border via a palette token, resolved from
    # the LIVE theme at render time (so theme switches restyle old cards).
    from widgets.chat.card import _CardFrame

    accent = theme.ACCENT.lstrip("#")
    rgb = ";".join(str(int(accent[i:i + 2], 16)) for i in (0, 2, 4))
    ansi = _ansi(_CardFrame(Text("x"), title="Spawn", edge="ACCENT"), width=30)
    assert_true(f"\x1b[38;2;{rgb}m" in ansi, f"accent edge emitted: {ansi!r}")

    ghost = theme.FG_GHOST.lstrip("#")
    rgb = ";".join(str(int(ghost[i:i + 2], 16)) for i in (0, 2, 4))
    ansi = _ansi(_CardFrame(Text("x"), title="Plain"), width=30)
    assert_true(f"\x1b[38;2;{rgb}m" in ansi, f"default ghost edge emitted: {ansi!r}")


def main() -> None:
    test_card_frame_height_counts_every_row()
    test_card_frame_chrome_layout()
    test_card_frame_edge_takes_scope_color()
    print("chat card tests passed")


if __name__ == "__main__":
    main()
