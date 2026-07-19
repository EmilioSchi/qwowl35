"""Tests for the chat widgets' low-level renderable primitives."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from widgets.chat.primitives import _BlockquoteFrame, _FullWidthLines, _line_with_bg  # noqa: E402

from chat_test_helpers import _ansi, _plain, _textual_content_height, assert_true  # noqa: E402


def test_block_rows_pad_background_to_width() -> None:
    ansi = _ansi(_FullWidthLines([_line_with_bg(Text("x"), "#111318")]), width=10)

    assert_true("\x1b[48;2;17;19;24m         " in ansi, "background padding emitted")


def test_fullwidth_wrap_height_counts_every_row() -> None:
    # Regression (output-not-displayed bug): every wrapped row must terminate with a
    # newline, including the last, so Textual's newline-based height measurement
    # matches the real row count and never crops the final row.
    rows = [_line_with_bg(Text(f"row {i}"), "#111318") for i in range(4)]
    height = _textual_content_height(_FullWidthLines(rows, wrap=True))
    assert_true(height == 4, f"every wrapped row measured (got {height}, want 4)")


def test_fullwidth_wrap_shows_all_lines_under_height_constraint() -> None:
    # Regression: the approval modal lives in an auto-height container, so the
    # render options carry a `height`. _FullWidthLines(wrap=True) must render each
    # logical line at its natural height — not pad every line to the container
    # height, which previously collapsed all but the first line into blanks.
    rows = [_line_with_bg(Text(f"line {i}"), "#111318") for i in range(5)]
    console = Console(width=30, file=StringIO())
    options = console.options.update(height=12)  # container imposes a height
    rendered = console.render_lines(_FullWidthLines(rows, wrap=True), options)
    joined = "\n".join("".join(seg.text for seg in row) for row in rendered)
    for i in range(5):
        assert_true(f"line {i}" in joined, f"line {i} visible under height constraint")


def test_blockquote_frame_height_counts_every_row() -> None:
    # Same height contract as _FullWidthLines: Textual counts newlines, so
    # every row — header and body — must terminate with one.
    height = _textual_content_height(
        _BlockquoteFrame(Text("a\nb"), title="User", timestamp="12:00:00")
    )
    assert_true(height == 3, f"header + 2 body rows (got {height})")

    height = _textual_content_height(_BlockquoteFrame(Text(""), title="User"))
    assert_true(height == 2, f"empty keeps one blank body row (got {height})")

    height = _textual_content_height(
        _BlockquoteFrame(Text("word " * 20), title="User"), width=20
    )
    assert_true(height >= 4, f"wrapped inner rows all measured (got {height})")


def test_blockquote_frame_chrome_layout() -> None:
    plain = _plain(
        _BlockquoteFrame(Text("hello"), title="User", timestamp="14:32:05")
    )
    lines = [line for line in plain.splitlines() if line.strip()]
    assert_true(
        lines[0].startswith("▌ User"),
        f"edge bar and title on header line: {lines[0]}",
    )
    assert_true(
        "14:32:05" in lines[0],
        f"timestamp on header line: {lines[0]}",
    )
    assert_true(
        lines[1].startswith("▌ hello"),
        f"body row with edge bar: {lines[1]}",
    )
    # Narrow terminals degrade (drop timestamp, truncate title) without crashing.
    for width in (20, 10):
        _ansi(
            _BlockquoteFrame(Text("hello"), title="Incoming Message", timestamp="14:32:05"),
            width=width,
        )


def test_blockquote_frame_edge_takes_scope_color() -> None:
    import theme

    accent = theme.ACCENT.lstrip("#")
    rgb = ";".join(str(int(accent[i:i + 2], 16)) for i in (0, 2, 4))
    ansi = _ansi(
        _BlockquoteFrame(Text("x"), title="User", edge="ACCENT"), width=30
    )
    assert_true(
        f"\x1b[38;2;{rgb}m" in ansi,
        f"accent edge emitted: {ansi!r}",
    )

    muted = theme.FG_MUTED.lstrip("#")
    rgb = ";".join(str(int(muted[i:i + 2], 16)) for i in (0, 2, 4))
    ansi = _ansi(_BlockquoteFrame(Text("x"), title="User"), width=30)
    assert_true(
        f"\x1b[38;2;{rgb}m" in ansi,
        f"default muted edge emitted: {ansi!r}",
    )


def test_blockquote_frame_narrow_drops_timestamp() -> None:
    plain = _plain(
        _BlockquoteFrame(Text("hello"), title="User", timestamp="14:32:05")
    )
    # At default width=100 the timestamp fits.
    assert_true("14:32:05" in plain, "timestamp present at default width")

    # At width=20 the timestamp should be dropped.
    narrow = _plain(
        _BlockquoteFrame(Text("hello"), title="User", timestamp="14:32:05")
    )
    # Render at narrow width via _ansi to check it doesn't crash.
    _ansi(
        _BlockquoteFrame(Text("hello"), title="User", timestamp="14:32:05"),
        width=20,
    )


def test_blockquote_frame_timestamp_fits_when_room() -> None:
    plain = _plain(
        _BlockquoteFrame(Text("hello"), title="User", timestamp="14:32:05")
    )
    lines = plain.splitlines()
    assert_true(
        "14:32:05" in lines[0],
        f"timestamp on header line at default width: {lines[0]}",
    )


def main() -> None:
    test_block_rows_pad_background_to_width()
    test_fullwidth_wrap_height_counts_every_row()
    test_fullwidth_wrap_shows_all_lines_under_height_constraint()
    test_blockquote_frame_height_counts_every_row()
    test_blockquote_frame_chrome_layout()
    test_blockquote_frame_edge_takes_scope_color()
    test_blockquote_frame_narrow_drops_timestamp()
    test_blockquote_frame_timestamp_fits_when_room()
    print("chat primitives tests passed")


if __name__ == "__main__":
    main()
