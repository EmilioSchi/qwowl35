"""Tests for theme-aware markdown rendering."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import theme  # noqa: E402

from chat_test_helpers import _ansi, assert_true  # noqa: E402


def test_markdown_elements_take_palette_colors() -> None:
    # Markdown headings/links/inline code color from the LIVE palette (not
    # Rich's built-in defaults), so a theme switch restyles old messages on
    # the next repaint — same idiom as the card edges above.
    from dataclasses import replace

    from widgets.chat.markdown import _markdown

    def triplet(hex_color: str) -> str:
        h = hex_color.lstrip("#")
        return ";".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))

    md = _markdown("## Head\n\n[link](https://x.y)\n\nuse `code` here")
    loud = replace(
        theme.DEFAULT, MD_HEADING="#aa1122", MD_LINK="#22aa33", MD_CODE="#3344aa"
    )
    try:
        theme.set_active(loud)
        ansi = _ansi(md, width=40)
        for token in ("MD_HEADING", "MD_LINK", "MD_CODE"):
            rgb = triplet(getattr(loud, token))
            assert_true(f"38;2;{rgb}" in ansi, f"{token} color emitted: {ansi!r}")
        # Live switch: the SAME renderable restyles under the next palette.
        theme.set_active(replace(loud, MD_HEADING="#7755ff"))
        ansi = _ansi(md, width=40)
        assert_true(f"38;2;{triplet('#7755ff')}" in ansi, "heading follows live switch")
        assert_true(f"38;2;{triplet('#aa1122')}" not in ansi, "old heading color gone")
    finally:
        theme.set_active(theme.DEFAULT)


def main() -> None:
    test_markdown_elements_take_palette_colors()
    print("chat markdown tests passed")


if __name__ == "__main__":
    main()
