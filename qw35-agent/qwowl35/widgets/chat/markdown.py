"""Theme-aware markdown rendering for chat messages and cards."""

from __future__ import annotations

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Markdown
from rich.styled import Styled
from rich.theme import Theme as RichTheme

import theme


def _code_theme() -> str:
    """Pygments/Rich syntax theme matching the active light/dark mode.

    ``monokai`` reads well on dark backgrounds, but its bright foregrounds wash
    out on a light theme, so use a light-oriented style there instead.
    """
    return "monokai" if theme.is_dark() else "default"


def _markdown_element_styles() -> RichTheme:
    """Rich style overrides tying markdown elements to the active palette.

    Rebuilt per render (cheap: a handful of style strings) so a live theme
    switch restyles headings/links/code on the next repaint, like every other
    ``theme.*`` read. Fenced code blocks keep the Pygments ``code_theme``.
    """
    heading = f"bold {theme.MD_HEADING}"
    return RichTheme(
        {
            **{f"markdown.h{i}": heading for i in range(1, 7)},
            "markdown.h1.border": str(theme.MD_HEADING),
            "markdown.link": f"underline {theme.MD_LINK}",
            "markdown.link_url": f"underline {theme.MD_LINK}",
            "markdown.code": f"{theme.MD_CODE} on {theme.BG_SURFACE}",
            "markdown.block_quote": f"italic {theme.MD_QUOTE}",
            "markdown.item.bullet": str(theme.MD_LIST),
            "markdown.item.number": str(theme.MD_LIST),
            "markdown.hr": str(theme.MD_HR),
        }
    )


class _ThemedMarkdown:
    """Markdown whose base prose color is the theme foreground and whose
    element styles (headings, links, code, quotes, bullets) come from the
    active palette — all resolved at render time so a live theme switch
    restyles on the next repaint. Parsed once at construction.
    """

    def __init__(self, text: str) -> None:
        self._md = Markdown(text, code_theme=_code_theme())

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        # The pushed theme stays active while the yielded renderable is
        # rendered (the consumer resumes this generator only afterwards).
        with console.use_theme(_markdown_element_styles()):
            yield Styled(self._md, theme.FG_BRIGHT)


def _markdown(text: str) -> _ThemedMarkdown:
    """Markdown renderable using the mode-appropriate code-block theme."""
    return _ThemedMarkdown(text)
