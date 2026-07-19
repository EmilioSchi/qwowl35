"""The scrollable chat transcript, styled after little-coder.

Each message is its own widget in a vertical scroller. Assistant and user text is
rendered as **Markdown** (headings, lists, inline code, syntax-highlighted code
blocks). Tool calls/results carry plain **text labels** (no emoji/unicode glyphs,
which many terminals won't render). Streaming assistant text is repainted on a
fast timer so generation visibly evolves without re-parsing on every token.
Reasoning collapses under an animated "» Thinking ..." label (theme-hued drift,
running only while it streams); clicking a card expands its text.
"""

from widgets.chat.card import CardFrame
from widgets.chat.chat_view import ChatView
from widgets.chat.primitives import BlockquoteFrame
from widgets.chat.terminal_chrome import set_terminal_host
from widgets.chat.thinking_block import ThinkingBlock
from widgets.chat.tool_block import ToolBlock

__all__ = ["BlockquoteFrame", "CardFrame", "ChatView", "ThinkingBlock", "ToolBlock", "set_terminal_host"]
