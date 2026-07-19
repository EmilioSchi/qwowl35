"""The owl mascot as a Textual widget.

The mascot sits at the top of the screen in normal flow (the chat scrolls inside
its own region below, so the owl never moves). It spans the full width and never
wraps: the accessory text (zzz, progress bars, a long error message…) and the
working-directory line below the brand are cropped to the terminal width rather
than pushed onto extra rows, so the four-line owl stays rock-stable.
"""

from __future__ import annotations

from rich.text import Text
from textual.widget import Widget

import mascot


class MascotWidget(Widget):
    """Renders one frame of the current owl animation, full width, no wrap."""

    DEFAULT_CSS = """
    MascotWidget {
        height: 4;
        width: 1fr;
        padding: 0 1;
    }
    """

    def __init__(self, info: str = "") -> None:
        super().__init__()
        self._animation: mascot.Animation = mascot.WAITING
        self._tick = 0
        self._info = info  # working directory shown under the brand

    @property
    def interval(self) -> float:
        return self._animation.interval

    @property
    def animation(self) -> mascot.Animation:
        return self._animation

    def set_animation(self, animation: mascot.Animation) -> None:
        """Switch to a new animation, restarting its frame cycle."""
        if animation is self._animation:
            return
        self._animation = animation
        self._tick = 0
        self.refresh()

    def set_info(self, info: str) -> None:
        """Update the working-directory line shown beneath the brand."""
        self._info = info
        self.refresh()

    def advance(self) -> None:
        self._tick += 1
        self.refresh()

    def render(self) -> Text:
        frame = self._animation.frame(self._tick)
        # mascot.render returns raw ANSI; Text.from_ansi keeps the colours without
        # re-interpreting Rich console markup. no_wrap + crop pins it to 4 rows.
        text = Text.from_ansi(mascot.render(frame, self._info))
        text.no_wrap = True
        text.overflow = "crop"
        return text
