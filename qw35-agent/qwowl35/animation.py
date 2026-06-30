"""Preview the owl mascot animations in the terminal.

Run with ``python -m qw35_client.animation``. Each state is played in turn,
redrawn in place with ANSI escapes. Ctrl-C exits cleanly.
"""

from __future__ import annotations

import sys
import time

from .mascot import (
    BASH,
    EDIT,
    INFERENCE,
    OK,
    PREFILL,
    THINKING,
    WAITING,
    WAKEUP,
    Animation,
    error,
    render,
)


LINES = len(render(WAITING.frame(0)).split("\n"))

# Move the cursor up `LINES` rows to the start of the block, then clear from
# there to the end of the screen before redrawing.
_REWIND = f"\033[{LINES}F\033[J"


def play(animation: Animation, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    tick = 0
    while True:
        if tick:
            sys.stdout.write(_REWIND)
        sys.stdout.write(render(animation.frame(tick)) + "\n")
        sys.stdout.flush()
        if time.monotonic() >= deadline:
            break
        time.sleep(animation.interval)
        tick += 1


def main() -> None:
    sys.stdout.write("\033[?25l")  # hide cursor
    try:
        for animation in (
            WAITING,
            WAKEUP,
            THINKING,
            EDIT,
            OK,
            INFERENCE,
            PREFILL,
            BASH,
            error("404", "Not Found."),
        ):
            print(f"\n{animation.name}:")
            play(animation, seconds=4.0)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")  # show cursor
        sys.stdout.flush()


if __name__ == "__main__":
    main()
