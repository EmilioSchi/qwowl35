"""Dynamic mascot frames that aren't baked into mascot.py.

`prefill(percent)` mirrors mascot.PREFILL but shows a *real* percentage (and a
matching bar height) reported by the server, instead of the canned 0→100 sweep.
Built from mascot's own primitives so mascot.py stays untouched.
"""

from __future__ import annotations

import mascot
from mascot import Animation, Frame, State

_BARS = ("▁", "▃", "▄", "▅", "▆", "▇", "█")


def prefill(percent: float) -> Animation:
    p = max(0, min(100, int(round(percent))))
    idx = min(len(_BARS) - 1, p * len(_BARS) // 100)
    bar = _BARS[idx]
    accessory = f"{mascot.GREEN}{bar}{mascot.RESET} {p:3d}%"
    return Animation(
        State.PREFILL.value,
        (Frame(f"ò{mascot.YELLOW},{mascot.RESET}ó", accessory),),
        interval=0.2,
    )
