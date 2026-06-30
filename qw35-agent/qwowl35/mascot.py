"""Owl mascot animations for the qw35 client.

The owl is four lines tall. The bottom two lines never change; only the eyes
(line 2) and the accessory text after the ``_`` (line 1) animate::

       _   <accessory>
     {<eyes>}
     /)_)
      " "

This module is pure data + rendering, with no terminal I/O, so the REPL can
drive it one tick at a time without blocking on the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ANSI SGR codes. Note: the ESC character must be written as \033 (or \x1b) —
# \e is not a valid Python escape, so it would be printed literally.
RESET = "\033[0m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
PURPLE = "\033[0;35m"
GRAY = "\x1b[38;5;240m"
WHITE = "\033[37m"

OWL_FOOT = (f"{PURPLE} /)_){RESET}", f'{YELLOW}  " "{RESET}')

# Brand shown to the right of the owl's feet. The name is tri-colored — "Qw"
# purple (matching the braces), "owl" yellow (the bird), "35" cyan — and the
# version trails in gray. Nothing is bold. VERSION mirrors qwowl35.__version__;
# kept literal here to avoid the flat-layout package import.
NAME = "Qwowl35"
VERSION = "0.1.0"


def _brand() -> str:
    return (
        f"{PURPLE}Qw{RESET}{YELLOW}owl{RESET}{CYAN}35{RESET}"
        f"{GRAY} v{VERSION}{RESET}"
    )

BG_BLACK = "\033[40m"
WHITE_BOLD = "\033[1;37m"
GREEN_BG = "\033[0;32m\033[40m"
YELLOW_BG = "\033[0;33m\033[40m"


@dataclass(frozen=True)
class Frame:
    eyes: str
    accessory: str = ""


@dataclass(frozen=True)
class Animation:
    name: str
    frames: tuple[Frame, ...]
    interval: float = 0.4

    def frame(self, tick: int) -> Frame:
        return self.frames[tick % len(self.frames)]


class State(str, Enum):
    WAITING = "waiting"
    WAKEUP = "wakeup"
    THINKING = "thinking"
    ERROR = "error"
    INFERENCE = "inference"
    BASH = "bash"
    PREFILL = "prefill"
    EDIT = "edit"
    OK = "ok"
    COPIED = "copied"
    WARN = "warn"
    INFO = "info"


def render(frame: Frame, info: str = "") -> str:
    head = f"{PURPLE}   _{RESET}"
    if frame.accessory:
        head += "   " + frame.accessory
    eyes = f" {PURPLE}{{{RESET}" + frame.eyes + f"{PURPLE}}}{RESET}"
    foot_top, foot_bottom = OWL_FOOT
    # Brand rides on the upper foot row; the working directory (``info``) sits
    # on the lower row beneath it, freeing the head row for the full accessory.
    foot_top = f"{foot_top}    {_brand()}"
    if info:
        foot_bottom = f"{foot_bottom}    {GRAY}{info}{RESET}"
    return "\n".join([head, eyes, foot_top, foot_bottom])


# Waiting for the prompt: sleepy eyes, growing "zzz".
WAITING = Animation(
    State.WAITING.value,
    (
        Frame(f"-{YELLOW},{RESET}-", f"{CYAN}z{RESET}"),
        Frame(f"-{YELLOW},{RESET}-", f"{CYAN}zz{RESET}"),
        Frame(f"-{YELLOW},{RESET}-", f"{CYAN}zzz{RESET}"),
    ),
    interval=0.9,
)

# Waking up: alert eyes, a blinking "!".
WAKEUP = Animation(
    State.WAKEUP.value,
    (
        Frame(f"o{YELLOW},{RESET}o", f"{YELLOW}!{RESET}"),
        Frame(f"o{YELLOW},{RESET}o", ""),
    ),
    interval=0.4,
)

# Thinking: squinting eyes, a blinking "?".
THINKING = Animation(
    State.THINKING.value,
    (
        Frame(f"ò{YELLOW},{RESET}ò", f"{YELLOW}?{RESET}"),
        Frame(f"ò{YELLOW},{RESET}ò", ""),
    ),
    interval=0.4,
)

# Generic inference: growing "..." while the eyes shift around.
INFERENCE = Animation(
    State.INFERENCE.value,
    (
        Frame(f"ò{YELLOW},{RESET}o", "\u2024"),
        Frame(f"o{YELLOW},{RESET}ò", "\u2025"),
        Frame(f"ò{YELLOW},{RESET}ò", "\u2026"),
        Frame(f"o{YELLOW},{RESET}o", "\u2025"),
    ),
    interval=0.8,
)


# Bash: the owl is at a shell. The accessory is a shell prompt with a
# solid black background — frame 0 ">_" alternates with frame 1 "$_". No
# foreground color is set, and there are no leading/trailing spaces, so it
# aligns with the other animations.
BASH = Animation(
    State.BASH.value,
    (
        Frame(
            f"ó{YELLOW},{RESET}ó",
            f"{BG_BLACK}{WHITE}>{RESET}{BG_BLACK}{WHITE}_{RESET}",
        ),
    ),
    interval=1.2,
)


# Prefill: the owl is consuming the prompt. Eyes "ò,ó", accessory a
# green loader that grows from '▁' up to '█' and back, frame by frame.
PREFILL = Animation(
    State.PREFILL.value,
    tuple(
        Frame(
            f"ò{YELLOW},{RESET}ó",
            f"{GREEN}{bar}{RESET} {percent:3d}%",
        )
        for bar, percent in zip(
            ("▁", "▃", "▄", "▅", "▆", "▇", "█"),
            (0, 17, 33, 50, 67, 83, 100),
        )
    ),
    interval=0.2,
)



# Edit: the owl is editing text. Eyes "ù,ù" and a braille "pie" loader
# cycling from a thin slice up to a full block and back.
EDIT = Animation(
    State.EDIT.value,
    tuple(
        Frame(
            f"ù{YELLOW},{RESET}ú",
            f"{glyph}",
        )
        for glyph in ("\u2801", "\u2809", "\u280b", "\u281b", "\u281f", "\u283f")
    ),
    interval=0.2,
)

# OK: a static success frame. A green checkmark accessory, eyes "ò,o".
OK = Animation(
    State.OK.value,
    (Frame(f"ò{YELLOW},{RESET}o", f"{GREEN}\u2713{RESET}"),),
    interval=0.9,
)


# Copied: a static frame confirming a clipboard copy. Green "copied"
# accessory, eyes "ò,o" (mirrors the OK success frame).
COPIED = Animation(
    State.COPIED.value,
    (Frame(f"ò{YELLOW},{RESET}o", f"{GREEN}copied{RESET}"),),
    interval=0.9,
)


def _format_error(code: str, message: str) -> str:
    """Render an error accessory: the ``code`` part is bold, the ``message``
    is the full human-readable detail (the widget crops it to the terminal
    width). Both pieces are colored red to match the eyes."""
    if not code:
        return f"{RED}{message}{RESET}"
    return f"{RED}{WHITE_BOLD}{code}{RESET}{RED} {message}{RESET}"


def warn(message: str = "") -> Animation:
    """Transient warning notice: yellow accessory, alert eyes. The owl carries
    short application warnings (not conversation content); keep ``message``
    short — the widget crops it to the terminal width."""
    return Animation(
        State.WARN.value,
        (Frame(f"o{YELLOW},{RESET}o", f"{YELLOW}{message}{RESET}"),),
        interval=0.9,
    )


def info(message: str = "") -> Animation:
    """Transient info notice: cyan accessory, calm eyes. Used for short
    application status (e.g. ``connected``), never conversation content."""
    return Animation(
        State.INFO.value,
        (Frame(f"o{YELLOW},{RESET}o", f"{CYAN}{message}{RESET}"),),
        interval=0.9,
    )


def error(code: str, message: str = "") -> Animation:
    """Build an error animation. The ``code`` is rendered bold (e.g. ``404``,
    a tool name, or ``ERR``); ``message`` is the human-readable detail, shown
    in full and cropped to the terminal width by the mascot widget."""
    return Animation(
        State.ERROR.value,
        (Frame(f"x{YELLOW},{RESET}x", _format_error(code, message)),),
        interval=0.9,
    )


ANIMATIONS: dict[str, Animation] = {
    WAITING.name: WAITING,
    WAKEUP.name: WAKEUP,
    THINKING.name: THINKING,
    INFERENCE.name: INFERENCE,
    BASH.name: BASH,
    PREFILL.name: PREFILL,
    EDIT.name: EDIT,
    OK.name: OK,
    COPIED.name: COPIED,
}
