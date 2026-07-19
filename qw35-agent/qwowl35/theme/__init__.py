"""The app's coloring — now a *live, swappable* palette.

Historically this module exposed a flat set of hex constants (``BG_BASE``,
``ACCENT``, …) that every region of the TUI drew from. Those names still work
exactly as before — ``theme.ACCENT`` etc. — but they are no longer frozen
literals: a module-level ``__getattr__`` (PEP 562) resolves each token against
the **currently active** :class:`Palette`. So any code that reads ``theme.X`` at
render time automatically follows a theme change, with no call-site changes.

Two consumption paths exist across the app and both are handled:

* **Rich ``Text`` styles** read ``theme.X`` per render → live via ``__getattr__``.
  (Style *constants captured at import time* are the one exception; those are
  converted to per-render helpers at their definition sites.)
* **Textual CSS** cannot read Python at paint time, so colors reach CSS as
  Textual theme variables (``$bg-base`` …). :func:`to_css_variables` builds that
  mapping from a palette; the app registers a Textual ``Theme`` per palette and
  ``self.theme = name`` restyles the whole DOM live.

Intentionally NOT governed here: the mascot art (``mascot.py``,
``mascot_states.py``) and the "Qwowl35" brand string keep their own
purple/yellow/cyan branding.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Palette:
    """A fully-resolved set of the colors the app draws from (all hex strings).

    Field names match the historical ``theme.*`` constants one-to-one so the
    rest of the app keeps its vocabulary.
    """

    # Surfaces
    BG_BASE: str       # soft black — default background of every region
    BG_SURFACE: str    # subtle lift — prompt input, user msgs, prompt mark
    # Text
    FG_BRIGHT: str     # primary text (assistant, user, inputs)
    FG_DIM: str        # secondary (shell output, status, queue, captions)
    FG_MUTED: str      # tertiary (file paths)
    FG_FAINT: str      # thinking / system italics
    FG_GHOST: str      # footer / faint line numbers
    # Accent
    ACCENT: str        # prompt mark, cursor, refs/libs, tool-pending
    # Semantic
    WARNING: str
    ERROR: str
    ERROR_SOFT: str    # inline tool-error text
    SUCCESS: str       # diff-add / OK / success titles
    SUCCESS_SOFT: str  # tool-success body text
    # Scrollbars
    SCROLL_BAR: str
    SCROLL_BAR_HOVER: str
    SCROLL_BAR_ACTIVE: str
    # Code / diff backgrounds
    CODE_BG: str
    DIFF_ADD_BG: str
    DIFF_REMOVE_BG: str
    DIFF_CONTEXT_BG: str
    # Markdown elements (chat prose); themes may pin them via markdown-* overrides
    MD_HEADING: str    # headings (bold)
    MD_LINK: str       # link text / URLs
    MD_CODE: str       # inline code foreground
    MD_QUOTE: str      # block quotes (italic)
    MD_LIST: str       # list bullets / enumerations
    MD_HR: str         # horizontal rules


# The original soft-black / teal palette, kept as the built-in default so the
# app has a working theme even with an empty ``themes/`` directory.
DEFAULT = Palette(
    BG_BASE="#15171c",
    BG_SURFACE="#1d2027",
    FG_BRIGHT="#cbd1d9",
    FG_DIM="#8f96a3",
    FG_MUTED="#9aa0aa",
    FG_FAINT="#6b7280",
    FG_GHOST="#5c5f66",
    ACCENT="#8abeb7",
    WARNING="#d6a55c",
    ERROR="#e06c75",
    ERROR_SOFT="#e0a0a0",
    SUCCESS="#8cc97a",
    SUCCESS_SOFT="#a9d59a",
    SCROLL_BAR="#3a3d46",
    SCROLL_BAR_HOVER="#4b4f5a",
    SCROLL_BAR_ACTIVE="#5a5f6c",
    CODE_BG="#15171c",
    DIFF_ADD_BG="#080f0a",
    DIFF_REMOVE_BG="#120707",
    DIFF_CONTEXT_BG="#15171c",
    MD_HEADING="#8abeb7",   # = ACCENT
    MD_LINK="#8abeb7",      # = ACCENT
    MD_CODE="#a9d59a",      # = SUCCESS_SOFT
    MD_QUOTE="#6b7280",     # = FG_FAINT
    MD_LIST="#8abeb7",      # = ACCENT
    MD_HR="#5c5f66",        # = FG_GHOST
)

_TOKENS = tuple(f.name for f in fields(Palette))

# The active palette. Swapped via ``set_active`` whenever the theme changes.
_active: Palette = DEFAULT


def active() -> Palette:
    """Return the currently active palette."""
    return _active


def set_active(palette: Palette) -> None:
    """Install ``palette`` as the active one; subsequent ``theme.X`` reads use it."""
    global _active
    _active = palette


def is_dark() -> bool:
    """Whether the active palette reads as dark (by background luminance).

    Used to pick light/dark-appropriate syntax-highlighting themes; the ``Palette``
    itself carries no mode flag, so we infer it from ``BG_BASE`` (always ``#rrggbb``).
    """
    h = _active.BG_BASE.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 128


def css_var_name(token: str) -> str:
    """Map a palette token (``BG_BASE``) to its Textual CSS variable (``bg-base``)."""
    return token.lower().replace("_", "-")


def to_css_variables(palette: Palette) -> dict[str, str]:
    """Build the Textual ``Theme.variables`` mapping (``bg-base`` → ``#...``)."""
    return {css_var_name(t): getattr(palette, t) for t in _TOKENS}


def __getattr__(name: str) -> str:
    """Resolve ``theme.BG_BASE`` etc. against the active palette (PEP 562)."""
    if name in _TOKENS:
        return getattr(_active, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
