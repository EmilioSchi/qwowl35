"""Single source of truth for the app's coloring.

Every region of the TUI draws from this palette so the whole app reads as one
coherent "mini-terminal" theme: a soft-black background, a teal accent, and
cool-grey text.

Intentionally NOT governed here: the mascot art (``mascot.py``,
``mascot_states.py``) and the "Qwowl35" brand string keep their own
purple/yellow/cyan branding.

Textual ``CSS``/``DEFAULT_CSS`` are class-level strings; widgets build them as
f-strings referencing these constants. Rich ``Text`` styles reference them
directly.
"""

from __future__ import annotations

# Surfaces
BG_BASE = "#15171c"      # soft black — default background of every region
BG_SURFACE = "#1d2027"   # subtle lift — prompt input, user msgs, prompt mark

# Text
FG_BRIGHT = "#cbd1d9"    # primary text (assistant, user, inputs)
FG_DIM = "#8f96a3"       # secondary (shell output, status, queue, captions)
FG_MUTED = "#9aa0aa"     # tertiary (file paths)
FG_FAINT = "#6b7280"     # thinking / system italics
FG_GHOST = "#5c5f66"     # footer / faint line numbers

# Accent
ACCENT = "#8abeb7"       # teal — prompt mark, cursor, refs/libs, tool-pending

# Semantic (retuned cooler to sit on soft-black)
WARNING = "#d6a55c"
ERROR = "#e06c75"
ERROR_SOFT = "#e0a0a0"   # inline tool-error text
SUCCESS = "#8cc97a"      # diff-add / OK / success titles
SUCCESS_SOFT = "#a9d59a"  # tool-success body text

# Scrollbars (neutral greys)
SCROLL_BAR = "#3a3d46"
SCROLL_BAR_HOVER = "#4b4f5a"
SCROLL_BAR_ACTIVE = "#5a5f6c"

# Code / diff backgrounds
CODE_BG = BG_BASE
DIFF_ADD_BG = "#080f0a"
DIFF_REMOVE_BG = "#120707"
DIFF_CONTEXT_BG = BG_BASE
