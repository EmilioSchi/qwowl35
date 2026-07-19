"""The TUI operating modes (Vim-inspired), shared by the app and orchestrator.

The user selects a mode BEFORE sending a prompt (Shift+Tab or /mode); the mode
is locked while a turn runs. NORMAL/PLAN/WEB/CHAT are user-selectable and pick
the agent that handles the turn; VISUAL and INSERT are display-only states the
orchestrator pushes while a sub-agent (explorer, editor) is active, restoring
the surrounding mode when it finishes.
"""

from __future__ import annotations

from enum import Enum


class Mode(str, Enum):
    NORMAL = "normal"
    VISUAL = "visual"
    INSERT = "insert"
    PLAN = "plan"
    WEB = "web"
    CHAT = "chat"


# The Shift+Tab / "/mode" cycle order. VISUAL and INSERT are never selectable:
# they mirror a sub-agent's lifetime, not a user choice.
USER_MODES: tuple[Mode, ...] = (Mode.NORMAL, Mode.PLAN, Mode.WEB, Mode.CHAT)


def next_mode(mode: Mode) -> Mode:
    """The next user-selectable mode in the cycle (NORMAL after a non-user mode)."""
    if mode not in USER_MODES:
        return Mode.NORMAL
    return USER_MODES[(USER_MODES.index(mode) + 1) % len(USER_MODES)]


# Mode-box fill per mode, as PALETTE TOKEN NAMES (theme.<TOKEN>), resolved at
# render time so a live theme switch recolors the box automatically. Existing
# tokens only — no per-theme JSON churn. VISUAL/INSERT mirror the sub-agent
# colors (Explorer/Editor cards in the chat log) so every visual reference to
# a role shares one color.
MODE_COLOR_TOKENS: dict[Mode, str] = {
    Mode.NORMAL: "ACCENT",
    Mode.INSERT: "WARNING",
    Mode.VISUAL: "ACCENT",
    Mode.PLAN: "ERROR_SOFT",
    Mode.WEB: "FG_BRIGHT",
    Mode.CHAT: "SUCCESS_SOFT",
}
