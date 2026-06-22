"""Minimal configuration for the qwowl35 agent.

Sampling (temperature, top_p, penalties, …) is owned by the server: launch
``qw35-server --mode <preset>`` to pick a profile. The client only carries the
think/no-think principal choice and an optional reasoning-effort override, so a
single server preset governs every connected client. See the qw35-server
``--mode`` help for the four official Qwen3.5 profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

# Thinking control, mirroring llama.cpp's `--reasoning on|off|auto`:
#   auto → send nothing; defer to the server's `--mode` default
#   on   → request thinking (enable_thinking=true) + optional reasoning_effort
#   off  → explicitly disable thinking (enable_thinking=false). This is the fix
#          for the Qwen3.5 "thinks by default" trap (cf. qwen-code #4505):
#          sampling params never disable thinking — the flag must be sent.
ALLOWED_THINK = ("auto", "on", "off")

# Reasoning levels for the optional effort override (only meaningful with
# thinking on). The server maps each to a thinking-token budget (low ≈ 20%,
# medium ≈ 50%, high ≈ 80% of max_tokens; xhigh is uncapped).
ALLOWED_EFFORTS = ("low", "medium", "high", "xhigh")

# Input prompt: history persistence + paste-collapsing thresholds (little-coder
# parity). History is stored as JSON-lines so multiline submissions round-trip.
HISTORY_DIR = Path.home() / ".qwowl35"
HISTORY_FILE = HISTORY_DIR / "history"
HISTORY_MAX = 100
PASTE_LINE_THRESHOLD = 10
PASTE_CHAR_THRESHOLD = 1000

# Tool result preview length before Ctrl+O expands it.
TOOL_PREVIEW_LINES = 20


@dataclass(frozen=True)
class Config:
    base_url: str = "http://127.0.0.1:8080"

    # Thinking principal choice (server preset decides when "auto").
    think: str = "auto"
    # Optional reasoning-effort override; only sent when thinking is on.
    reasoning_effort: str | None = None

    # Networking.
    request_timeout: float = 600.0
    # Optional per-request completion cap. None means use the server default.
    max_tokens: int | None = None

    # When true, the bash tool runs in restricted mode. Default false: bash is
    # gated only by the interactive approval modal.
    restricted_bash: bool = False

    def __post_init__(self) -> None:
        if self.think not in ALLOWED_THINK:
            object.__setattr__(self, "think", "auto")
        if self.reasoning_effort is not None and self.reasoning_effort not in ALLOWED_EFFORTS:
            object.__setattr__(self, "reasoning_effort", None)

    def gen_params(self) -> dict:
        """Request fields spread into a chat-completions request.

        Only thinking and optional token-limit fields are sent; sampling is the
        server's job.
        """
        params: dict = {}
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.think == "off":
            # Explicit disable. Omit reasoning_effort so it can't re-enable.
            params["enable_thinking"] = False
            return params
        if self.think == "on":
            params.update({"enable_thinking": True, "preserve_thinking": True})
            if self.reasoning_effort is not None:
                params["reasoning_effort"] = self.reasoning_effort
            return params
        # "auto": defer entirely to the server's --mode default.
        return params


def clamp_think(value: str | None) -> str | None:
    """Map a requested think state onto an allowed value (None = unset)."""
    if value and value.lower() in ALLOWED_THINK:
        return value.lower()
    return None


def clamp_effort(value: str | None) -> str | None:
    """Map a requested effort onto an allowed level (None = unset)."""
    if value and value.lower() in ALLOWED_EFFORTS:
        return value.lower()
    return None


def load_config(
    *,
    base_url: str | None = None,
    think: str | None = None,
    reasoning_effort: str | None = None,
    restricted_bash: bool | None = None,
    max_tokens: int | None = None,
) -> Config:
    """Build a Config from dataclass defaults, overridden by explicit args.

    Configuration is CLI-only; there are no environment-variable overrides.
    """
    cfg = Config()

    if base_url:
        cfg = replace(cfg, base_url=base_url)
    arg_think = clamp_think(think)
    if arg_think:
        cfg = replace(cfg, think=arg_think)
    arg_effort = clamp_effort(reasoning_effort)
    if arg_effort:
        cfg = replace(cfg, reasoning_effort=arg_effort)
    if restricted_bash is not None:
        cfg = replace(cfg, restricted_bash=restricted_bash)
    if max_tokens is not None and max_tokens > 0:
        cfg = replace(cfg, max_tokens=max_tokens)

    return cfg
