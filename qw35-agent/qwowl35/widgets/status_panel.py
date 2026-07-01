"""Compact top-of-screen runtime status shown beside the mascot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.text import Text
from textual.widget import Widget

import theme


# Mirrors the server's thinking-budget fractions
# (qw35-server `thinking_budget_for`): low 4%, medium 10%, high 16% of the
# answer budget; xhigh/unspecified fall back to the 16% backstop. Keep in sync.
EFFORT_CAP_PERCENT = {
    "low": 4,
    "medium": 10,
    "high": 16,
    "xhigh": 16,
}


def rough_token_count(text: str) -> int:
    """Cheap live estimate for streamed reasoning text."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def percent(used: int | None, total: int | None) -> float | None:
    if used is None or total is None or total <= 0:
        return None
    return max(0.0, min(100.0, (used / total) * 100))


def effort_cap_percent(think: str, effort: str | None, inferred_thinking: bool = False) -> int | None:
    if think == "off":
        return None
    if think == "on":
        return EFFORT_CAP_PERCENT.get(effort or "xhigh", 16)
    if inferred_thinking:
        return EFFORT_CAP_PERCENT.get(effort or "xhigh", 16)
    return None


def display_path(path: str, max_len: int = 80) -> str:
    home = str(Path.home())
    if path == home:
        path = "~"
    elif path.startswith(home + "/"):
        path = "~/" + path[len(home) + 1:]
    if len(path) <= max_len:
        return path
    keep = max(8, max_len - 3)
    return "..." + path[-keep:]


def compact_count(value: int | None) -> str:
    if value is None:
        return "0"
    if value >= 1_000_000:
        compact = value / 1_000_000
        return f"{compact:.1f}m" if value % 1_000_000 else f"{value // 1_000_000}m"
    if value >= 1_000:
        compact = value / 1_000
        return f"{compact:.1f}k" if value % 1_000 else f"{value // 1_000}k"
    return f"{value:,}"


def fmt_percent(value: float | None, digits: int = 0) -> str:
    if value is None:
        return f"{0:.{digits}f}%"
    return f"{value:.{digits}f}%"


def fmt_tps(value: float | None) -> str:
    """Decode speed in a fixed 4-character field (keeps the bar from jittering).

    Typical qw35 decode sits at ~10-20 tok/s, so one decimal fits; values of
    100+ drop the decimal to stay within four columns.
    """
    if value is None or value <= 0:
        return " 0.0"
    if value >= 100:
        return f"{value:4.0f}"
    return f"{value:4.1f}"


def decode_summary(value: float | None) -> str:
    return f"{fmt_tps(value)} tok/s"


def context_summary(prompt_tokens: int | None, ctx_size: int | None) -> str:
    return f"{fmt_percent(percent(prompt_tokens, ctx_size), 1)}/{compact_count(ctx_size)}"


def compose_status_bar(left: Text, right: Text, width: int | None) -> Text:
    """Justify ``left``/``right`` on one row, or stack them when too narrow.

    Stacking keeps the right-hand host info visible instead of clipping it on a
    narrow terminal; ``overflow="fold"`` folds either half if even one row can't
    fit.
    """
    out = Text(overflow="fold")
    if width and left.cell_len + right.cell_len + 1 > width:
        out.append_text(left)
        out.append("\n")
        out.append_text(right)
        return out
    gap = max(1, width - left.cell_len - right.cell_len) if width else 2
    out.append_text(left)
    out.append(" " * gap)
    out.append_text(right)
    return out


def host_label(base_url: str) -> str:
    """Compact host string for the footer (scheme stripped)."""
    for prefix in ("http://", "https://"):
        if base_url.startswith(prefix):
            return base_url[len(prefix):]
    return base_url


def think_summary(
    think: str,
    effort: str | None,
    used_percent: float | None = None,
    inferred_thinking: bool = False,
) -> str:
    if think == "off":
        return "disabled think"
    if think == "on":
        label = effort or "default"
    elif inferred_thinking:
        label = effort or "auto"
    else:
        label = "auto"
    if used_percent is None:
        return f"{label} think"
    return f"{label} think {fmt_percent(used_percent)}"


def context_line(state: "StatusState") -> str:
    """Footer-left text: context fill plus live thinking-budget usage."""
    cap = effort_cap_percent(state.think, state.effort, state.inferred_thinking)
    basis = _thinking_basis(state.ctx_size, state.prompt_tokens)
    cap_tokens = round(basis * cap / 100) if cap is not None and basis is not None else None
    think_used = percent(state.reasoning_estimate, cap_tokens) if state.reasoning_estimate else None
    return (
        f"{decode_summary(state.decode_tps)}  "
        f"{context_summary(state.prompt_tokens, state.ctx_size)}  "
        f"{think_summary(state.think, state.effort, think_used, state.inferred_thinking)}"
    )


@dataclass
class StatusState:
    base_url: str
    think: str
    effort: str | None
    model: str | None = None
    ready: bool | None = None
    ctx_size: int | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_estimate: int = 0
    inferred_thinking: bool = False
    decode_tps: float | None = None


class StatusBar(Widget):
    """Single footer row: context/think at the left, host at the right.

    Owns the runtime :class:`StatusState`; the app feeds it via the ``update_*``
    methods. The working directory is shown by the mascot, not here.
    """

    DEFAULT_CSS = """
    StatusBar {
        height: auto;
        width: 1fr;
        padding: 0 1;
        color: $fg-dim;
        background: $bg-base;
    }
    """

    def __init__(self, *, base_url: str, think: str, effort: str | None) -> None:
        super().__init__()
        self.state = StatusState(base_url=base_url, think=think, effort=effort)

    def update_health(self, payload: dict | None) -> None:
        if not payload:
            self.state.ready = False
            self.refresh()
            return
        self.state.model = _str_or_none(payload.get("model")) or self.state.model
        self.state.ready = bool(payload.get("decoder_ready"))
        self.state.ctx_size = _int_or_none(payload.get("ctx_size")) or self.state.ctx_size
        self.refresh()

    def update_props(self, payload: dict | None) -> None:
        if not payload:
            return
        settings = payload.get("default_generation_settings")
        if isinstance(settings, dict):
            self.state.ctx_size = _int_or_none(settings.get("n_ctx")) or self.state.ctx_size
            self.refresh()

    def reset_generation(self) -> None:
        self.state.reasoning_estimate = 0
        self.state.inferred_thinking = False
        self.refresh()

    def update_prefill(self, total: int | None) -> None:
        if total is not None:
            self.state.prompt_tokens = total
            self.refresh()

    def update_reasoning(self, text: str) -> None:
        self.state.reasoning_estimate += rough_token_count(text)
        self.state.inferred_thinking = True
        self.refresh()

    def update_usage(self, usage: dict | None) -> None:
        if not usage:
            return
        self.state.prompt_tokens = _int_or_none(usage.get("prompt_tokens")) or self.state.prompt_tokens
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            self.state.cached_tokens = _int_or_none(details.get("cached_tokens"))
        self.refresh()

    def update_timings(self, timings: dict | None) -> None:
        """Pick up the server's measured decode rate (qw35_timings.eval_tps)."""
        if not timings:
            return
        tps = _float_or_none(timings.get("eval_tps"))
        if tps is not None:
            self.state.decode_tps = tps
            self.refresh()

    def _host_text(self) -> Text:
        state = self.state
        if state.ready is True:
            dot_style = theme.SUCCESS
        elif state.ready is False:
            dot_style = theme.ERROR
        else:
            dot_style = theme.FG_GHOST
        label = host_label(state.base_url)
        out = Text()
        out.append("● ", style=dot_style)
        out.append(label, style=theme.FG_DIM)
        return out

    def render(self) -> Text:
        left = Text(context_line(self.state), style=theme.ACCENT)
        right = self._host_text()
        return compose_status_bar(left, right, self.size.width)


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _thinking_basis(ctx_size: int | None, prompt_tokens: int | None) -> int | None:
    if ctx_size is None:
        return None
    if prompt_tokens is None:
        return ctx_size
    return max(0, ctx_size - prompt_tokens)
