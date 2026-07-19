"""The mode agents: one module per agent, one registry to look them up.

The user-selected TUI mode picks the agent that handles the turn: NORMAL runs
the freestyle executor, PLAN runs the planner (which spawns explorers through
its `explore` tool and hands todos to executors), WEB the web agent, CHAT the
chat agent. The editor (executor's `edit`) and explorer (planner's `explore`)
are sub-agents: segregated toolsets, fresh contexts on the scratch GPU
session, dropped when they finish.
"""

from __future__ import annotations

from . import chat, editor, explorer, freestyle, planner, web
from .base import SESSION_MAIN, SESSION_SCRATCH, AgentSpec
from .pipeline import PipelineRegistry

AGENTS: dict[str, AgentSpec] = {
    "chat": chat.SPEC,
    "web": web.SPEC,
    "explore": explorer.SPEC,
    "planner": planner.SPEC,
    "execute": freestyle.SPEC,
    "editor": editor.SPEC,
}

__all__ = [
    "AGENTS",
    "AgentSpec",
    "SESSION_MAIN",
    "SESSION_SCRATCH",
    "PipelineRegistry",
    "chat",
    "editor",
    "explorer",
    "freestyle",
    "planner",
    "web",
]
