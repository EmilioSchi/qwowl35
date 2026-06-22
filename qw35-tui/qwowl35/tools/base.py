"""Convention every tool package under ``tools/`` follows.

Each tool package exposes two things the registry needs:

- the callable(s) it dispatches (``BashTool``, ``HashlineTools``), and
- ``GUIDANCE``: a human-language section appended to the system prompt.

``ToolSpec`` bundles a tool's OpenAI ``tools`` schema(s) with that guidance so
the registry can iterate one ordered list to produce *both* the wire ``tools``
array and the prompt's per-tool guidance. The system prompt is therefore
assembled dynamically from whatever tools are registered, mirroring how
Qwen-Agent builds its prompt from a registry of tools rather than hardcoding a
tool list in prose.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool group: its OpenAI schema(s) and prompt guidance."""

    schemas: list[dict]
    guidance: str
