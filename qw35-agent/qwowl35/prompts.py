"""The system prompt, assembled dynamically from the registered tools.

The qw35-server also injects its own formal ``# Tools`` block from the ``tools``
field of the request (the canonical Qwen3.5 advertisement). This prompt is the
human-language layer on top: a small tool-agnostic preamble plus each tool's own
``GUIDANCE`` section, collected from the registry — so the prompt follows the
registered tool list instead of hardcoding tool names in prose (mirroring how
Qwen-Agent builds its prompt from a registry of tools).
"""

from __future__ import annotations

import os
import platform

from tools.bash import GUIDANCE as BASH_GUIDANCE
from tools.files import GUIDANCE as FILES_GUIDANCE

# Tool-agnostic framing. Per-tool usage notes and XML examples live with each
# tool (``tools/<tool>/guidance.py``) and are appended after this.
PREAMBLE = """\
You are qwowl35, a coding agent. Read, edit, and run files to solve
the task. Be concise; verify before declaring done. Working dir: <<CWD>>. Use
relative paths. Platform: <<PLATFORM>>.

Do not emit JSON inside <tool_call>. Use nested XML; do not put arguments as XML attributes.
Each call has one <function=tool_name> element and child <parameter=name>value</parameter> elements.
Do not wrap one tool inside bash; use the file tools directly.

Work loop: beginTransaction -> mutate by id -> verify with bash."""

# Default guidance order when no registry is supplied (matches ToolRegistry's
# schemas() order: file tools first, then bash).
_DEFAULT_GUIDANCE_SECTIONS = [FILES_GUIDANCE, BASH_GUIDANCE]


def build_system_message(cwd: str | None = None, registry=None) -> dict:
    """Build the system message, grounding the model in its working directory.

    The model is otherwise never told where it is, so the old "use absolute paths
    when in doubt" advice made it fabricate roots like ``/app`` or ``/home/...``.
    Injecting the real cwd + preferring relative paths fixes that. ``cwd``
    defaults to the process working directory at call time (the TUI's launch dir,
    or the headless runner's scratch dir after it chdirs in).

    Per-tool guidance is taken from ``registry.guidance_sections()`` when a
    registry is given, so the prompt follows the actually-registered tools. When
    no registry is supplied, all built-in tools' guidance is used.
    """
    if cwd is None:
        cwd = os.getcwd()
    sections = (
        registry.guidance_sections()
        if registry is not None
        else _DEFAULT_GUIDANCE_SECTIONS
    )
    content = "\n\n".join([PREAMBLE, *sections])
    content = content.replace("<<CWD>>", cwd).replace("<<PLATFORM>>", _platform_summary())
    return {"role": "system", "content": content}


def _platform_summary() -> str:
    system = platform.system() or os.name
    machine = platform.machine()
    if system == "Darwin":
        detail = "macOS/Darwin with BSD userland"
    elif system == "Linux":
        detail = "Linux with GNU userland"
    else:
        detail = system
    return f"{detail} {machine}".strip()
