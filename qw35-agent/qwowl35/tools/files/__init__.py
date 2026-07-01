"""Id-based file tools (beginTransaction/edit/insert/delete), backed by the hashline library.

Exposes ``HashlineTools`` (the OpenAI-schema adapter) and ``GUIDANCE`` (the
system-prompt section for these tools). The vendored ``hashline`` package holds
the actual, fragile edit logic and is imported unchanged via ``adapter``.
"""

from __future__ import annotations

from .adapter import HashlineTools  # noqa: F401
from .guidance import GUIDANCE  # noqa: F401

__all__ = ["HashlineTools", "GUIDANCE"]
