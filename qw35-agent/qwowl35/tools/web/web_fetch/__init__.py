"""Web-stage tool: qwen-code's `web_fetch`, ported to a local fetch.

Same wire name, parameters, and Accept-header negotiation as qwen-code's
web-fetch.ts (pinned commit 7417805). The original pipes the page through a
small cloud model; here the normalized text is returned directly and the agent
does its own reading — the `prompt` parameter is echoed back as the framing so
the model remembers what it was looking for.
"""

from __future__ import annotations

from .executor import FETCH_TIMEOUT_SECONDS, MAX_CONTENT_CHARS, WEB_FETCH_SCHEMA, WebFetchTool
from .guidance import GUIDANCE

WEB_FETCH_NAME = "web_fetch"

__all__ = [
    "GUIDANCE",
    "WEB_FETCH_NAME",
    "WEB_FETCH_SCHEMA",
    "WebFetchTool",
    "FETCH_TIMEOUT_SECONDS",
    "MAX_CONTENT_CHARS",
]
