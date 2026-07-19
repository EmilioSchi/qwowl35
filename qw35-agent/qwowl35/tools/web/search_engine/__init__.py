"""Web-stage tool: `search_engine`, DuckDuckGo search for the web agent.

Scrapes the JS-free HTML endpoint (stdlib only) for ranked results and asks
the public Instant Answer API for an abstract/definition card. Web-agent
only: `web_fetch` reads a page, this finds which page to read.
"""

from __future__ import annotations

from .executor import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS_CAP,
    SEARCH_ENGINE_SCHEMA,
    SearchEngineTool,
)
from .guidance import GUIDANCE

SEARCH_ENGINE_NAME = "search_engine"

__all__ = [
    "GUIDANCE",
    "SEARCH_ENGINE_NAME",
    "SEARCH_ENGINE_SCHEMA",
    "SearchEngineTool",
    "DEFAULT_MAX_RESULTS",
    "MAX_RESULTS_CAP",
]
