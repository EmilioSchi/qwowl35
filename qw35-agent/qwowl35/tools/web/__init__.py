"""Web-stage tools: `web_fetch` (pull one page) and `search_engine` (find pages).

Both are wired only into the WEB stage's toolset — no other agent sees them.
"""

from __future__ import annotations

from .search_engine import SEARCH_ENGINE_NAME, SEARCH_ENGINE_SCHEMA, SearchEngineTool
from .web_fetch import WEB_FETCH_NAME, WEB_FETCH_SCHEMA, WebFetchTool

__all__ = [
    "SEARCH_ENGINE_NAME",
    "SEARCH_ENGINE_SCHEMA",
    "SearchEngineTool",
    "WEB_FETCH_NAME",
    "WEB_FETCH_SCHEMA",
    "WebFetchTool",
]
