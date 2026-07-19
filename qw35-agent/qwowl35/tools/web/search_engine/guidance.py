"""System-prompt guidance for the search_engine tool."""

from __future__ import annotations

GUIDANCE = """\
Use `search_engine` to find pages when you don't already have a URL: give a
short keyword query and it returns ranked results (title, URL, snippet) plus
an instant-answer card when one exists. Pick the most promising result and
`web_fetch` its URL to read it — snippets alone are rarely enough to answer
from. Refine the query and search again if the results miss the mark."""
