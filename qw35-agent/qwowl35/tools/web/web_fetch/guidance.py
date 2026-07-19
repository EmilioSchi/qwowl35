"""System-prompt guidance for the web_fetch tool."""

from __future__ import annotations

GUIDANCE = """\
Use `web_fetch` to pull a page's content as plain text. Give the full URL and a
`prompt` saying what you are looking for; extract what matters from the returned
text yourself and quote only the relevant parts in your findings.
Long results are auto-compressed; a `[compressed: ...]` tail reports what was
elided — re-call the same tool with `compress:false` only if you truly need
everything."""
