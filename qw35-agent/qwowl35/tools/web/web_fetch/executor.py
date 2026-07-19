"""`web_fetch` executor: HTTP GET + HTML-to-text normalization."""

from __future__ import annotations

import re
from html.parser import HTMLParser

import httpx

FETCH_TIMEOUT_SECONDS = 10.0
# Cap what one page may occupy in the model's context.
MAX_CONTENT_CHARS = 100_000

_SAFARI_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15"
)

_ACCEPT_HEADERS = {
    "auto": "text/markdown, text/html;q=0.7, text/plain;q=0.5, */*;q=0.1",
    "markdown": "text/markdown, */*;q=0.1",
    "html": "text/html, */*;q=0.1",
    "text": "text/plain, */*;q=0.1",
}

WEB_FETCH_SCHEMA = {
    "name": "web_fetch",
    "description": (
        "Fetches content from a specified URL and returns it as plain text\n"
        "- Takes a URL and a prompt as input\n"
        "- Supports content negotiation for markdown (reduces tokens by ~80%)\n"
        "- Fetches the URL content, converts HTML to text if needed\n"
        "- Use this tool when you need to retrieve and analyze web content\n\n"
        "Usage notes:\n"
        "  - The URL must be a fully-formed valid URL\n"
        "  - The prompt should describe what information you want to extract "
        "from the page\n"
        "  - format parameter (optional): controls only the Accept header sent "
        "to the server. All content is normalized to plain text.\n"
        "  - This tool is read-only and does not modify any files\n"
        "  - Results may be truncated if the content is very large"
    ),
    "parameters": {
        "properties": {
            "url": {
                "description": "The URL to fetch content from",
                "type": "string",
            },
            "prompt": {
                "description": "The prompt to run on the fetched content",
                "type": "string",
            },
            "format": {
                "description": (
                    "Preferred content format (Accept header only): auto "
                    "(default, prefers markdown), markdown, html, or text. All "
                    "content is normalized to plain text."
                ),
                "type": "string",
                "enum": ["auto", "markdown", "html", "text"],
            },
            "compress": {
                "type": "boolean",
                "description": "Optional: false returns the full uncompressed output.",
            },
        },
        "required": ["url", "prompt"],
        "type": "object",
    },
}


class _TextExtractor(HTMLParser):
    """Minimal HTML → readable text: drops script/style, keeps block breaks."""

    _SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    _BLOCK = {
        "p", "div", "section", "article", "header", "footer", "li", "ul", "ol",
        "table", "tr", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "pre",
        "blockquote",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            self._chunks.append(data)

    def text(self) -> str:
        text = "".join(self._chunks)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - fall back to a crude strip
        return re.sub(r"<[^>]+>", " ", html)
    return parser.text()


class WebFetchTool:
    """Synchronous fetch (the registry runs it in a thread)."""

    def execute(self, arguments: dict) -> str:
        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            return "Error: 'url' is required."
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return f"Error: The URL must be a fully-formed valid URL: {url!r}"
        prompt = arguments.get("prompt")
        prompt = prompt.strip() if isinstance(prompt, str) else ""
        fmt = arguments.get("format")
        accept = _ACCEPT_HEADERS.get(fmt if isinstance(fmt, str) else "auto", _ACCEPT_HEADERS["auto"])

        try:
            # HTTP/2 matters here: some sites' bot-protection (e.g. Wikimedia's
            # edge) blocks plain HTTP/1.1 clients outright, regardless of
            # User-Agent, and only allows the request through over h2.
            with httpx.Client(http2=True) as client:
                response = client.get(
                    url,
                    headers={"Accept": accept, "User-Agent": _SAFARI_USER_AGENT},
                    timeout=FETCH_TIMEOUT_SECONDS,
                    follow_redirects=True,
                )
        except httpx.HTTPError as exc:
            return f"Error: fetch failed for {url}: {exc}"
        if response.status_code >= 400:
            return f"Error: {url} returned HTTP {response.status_code}."

        content_type = response.headers.get("content-type", "").lower()
        body = response.text
        if "html" in content_type or (
            "text/" not in content_type and body.lstrip()[:1] == "<"
        ):
            text = html_to_text(body)
        else:
            text = body.strip()

        truncated = len(text) > MAX_CONTENT_CHARS
        if truncated:
            text = text[:MAX_CONTENT_CHARS]
        header = f"Content from {url}"
        if prompt:
            header += f" (you wanted: {prompt})"
        note = "\n\n(Content truncated at 100000 characters.)" if truncated else ""
        return f"{header}:\n\n{text}{note}"
