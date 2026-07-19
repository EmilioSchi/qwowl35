"""`search_engine` executor: DuckDuckGo HTML search + Instant Answer API.

Stdlib only: the HTML endpoint (html.duckduckgo.com) is scraped with an
HTMLParser subclass, pagination follows the "Next" form's hidden inputs, and
the public Instant Answer API supplies the abstract/definition card when one
exists. Live sport/weather/stock cards are rendered only by the JS frontend
and are not exposed here.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from http.cookiejar import CookieJar

HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
API_ENDPOINT = "https://api.duckduckgo.com/"

REQUEST_TIMEOUT_SECONDS = 15.0
# Between pagination requests (be polite).
PAGINATION_PAUSE_SECONDS = 0.5
DEFAULT_MAX_RESULTS = 10
MAX_RESULTS_CAP = 25

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

SEARCH_ENGINE_SCHEMA = {
    "name": "search_engine",
    "description": (
        "Searches the web (DuckDuckGo) and returns a ranked list of results\n"
        "- Takes a search query and returns title, URL and snippet per result\n"
        "- Also returns an instant-answer card (abstract, definition, computed "
        "answer) when one exists\n"
        "- Use this tool to FIND pages; use web_fetch to READ a result's URL\n\n"
        "Usage notes:\n"
        "  - Keep queries short and keyword-like, as in a search box\n"
        "  - max_results (optional): how many results to return "
        f"(default {DEFAULT_MAX_RESULTS}, max {MAX_RESULTS_CAP})\n"
        "  - This tool is read-only and does not modify any files"
    ),
    "parameters": {
        "properties": {
            "query": {
                "description": "The search terms",
                "type": "string",
            },
            "max_results": {
                "description": (
                    f"Maximum number of results to return (default "
                    f"{DEFAULT_MAX_RESULTS}, max {MAX_RESULTS_CAP})"
                ),
                "type": "integer",
            },
        },
        "required": ["query"],
        "type": "object",
    },
}


@dataclass
class SearchResult:
    title: str
    description: str
    url: str
    domain: str


@dataclass
class InstantAnswer:
    heading: str = ""
    abstract: str = ""
    abstract_source: str = ""
    abstract_url: str = ""
    answer: str = ""
    definition: str = ""
    infobox: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.heading or self.abstract or self.answer or self.definition)


class _HttpClient:
    """Cookie-carrying urllib opener (DuckDuckGo's HTML endpoint wants one)."""

    def __init__(self, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        self.timeout = timeout
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(CookieJar())
        )
        self.opener.addheaders = [("User-Agent", _USER_AGENT)]

    def _read(self, response) -> str:  # noqa: ANN001
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")

    def get(self, url: str, params: dict | None = None) -> str:
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        with self.opener.open(url, timeout=self.timeout) as resp:
            return self._read(resp)

    def post(self, url: str, data: dict) -> str:
        body = urllib.parse.urlencode(data).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with self.opener.open(request, timeout=self.timeout) as resp:
            return self._read(resp)


class ResultsPageParser(HTMLParser):
    """Parses one html.duckduckgo.com results page: result divs + "Next" form."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self.next_page: dict | None = None  # hidden inputs of the "Next" form

        self._div_depth = 0        # current <div> nesting depth
        self._result_depth = None  # depth at which the open result div started
        self._nav_depth = None     # depth at which an open nav-link div started
        self._current: dict | None = None
        self._is_ad = False
        self._capture = None       # "title" | "description" | "domain"
        self._text: list[str] = []
        self._form: dict | None = None  # hidden inputs of the form being read
        self._form_is_next = False

    @staticmethod
    def _classes(attrs) -> list[str]:  # noqa: ANN001
        for name, value in attrs:
            if name == "class" and value:
                return value.split()
        return []

    @staticmethod
    def _attr(attrs, wanted):  # noqa: ANN001
        for name, value in attrs:
            if name == wanted:
                return value
        return None

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag == "div":
            self._div_depth += 1
            classes = self._classes(attrs)
            if "result" in classes and self._result_depth is None:
                self._result_depth = self._div_depth
                self._is_ad = any(c.startswith("result--ad") for c in classes)
                self._current = {"title": "", "description": "",
                                 "url": "", "domain": ""}
            elif "nav-link" in classes and self._nav_depth is None:
                self._nav_depth = self._div_depth

        elif tag in ("a", "span") and self._current is not None:
            classes = self._classes(attrs)
            if "result__a" in classes:
                self._current["url"] = self._attr(attrs, "href") or ""
                self._start_capture("title")
            elif "result__snippet" in classes:
                self._start_capture("description")
            elif "result__url" in classes:
                self._start_capture("domain")

        elif tag == "form" and self._nav_depth is not None:
            self._form = {}
            self._form_is_next = False

        elif tag == "input" and self._form is not None:
            itype = (self._attr(attrs, "type") or "").lower()
            name = self._attr(attrs, "name")
            value = self._attr(attrs, "value") or ""
            if itype == "hidden" and name:
                self._form[name] = value
            elif itype == "submit" and value.strip().lower() == "next":
                self._form_is_next = True

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("a", "span") and self._capture is not None:
            self._end_capture()

        elif tag == "div":
            if self._div_depth == self._result_depth:
                self._flush_result()
                self._result_depth = None
            if self._div_depth == self._nav_depth:
                self._nav_depth = None
            self._div_depth = max(0, self._div_depth - 1)

        elif tag == "form" and self._form is not None:
            if self._form_is_next and self.next_page is None:
                self.next_page = self._form
            self._form = None

    def _start_capture(self, into: str) -> None:
        if self._capture is not None:   # unbalanced markup: close previous
            self._end_capture()
        self._capture = into
        self._text = []

    def _end_capture(self) -> None:
        text = " ".join("".join(self._text).split())
        self._current[self._capture] = text
        self._capture = None
        self._text = []

    def _flush_result(self) -> None:
        if self._capture is not None:
            self._end_capture()
        if self._current and self._current["url"] and not self._is_ad:
            self.results.append(self._current)
        self._current = None
        self._is_ad = False


def clean_url(href: str) -> str:
    """DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<real-url>."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        real = urllib.parse.parse_qs(parsed.query).get("uddg")
        if real:
            return real[0]
    return href


class SearchEngine:
    REGION = "wt-wt"  # no localization

    def __init__(
        self,
        pause: float = PAGINATION_PAUSE_SECONDS,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.pause = pause
        self.http = _HttpClient(timeout=timeout)

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[SearchResult]:
        results: list[SearchResult] = []
        form: dict = {"q": query, "b": "", "kl": self.REGION}

        while len(results) < max_results:
            html = self.http.post(HTML_ENDPOINT, form)
            parser = ResultsPageParser()
            parser.feed(html)

            for raw in parser.results:
                results.append(
                    SearchResult(
                        title=raw["title"],
                        description=raw["description"],
                        url=clean_url(raw["url"]),
                        domain=raw["domain"],
                    )
                )
                if len(results) >= max_results:
                    break

            if parser.next_page is None:
                break
            form = parser.next_page
            time.sleep(self.pause)

        return results

    def instant_answer(self, query: str) -> InstantAnswer:
        raw = self.http.get(
            API_ENDPOINT,
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        )
        j = json.loads(raw)

        infobox = {}
        for item in (j.get("Infobox") or {}).get("content", []):
            if item.get("label"):
                infobox[item["label"]] = item.get("value", "")

        return InstantAnswer(
            heading=j.get("Heading", ""),
            abstract=j.get("AbstractText", ""),
            abstract_source=j.get("AbstractSource", ""),
            abstract_url=j.get("AbstractURL", ""),
            answer=j.get("Answer", ""),
            definition=j.get("Definition", ""),
            infobox=infobox,
        )


def format_results(card: InstantAnswer, results: list[SearchResult], query: str) -> str:
    """One plain-text block: the instant-answer card, then numbered results."""
    lines: list[str] = [f"Search results for {query!r}:"]
    if not card.is_empty:
        lines.append("")
        if card.heading:
            lines.append(card.heading)
        if card.answer:
            lines.append(f"  answer: {card.answer}")
        if card.definition:
            lines.append(f"  definition: {card.definition}")
        if card.abstract:
            lines.append(f"  {card.abstract}")
            if card.abstract_source or card.abstract_url:
                lines.append(
                    f"  source: {card.abstract_source} — {card.abstract_url}"
                )
    if not results:
        lines.append("\nNo results.")
        return "\n".join(lines)
    for i, r in enumerate(results, 1):
        lines.append(f"\n{i}. {r.title}")
        lines.append(f"   {r.url}")
        if r.description:
            lines.append(f"   {r.description}")
    return "\n".join(lines)


class SearchEngineTool:
    """Synchronous search (the registry runs it in a thread)."""

    def __init__(self) -> None:
        self.engine = SearchEngine()

    def execute(self, arguments: dict) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return "Error: 'query' is required."
        query = query.strip()
        max_results = arguments.get("max_results")
        if not isinstance(max_results, int) or isinstance(max_results, bool):
            max_results = DEFAULT_MAX_RESULTS
        max_results = max(1, min(max_results, MAX_RESULTS_CAP))

        try:
            results = self.engine.search(query, max_results=max_results)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return f"Error: search failed for {query!r}: {exc}"

        # The card is a bonus: never let its endpoint failing kill the search.
        try:
            card = self.engine.instant_answer(query)
        except Exception:  # noqa: BLE001
            card = InstantAnswer()

        return format_results(card, results, query)
