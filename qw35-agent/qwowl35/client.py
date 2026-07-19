"""Async streaming client for the qw35-server OpenAI-compatible API.

The server parses the model's ``<tool_call>`` XML into structured tool-call
deltas for us, so this client only has to: (1) read the SSE stream, (2) classify
each delta into a small set of events, and (3) reassemble the per-call argument
JSON by its stable ``index``.
"""

from __future__ import annotations

import json
import re
import shlex
from html import unescape as html_unescape
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

import httpx


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class Qw35Error(Exception):
    """A typed error carrying the server's error code and HTTP status."""

    def __init__(self, code: str, message: str, http_status: int | None = None, kind: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.kind = kind

    @property
    def is_unavailable(self) -> bool:
        return self.http_status == 501 or self.code == "inference_unavailable"

    def short_code(self) -> str:
        """A compact code for the mascot error frame (bold part)."""
        if self.http_status:
            return str(self.http_status)
        return (self.code or "ERR")[:6]


# --------------------------------------------------------------------------- #
# Stream events
# --------------------------------------------------------------------------- #
@dataclass
class ContentDelta:
    text: str


@dataclass
class ReasoningDelta:
    text: str


@dataclass
class ToolCallBegin:
    index: int
    id: str
    name: str


@dataclass
class ToolCallArgsDelta:
    index: int
    fragment: str


@dataclass
class ToolCallName:
    """qw35 side-channel: the streamed call's function name became known.

    With ``stream_tool_call_xml`` the Begin delta arrives the moment the server
    sees ``<tool_call>`` (empty name, raw XML fragments follow); the name is
    delivered as soon as the header is recognizable and again, authoritatively,
    when the block parses. The last one wins."""

    index: int
    name: str


@dataclass
class ToolCallFinal:
    """qw35 side-channel: authoritative parsed arguments JSON for the call.

    Replaces (not appends to) whatever raw XML fragments were streamed."""

    index: int
    arguments: str


@dataclass
class ToolCallDemoted:
    """qw35 side-channel: the streamed block failed to parse as a tool call.

    The call at ``index`` must be dropped; its raw text follows as ordinary
    content/reasoning deltas (and the agent's malformed-call retry sees it)."""

    index: int


@dataclass
class PrefillProgress:
    percent: float
    processed: int
    total: int
    # The serving session's live context ceiling (qw35 servers grow it on
    # demand); None from servers that don't report it.
    session_ctx: int | None = None


@dataclass
class Finish:
    reason: str


@dataclass
class Usage:
    usage: dict
    timings: dict


StreamEvent = (
    ContentDelta
    | ReasoningDelta
    | ToolCallBegin
    | ToolCallArgsDelta
    | ToolCallName
    | ToolCallFinal
    | ToolCallDemoted
    | PrefillProgress
    | Finish
    | Usage
)


# --------------------------------------------------------------------------- #
# Accumulator: builds one assistant turn from the event stream
# --------------------------------------------------------------------------- #
@dataclass
class _PendingCall:
    id: str
    name: str
    args_buffer: str = ""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    index: int = 0


@dataclass
class AssistantTurn:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict | None = None
    timings: dict | None = None


def _parse_tool_args(buffer: str) -> dict:
    """Parse a tool call's argument JSON tolerantly.

    Models frequently emit edit/text fields containing real code with literal
    newlines, tabs, or quotes that aren't strictly escaped. Python's json rejects
    control chars inside strings, which would make a fully-present field look
    'missing'. We therefore accept lenient JSON (strict=False) and try a couple of
    light repairs before giving up. Field ORDER never matters (JSON is unordered)."""
    text = (buffer or "").strip()
    if not text:
        return {}
    # 1) strict, 2) allow literal control chars in strings (the common case for
    # multi-line code), 3) strip a trailing comma before } or ].
    candidates = [text, text]
    last_error: json.JSONDecodeError | None = None
    for i, candidate in enumerate(candidates):
        try:
            value = json.loads(candidate, strict=(i == 0))
            if isinstance(value, dict):
                return value
            return _invalid_tool_args(_non_object_json_error(value))
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    repaired = re.sub(r",\s*([}\]])", r"\1", text)
    if repaired != text:
        try:
            value = json.loads(repaired, strict=False)
            if isinstance(value, dict):
                return value
            return _invalid_tool_args(_non_object_json_error(value))
        except json.JSONDecodeError as exc:
            last_error = exc
            pass
    command = recover_json_string_field_object(text, "command")
    if command is not None:
        return {"command": command}
    recovered = _parse_unquoted_content_object(text)
    if recovered is not None:
        return recovered
    xml_args = parse_xml_tool_args(text)
    if xml_args is not None:
        return xml_args
    return _invalid_tool_args(_json_error_message(last_error))


def _invalid_tool_args(detail: str | None = None) -> dict:
    error = {"_invalid_json": True}
    if detail:
        error["_json_error"] = detail
    return error


def _json_error_message(exc: json.JSONDecodeError | None) -> str | None:
    if exc is None:
        return None
    return f"{exc.msg} at line {exc.lineno} column {exc.colno}"


def _non_object_json_error(value: object) -> str:
    if value is None:
        kind = "null"
    elif isinstance(value, list):
        kind = "array"
    elif isinstance(value, str):
        kind = "string"
    elif isinstance(value, bool):
        kind = "boolean"
    elif isinstance(value, (int, float)):
        kind = "number"
    else:
        kind = type(value).__name__
    return f"parsed as {kind}; tool arguments must be a JSON object"


def recover_json_string_field_object(text: str, field: str) -> str | None:
    """Recover `{"field":"..."}` when the string contains unescaped quotes.

    This intentionally handles only the single-field object shape used by bash
    calls, plus the common typo `{"field="..."}`. The opening quote after the
    colon/equal sign and the final quote before the closing brace are treated as
    the JSON string delimiters; any quotes inside become literal command text.
    """
    stripped = (text or "").strip()
    match = re.match(
        r'^\{\s*"' + re.escape(field) + r'"\s*:\s*"'
        r'|^\{\s*"' + re.escape(field) + r'\s*=\s*"',
        stripped,
        re.DOTALL,
    )
    if match is None or not stripped.endswith("}"):
        return None

    before_brace = stripped.rstrip("}").rstrip()
    closing_quote = _last_unescaped_quote(before_brace)
    if closing_quote < match.end():
        return None
    recovered = _decode_relaxed_json_string(before_brace[match.end():closing_quote])
    return _trim_dangling_recovered_quote(recovered)


def _last_unescaped_quote(text: str) -> int:
    for index in range(len(text) - 1, -1, -1):
        if text[index] != '"':
            continue
        slash_count = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            slash_count += 1
            cursor -= 1
        if slash_count % 2 == 0:
            return index
    return -1


def _decode_relaxed_json_string(raw: str) -> str:
    escapes = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    out: list[str] = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= len(raw):
            out.append("\\")
            break
        esc = raw[i]
        i += 1
        if esc == "u" and i + 4 <= len(raw):
            digits = raw[i:i + 4]
            try:
                out.append(chr(int(digits, 16)))
                i += 4
                continue
            except ValueError:
                pass
        if esc in escapes:
            out.append(escapes[esc])
        else:
            out.append("\\" + esc)
    return "".join(out)


def _trim_dangling_recovered_quote(command: str) -> str:
    if not command.endswith(('"', "'")):
        return command
    try:
        shlex.split(command)
        return command
    except ValueError:
        trimmed = command[:-1].rstrip()
    try:
        shlex.split(trimmed)
        return trimmed
    except ValueError:
        return command


def parse_xml_tool_args(text: str) -> dict | None:
    """Recover nested XML tool arguments if they reach the client unchanged."""
    if "<" not in text or ">" not in text:
        return None

    args: dict[str, object] = {}
    for name, value in _iter_xml_parameters(text):
        args[name] = _json_or_string(value)
    if args:
        return args

    attrs = _parse_compact_xml_attributes(text)
    return attrs or None


def recover_xml_parameter(text: str, field: str, *, partial: bool = False) -> str | None:
    """Return one XML parameter value, optionally from an unfinished tag body."""
    cursor = 0
    while True:
        start = text.find("<parameter", cursor)
        if start == -1:
            return None
        open_end = text.find(">", start)
        if open_end == -1:
            return None
        tag_body = text[start + 1:open_end].strip()
        name = _parameter_name_from_open_tag(tag_body)
        value_start = open_end + 1
        close = re.search(r"</\s*parameter\s*>", text[value_start:], re.IGNORECASE)
        if name == field:
            if close is None:
                if not partial:
                    return None
                raw = _trim_partial_xml_close(text[value_start:])
            else:
                raw = text[value_start:value_start + close.start()]
            return _xml_text_value(raw, complete=close is not None)
        cursor = value_start + (close.end() if close is not None else 0)


def _iter_xml_parameters(text: str) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = []
    cursor = 0
    while True:
        start = text.find("<parameter", cursor)
        if start == -1:
            break
        open_end = text.find(">", start)
        if open_end == -1:
            break
        tag_body = text[start + 1:open_end].strip()
        name = _parameter_name_from_open_tag(tag_body)
        value_start = open_end + 1
        close = re.search(r"</\s*parameter\s*>", text[value_start:], re.IGNORECASE)
        if close is None:
            cursor = value_start
            continue
        if name:
            raw = text[value_start:value_start + close.start()]
            params.append((name, _xml_text_value(raw, complete=True)))
        cursor = value_start + close.end()
    return params


def _parameter_name_from_open_tag(tag_body: str) -> str | None:
    if not tag_body.startswith("parameter"):
        return None
    tail = tag_body[len("parameter"):].strip()
    if tail.startswith("="):
        value = tail[1:].strip()
        if not value:
            return None
        if value[0] in ("'", '"'):
            quote = value[0]
            end = value.find(quote, 1)
            return html_unescape(value[1:end]) if end != -1 else None
        return html_unescape(re.split(r"\s+", value, maxsplit=1)[0].strip())
    match = re.search(
        r'\bname\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))',
        tail,
        re.DOTALL,
    )
    if match is None:
        return None
    return html_unescape(next(group for group in match.groups() if group is not None))


def _xml_text_value(raw: str, *, complete: bool) -> str:
    if complete:
        value = _strip_parameter_boundary_newlines(raw)
    else:
        # Mid-stream: the LEADING tag-boundary newline is already certain, so
        # strip it (a growing bash box must not start with a blank line); the
        # trailing one must stay — more value text may still arrive.
        value = raw
        if value.startswith("\r\n"):
            value = value[2:]
        elif value.startswith("\n"):
            value = value[1:]
    return html_unescape(value)


def _strip_parameter_boundary_newlines(value: str) -> str:
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    return value


def _trim_partial_xml_close(raw: str) -> str:
    lt = raw.rfind("<")
    if lt != -1 and "</parameter".startswith(raw[lt:]):
        return raw[:lt]
    return raw


def _parse_compact_xml_attributes(text: str) -> dict | None:
    stripped = text.strip()
    if not stripped.startswith("<") or stripped.startswith("</"):
        return None
    end = stripped.find(">")
    if end == -1:
        return None
    tag_body = stripped[1:end].strip().rstrip("/").strip()
    if not tag_body or tag_body.startswith(("tool_call", "function", "parameter")):
        return None
    tail = tag_body.split(None, 1)[1] if re.search(r"\s", tag_body) else ""
    if not tail:
        return {}
    attrs: dict[str, object] = {}
    for match in re.finditer(r'\b([A-Za-z_][\w.-]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\')', tail, re.DOTALL):
        value = html_unescape(match.group(2) if match.group(2) is not None else match.group(3))
        attrs[match.group(1)] = _json_or_string(value)
    return attrs


def _json_or_string(value: str) -> object:
    trimmed = value.strip()
    if trimmed.startswith(("{", "[")):
        try:
            return json.loads(trimmed)
        except json.JSONDecodeError:
            pass
    return value


def _parse_unquoted_content_object(text: str) -> dict | None:
    """Recover a common malformed tool call with raw escaped code as content.

    Some local model samples emit `{"file":"x","anchor":"1:aa","content":\n...}`
    where the code body is not wrapped as a JSON string, but still uses JSON
    escapes for newlines and quotes. Keep this repair narrow: the object prefix
    up through `content:` must parse after substituting an empty string, and the
    raw content must be the final field.
    """
    match = re.search(r'"content"\s*:\s*', text)
    if match is None or not text.startswith("{") or not text.endswith("}"):
        return None

    prefix = text[: match.end()]
    try:
        base = json.loads(prefix + '""}', strict=False)
    except json.JSONDecodeError:
        return None
    if not isinstance(base, dict):
        return None

    raw = text[match.end() : -1].strip()
    if raw.endswith('"'):
        raw = raw[:-1]
    try:
        content = json.loads(f'"{raw}"', strict=False)
    except json.JSONDecodeError:
        content = raw
    if not isinstance(content, str):
        return None
    base["content"] = content.lstrip("\r\n")
    return base


class StreamAccumulator:
    """Folds a stream of events into a single :class:`AssistantTurn`.

    Tool calls are keyed by their delta ``index`` because the ``id``/``name``
    only arrive on the begin delta while argument fragments may carry only the
    ``index``.
    """

    def __init__(self) -> None:
        self.content = ""
        self.reasoning = ""
        self._calls: dict[int, _PendingCall] = {}
        self.finish_reason = ""
        self.usage: dict | None = None
        self.timings: dict | None = None

    def add(self, event: StreamEvent) -> None:
        if isinstance(event, ContentDelta):
            self.content += event.text
        elif isinstance(event, ReasoningDelta):
            self.reasoning += event.text
        elif isinstance(event, ToolCallBegin):
            self._calls[event.index] = _PendingCall(id=event.id, name=event.name)
        elif isinstance(event, ToolCallArgsDelta):
            call = self._calls.get(event.index)
            if call is None:
                # Defensive: args before begin — create a stub.
                call = self._calls.setdefault(event.index, _PendingCall(id="", name=""))
            call.args_buffer += event.fragment
        elif isinstance(event, ToolCallName):
            call = self._calls.get(event.index)
            if call is not None:
                call.name = event.name
        elif isinstance(event, ToolCallFinal):
            call = self._calls.get(event.index)
            if call is not None:
                # Authoritative parsed JSON replaces the streamed raw XML. If
                # the stream dies before this chunk, the raw buffer still falls
                # through _parse_tool_args's XML recovery.
                call.args_buffer = event.arguments
        elif isinstance(event, ToolCallDemoted):
            # The block was not a tool call after all; its text arrives as
            # ordinary content/reasoning deltas.
            self._calls.pop(event.index, None)
        elif isinstance(event, Finish):
            self.finish_reason = event.reason
        elif isinstance(event, Usage):
            self.usage = event.usage
            self.timings = event.timings

    def finalize(self) -> AssistantTurn:
        calls: list[ToolCall] = []
        for index in sorted(self._calls):
            pending = self._calls[index]
            args = _parse_tool_args(pending.args_buffer)
            calls.append(
                ToolCall(
                    id=pending.id or f"call_{index}",
                    name=pending.name,
                    arguments=args,
                    index=index,
                )
            )
        return AssistantTurn(
            content=self.content,
            reasoning=self.reasoning,
            tool_calls=calls,
            finish_reason=self.finish_reason,
            usage=self.usage,
            timings=self.timings,
        )


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class Qw35Client:
    def __init__(
        self,
        base_url: str,
        timeout: float = 600.0,
        stream_tool_xml: bool = True,
        raw_sink: Callable[[str], None] | None = None,
        request_sink: Callable[[dict], None] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # Ask the server to stream tool-call bodies incrementally (raw XML
        # fragments + qw35_tool_call side-channel) so the TUI can show a call
        # growing live. Off = the buffered OpenAI-shape stream.
        self.stream_tool_xml = stream_tool_xml
        # Optional debug observers for every stream this client opens: the
        # exact outgoing payload and each raw SSE chunk. Used by the headless
        # harness to capture a full-fidelity trace; the interactive app leaves
        # them unset. Observers only — they never touch the request.
        self.raw_sink = raw_sink
        self.request_sink = request_sink
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        resp = await self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    async def props(self) -> dict:
        resp = await self._client.get("/props")
        resp.raise_for_status()
        return resp.json()

    async def decoder_ready(self) -> bool:
        try:
            return bool((await self.health()).get("decoder_ready"))
        except Exception:
            return False

    async def stream_chat(
        self,
        *,
        messages: list[dict],
        tools: list[dict] | None = None,
        raw_sink: Callable[[str], None] | None = None,
        request_sink: Callable[[dict], None] | None = None,
        **gen_params,
    ) -> AsyncIterator[StreamEvent]:
        """Open the chat-completions SSE stream and yield classified events.

        ``raw_sink`` receives each raw SSE ``data:`` payload before it is
        classified (the exact text the model emitted, malformed tool calls
        included); ``request_sink`` receives a copy of the full outgoing
        payload. Per-call sinks override the client-wide defaults set in the
        constructor. Sink failures are swallowed — observers can never break
        a stream.
        """
        body: dict = {
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **gen_params,
        }
        if self.stream_tool_xml:
            body["stream_tool_call_xml"] = True
        if tools:
            body["tools"] = tools

        effective_raw_sink = raw_sink if raw_sink is not None else self.raw_sink
        effective_request_sink = (
            request_sink if request_sink is not None else self.request_sink
        )
        if effective_request_sink is not None:
            try:
                effective_request_sink(
                    {
                        "messages": json.loads(json.dumps(messages)),
                        "tools": json.loads(json.dumps(tools or [])),
                        "params": json.loads(json.dumps(gen_params, default=str)),
                    }
                )
            except Exception:
                pass

        try:
            async with self._client.stream("POST", "/v1/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise _error_from_response(resp)
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if effective_raw_sink is not None:
                        try:
                            effective_raw_sink(data)
                        except Exception:
                            pass
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if "error" in chunk:
                        raise _error_from_payload(chunk.get("error") or {})
                    for event in _classify_chunk(chunk):
                        yield event
        except httpx.HTTPError as exc:
            raise Qw35Error("connection_error", str(exc), kind="network") from exc


def _error_from_response(resp: httpx.Response) -> Qw35Error:
    try:
        error = resp.json().get("error", {})
    except Exception:
        error = {}
    fallback = f"HTTP {resp.status_code}"
    return _error_from_payload(error, http_status=resp.status_code, fallback=fallback)


def _error_from_payload(
    payload: dict,
    *,
    http_status: int | None = None,
    fallback: str = "server error",
) -> Qw35Error:
    if not isinstance(payload, dict):
        payload = {}
    code = str(payload.get("code") or "http_error")
    message = str(payload.get("message") or fallback)
    kind = str(payload.get("type") or "")
    return Qw35Error(code, message, http_status=http_status, kind=kind)


def _classify_chunk(chunk: dict):
    """Turn one chat.completions.chunk into zero or more StreamEvents."""
    # Choice-less chunks carry our custom side-channels (prefill progress, final
    # usage). OpenAI clients ignore both the empty choices and the extra fields.
    choices = chunk.get("choices") or []
    if not choices:
        prefill = chunk.get("qw35_prefill")
        if prefill is not None:
            total = prefill.get("total", 0) or 0
            processed = prefill.get("processed", 0) or 0
            percent = prefill.get("percent")
            if percent is None:
                percent = (processed / total * 100.0) if total else 0.0
            session_ctx = prefill.get("session_ctx")
            yield PrefillProgress(
                percent=float(percent),
                processed=processed,
                total=total,
                session_ctx=int(session_ctx) if session_ctx else None,
            )
        tool_side = chunk.get("qw35_tool_call")
        if isinstance(tool_side, dict):
            kind = tool_side.get("event")
            index = tool_side.get("index", 0)
            if kind == "name":
                yield ToolCallName(index=index, name=str(tool_side.get("name", "")))
            elif kind == "final":
                yield ToolCallFinal(index=index, arguments=str(tool_side.get("arguments", "")))
            elif kind == "demoted":
                yield ToolCallDemoted(index=index)
        if chunk.get("usage") is not None:
            yield Usage(usage=chunk.get("usage") or {}, timings=chunk.get("qw35_timings") or {})
        return

    choice = choices[0]
    delta = choice.get("delta") or {}

    reasoning = delta.get("reasoning_content")
    if reasoning:
        yield ReasoningDelta(reasoning)

    content = delta.get("content")
    if content:
        yield ContentDelta(content)

    for tc in delta.get("tool_calls") or []:
        index = tc.get("index", 0)
        fn = tc.get("function") or {}
        if tc.get("id") or fn.get("name"):
            yield ToolCallBegin(index=index, id=tc.get("id", ""), name=fn.get("name", ""))
        args = fn.get("arguments")
        if args:
            yield ToolCallArgsDelta(index=index, fragment=args)

    finish = choice.get("finish_reason")
    if finish:
        yield Finish(finish)

    # Some servers attach usage to the same chunk as the finish.
    if chunk.get("usage") is not None:
        yield Usage(usage=chunk.get("usage") or {}, timings=chunk.get("qw35_timings") or {})
