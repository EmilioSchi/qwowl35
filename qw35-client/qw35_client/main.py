from __future__ import annotations

import argparse
import atexit
import itertools
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

# Line editing, history, and (critically) multi-line paste handling come from
# readline. macOS ships libedit under the `readline` name, which submits a paste
# one line per Enter; the `gnureadline` wheel provides real GNU readline, whose
# bracketed-paste support returns a whole paste as a single input(). Prefer it,
# fall back to the stdlib module, then to no line editing at all.
try:
    import gnureadline as readline
except ImportError:  # pragma: no cover - gnureadline not installed
    try:
        import readline
    except ImportError:  # pragma: no cover - not all platforms ship readline
        readline = None  # type: ignore[assignment]


DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL = "qwen3.5-9b"

# Input history.
HISTORY_FILE = os.path.expanduser("~/.qw35_client_history")
HISTORY_LENGTH = 1000

# Thinking control, mirroring qw35-agent (qwowl35) and llama.cpp's reasoning flag:
#   auto → send nothing; defer to the server's `--mode` default
#   on   → request thinking (enable_thinking=true) + optional reasoning_effort
#   off  → explicitly disable thinking (enable_thinking=false). Qwen3.5 "thinks
#          by default", so the flag must be sent to turn it off (qwen-code #4505).
# By default the reasoning itself is hidden; pass --show-thinking to reveal it.
ALLOWED_THINK = ("auto", "on", "off")
ALLOWED_EFFORTS = ("low", "medium", "high", "xhigh")


@dataclass
class ChatConfig:
    base_url: str
    model: str
    stream: bool
    show_stats: bool
    max_tokens: int | None
    # Sampling is owned by the server's --mode preset; only sent when the user
    # explicitly overrides it (None = defer to the server, like the TUI).
    temperature: float | None
    top_p: float | None
    timeout: float
    system: str | None
    # Thinking: principal choice + optional effort override; hidden unless asked.
    think: str
    reasoning_effort: str | None
    show_thinking: bool

    def gen_params(self) -> dict[str, Any]:
        """Thinking/token request fields, mirroring qwowl35's Config.gen_params."""
        params: dict[str, Any] = {}
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.think == "off":
            # Explicit disable. Omit reasoning_effort so it can't re-enable.
            params["enable_thinking"] = False
        elif self.think == "on":
            params["enable_thinking"] = True
            params["preserve_thinking"] = True
            if self.reasoning_effort is not None:
                params["reasoning_effort"] = self.reasoning_effort
        # "auto": send nothing; defer entirely to the server's --mode default.
        return params


@dataclass
class ChatResult:
    content: str
    usage: dict[str, Any] | None
    timings: dict[str, Any] | None
    reasoning: str = ""


class ApiError(RuntimeError):
    pass


class ThinkFilter:
    """Strip ``<think>...</think>`` spans from a streamed content sequence.

    The server normally streams reasoning on the separate ``reasoning_content``
    channel, but some ``--mode`` presets fold it inline into ``content``. This
    filter removes those spans even when a tag is split across SSE chunks, so the
    visible answer (and stored history) never contains the thinking. The removed
    text is accumulated in :attr:`thought` so callers can show it on request.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._in_think = False
        self._pending = ""  # a partial tag held back at a chunk boundary
        self.thought = ""

    def feed(self, text: str) -> str:
        data = self._pending + text
        self._pending = ""
        out: list[str] = []
        i = 0
        while i < len(data):
            if not self._in_think:
                idx = data.find(self._OPEN, i)
                if idx == -1:
                    safe = self._hold(data, i, self._OPEN)
                    out.append(data[i:safe])
                    self._pending = data[safe:]
                    break
                out.append(data[i:idx])
                i = idx + len(self._OPEN)
                self._in_think = True
            else:
                idx = data.find(self._CLOSE, i)
                if idx == -1:
                    safe = self._hold(data, i, self._CLOSE)
                    self.thought += data[i:safe]
                    self._pending = data[safe:]
                    break
                self.thought += data[i:idx]
                i = idx + len(self._CLOSE)
                self._in_think = False
        return "".join(out)

    def finish(self) -> str:
        """Flush any held-back text once the stream ends."""
        pending, self._pending = self._pending, ""
        if self._in_think:
            self.thought += pending
            return ""
        return pending

    @staticmethod
    def _hold(data: str, start: int, tag: str) -> int:
        """Index up to which it is safe to emit; the suffix may be a partial tag."""
        for k in range(min(len(tag) - 1, len(data) - start), 0, -1):
            if data.endswith(tag[:k]):
                return len(data) - k
        return len(data)


class ThinkingSpinner:
    """A transient stderr spinner shown while reasoning is hidden.

    Writes only to a TTY so piped/redirected output stays clean, and erases
    itself before the answer is printed.
    """

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self) -> None:
        self._frames = itertools.cycle(self._FRAMES)
        self._active = False

    @property
    def _enabled(self) -> bool:
        return sys.stderr.isatty()

    def tick(self) -> None:
        if not self._enabled:
            return
        self._active = True
        sys.stderr.write(f"\r\x1b[2m{next(self._frames)} thinking…\x1b[0m")
        sys.stderr.flush()

    def clear(self) -> None:
        if self._active and self._enabled:
            sys.stderr.write("\r\x1b[K")
            sys.stderr.flush()
        self._active = False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ChatConfig(
        base_url=args.base_url.rstrip("/"),
        model=args.model,
        stream=not args.no_stream,
        show_stats=not args.no_stats,
        max_tokens=args.max_tokens if args.max_tokens > 0 else None,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        system=args.system,
        think=args.think,
        reasoning_effort=args.reasoning_effort,
        show_thinking=args.show_thinking,
    )

    try:
        check_server(config)
    except ApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    run_repl(config)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qw35-cli",
        description="Interactive chat client for a local qw35 server.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Do not print the token/throughput summary after each answer.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Cap completion tokens; a runaway generation otherwise runs until the"
        " context is exhausted. Use 0 for no cap. Default: 8192",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override sampling temperature. Default: defer to the server --mode preset.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Override nucleus sampling top_p. Default: defer to the server --mode preset.",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--system", default=None)
    parser.add_argument(
        "--think",
        choices=ALLOWED_THINK,
        default="auto",
        help="thinking mode: auto defers to the server --mode default, on requests "
        "thinking, off disables it (default auto). Same semantics as qw35-agent.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=ALLOWED_EFFORTS,
        default=None,
        help="optional thinking budget when --think on: low/medium/high cap the "
        "reasoning budget, xhigh is uncapped (only sent when given)",
    )
    parser.add_argument(
        "--show-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the model's reasoning (dimmed) under a thinking> header (default); "
        "use --no-show-thinking to hide it and show only a spinner",
    )
    return parser.parse_args(argv)


def check_server(config: ChatConfig) -> None:
    try:
        data = request_json(config, "GET", "/v1/models", None)
    except ApiError as exc:
        raise ApiError(
            f"cannot reach qw35 server at {config.base_url}. "
            "Start it with `cargo run -- --port 8080` from the project root."
        ) from exc

    models = [item.get("id") for item in data.get("data", [])]
    if config.model not in models:
        available = ", ".join(str(model) for model in models) or "none"
        raise ApiError(f"model {config.model!r} is not available; server reports: {available}")


def init_readline() -> None:
    """Enable readline line-editing, persistent history, and bracketed paste.

    All best-effort: any piece the platform's readline can't do is skipped. With
    gnureadline (GNU readline), bracketed paste makes a multi-line paste arrive
    as a single input() rather than one message per line.
    """
    if readline is None:
        return
    try:
        readline.parse_and_bind("set enable-bracketed-paste on")
    except Exception:
        pass
    try:
        readline.read_history_file(HISTORY_FILE)
    except (FileNotFoundError, OSError):
        pass
    try:
        readline.set_history_length(HISTORY_LENGTH)
    except Exception:
        pass
    atexit.register(_save_history)


def _save_history() -> None:
    if readline is None:
        return
    try:
        readline.write_history_file(HISTORY_FILE)
    except OSError:
        pass


def run_repl(config: ChatConfig) -> None:
    init_readline()
    messages: list[dict[str, str]] = []
    if config.system:
        messages.append({"role": "system", "content": config.system})

    print(f"qw35-cli connected to {config.base_url} ({config.model})")
    print(f"thinking: {config.think} (hidden)" if not config.show_thinking else f"thinking: {config.think} (shown)")
    print(
        "Type /exit to quit, /clear to reset, /system <text> to set a system prompt, "
        "/think <auto|on|off> to switch thinking, /show-thinking to toggle reasoning."
    )

    while True:
        try:
            user_text = input("\nuser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user_text:
            continue

        if handle_command(user_text, messages, config):
            if user_text in {"/exit", "/quit"}:
                return
            continue

        messages.append({"role": "user", "content": user_text})

        try:
            if config.stream:
                result = chat_stream(config, messages)
            else:
                result = chat_once(config, messages)
            print()
            if config.show_stats:
                print_stats(result)
        except ApiError as exc:
            print(f"\nerror: {exc}", file=sys.stderr)
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": result.content})


def handle_command(text: str, messages: list[dict[str, str]], config: ChatConfig) -> bool:
    if text in {"/exit", "/quit"}:
        return True

    if text == "/show-thinking":
        config.show_thinking = not config.show_thinking
        print(f"thinking is now {'shown' if config.show_thinking else 'hidden'}")
        return True

    if text.startswith("/think"):
        value = text.removeprefix("/think").strip().lower()
        if value in ALLOWED_THINK:
            config.think = value
            print(f"thinking mode set to {value}")
        elif not value:
            print(f"thinking mode is {config.think} (usage: /think <auto|on|off>)")
        else:
            print(f"unknown thinking mode: {value!r} (expected auto, on, or off)")
        return True

    if text == "/clear":
        system = next((msg for msg in messages if msg["role"] == "system"), None)
        messages.clear()
        if system:
            messages.append(system)
        print("history cleared")
        return True

    if text == "/history":
        for msg in messages:
            print(f"{msg['role']}: {msg['content']}")
        return True

    if text.startswith("/system "):
        prompt = text.removeprefix("/system ").strip()
        messages[:] = [msg for msg in messages if msg["role"] != "system"]
        if prompt:
            messages.insert(0, {"role": "system", "content": prompt})
            print("system prompt set")
        else:
            print("system prompt cleared")
        return True

    if text.startswith("/"):
        print(f"unknown command: {text}")
        return True

    return False


def chat_once(config: ChatConfig, messages: list[dict[str, str]]) -> ChatResult:
    payload = chat_payload(config, messages, stream=False)
    data = request_json(config, "POST", "/v1/chat/completions", payload)
    try:
        message = data["choices"][0]["message"]
        raw_content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise ApiError(f"unexpected response shape: {data}") from exc

    # Strip any inline <think>...</think> the server folded into content, and keep
    # the reasoning out of the visible answer / stored history.
    filtered = ThinkFilter()
    content = filtered.feed(raw_content) + filtered.finish()
    reasoning = (reasoning + filtered.thought).strip()

    if config.show_thinking and reasoning:
        print_thinking(reasoning)
    print("assistant> " + content, end="", flush=True)

    return ChatResult(
        content=content,
        usage=object_or_none(data.get("usage")),
        timings=object_or_none(data.get("qw35_timings")),
        reasoning=reasoning,
    )


def chat_stream(config: ChatConfig, messages: list[dict[str, str]]) -> ChatResult:
    payload = chat_payload(config, messages, stream=True)
    request = build_request(config, "POST", "/v1/chat/completions", payload)
    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    usage: dict[str, Any] | None = None
    timings: dict[str, Any] | None = None
    think_filter = ThinkFilter()
    spinner = ThinkingSpinner()
    printed_prefix = False
    thinking_header = False

    def emit_content(text: str) -> None:
        nonlocal printed_prefix
        if not text:
            return
        if not printed_prefix:
            spinner.clear()
            print("assistant> ", end="", flush=True)
            printed_prefix = True
        chunks.append(text)
        print(text, end="", flush=True)

    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            for data in iter_sse_data(response):
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise ApiError(chunk["error"].get("message", str(chunk["error"])))
                usage = object_or_none(chunk.get("usage")) or usage
                timings = object_or_none(chunk.get("qw35_timings")) or timings
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        reasoning_chunks.append(reasoning)
                        if config.show_thinking:
                            if not thinking_header:
                                print("\x1b[2mthinking> ", end="", flush=True)
                                thinking_header = True
                            print(reasoning, end="", flush=True)
                        else:
                            spinner.tick()
                    content = delta.get("content")
                    if content:
                        if thinking_header:
                            print("\x1b[0m\n", end="", flush=True)
                            thinking_header = False
                        emit_content(think_filter.feed(content))
    except urllib.error.HTTPError as exc:
        raise http_error(exc) from exc
    except urllib.error.URLError as exc:
        raise ApiError(str(exc.reason)) from exc
    except json.JSONDecodeError as exc:
        raise ApiError(f"invalid streaming JSON: {exc}") from exc

    if thinking_header:
        print("\x1b[0m", end="", flush=True)
    emit_content(think_filter.finish())
    spinner.clear()
    if not printed_prefix:
        print("assistant> ", end="", flush=True)

    reasoning = ("".join(reasoning_chunks) + think_filter.thought).strip()
    return ChatResult(
        content="".join(chunks), usage=usage, timings=timings, reasoning=reasoning
    )


def print_thinking(reasoning: str) -> None:
    """Print reasoning dimmed, under a header, when --show-thinking is on."""
    print("\x1b[2mthinking> " + reasoning + "\x1b[0m")


def print_stats(result: ChatResult) -> None:
    timings = result.timings or {}
    usage = result.usage or {}

    prompt_tokens = int_or_none(timings.get("prompt_eval_count")) or int_or_none(
        usage.get("prompt_tokens")
    )
    eval_tokens = int_or_none(timings.get("eval_count")) or int_or_none(
        usage.get("completion_tokens")
    )
    prompt_tps = float_or_none(timings.get("prompt_eval_tps"))
    eval_tps = float_or_none(timings.get("eval_tps"))
    if prompt_tokens is None and eval_tokens is None:
        return

    print(
        "[ "
        f"Prompt: {format_count(prompt_tokens)} tok, {format_tps(prompt_tps)} t/s | "
        f"Generation: {format_count(eval_tokens)} tok, {format_tps(eval_tps)} t/s"
        " ]"
    )


def format_count(value: int | None) -> str:
    return str(value) if value is not None else "?"


def format_tps(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "?"


def object_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def chat_payload(
    config: ChatConfig,
    messages: list[dict[str, str]],
    *,
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "stream": stream,
        "stream_options": {"include_usage": True},
    }
    payload.update(config.gen_params())
    return payload


def request_json(
    config: ChatConfig,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    request = build_request(config, method, path, payload)
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise http_error(exc) from exc
    except urllib.error.URLError as exc:
        raise ApiError(str(exc.reason)) from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ApiError(f"invalid JSON response: {exc}") from exc


def build_request(
    config: ChatConfig,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
) -> urllib.request.Request:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(
        f"{config.base_url}{path}",
        data=body,
        headers=headers,
        method=method,
    )


def iter_sse_data(response: Any) -> Iterable[str]:
    event_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            data = "\n".join(event_lines)
            event_lines.clear()
            if data:
                yield data
            continue
        if line.startswith("data:"):
            event_lines.append(line.removeprefix("data:").lstrip())

    if event_lines:
        yield "\n".join(event_lines)


def http_error(exc: urllib.error.HTTPError) -> ApiError:
    body = exc.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return ApiError(f"HTTP {exc.code}: {body or exc.reason}")

    error = data.get("error")
    if isinstance(error, dict):
        return ApiError(f"HTTP {exc.code}: {error.get('message', error)}")
    return ApiError(f"HTTP {exc.code}: {data}")
