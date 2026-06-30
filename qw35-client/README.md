# qw35-cli

Interactive Python chat client for the local `qw35` server.

<pre>
<span style="color:#a347ba">   _</span>   ..
 <span style="color:#a347ba">{</span><span style="color:#585858">o</span><span style="color:#d7af00">,</span><span style="color:#585858">ò</span><span style="color:#a347ba">}</span>
<span style="color:#a347ba"> /)_)</span>
<span style="color:#d7af00">  " "</span>
</pre>

Start the server first:

```sh
cargo run -- --port 8080
```

Run the CLI:

```sh
python -m qw35_client
```

On macOS, install `gnureadline` so multi-line pastes arrive as a single message
(the stdlib `readline` is libedit there, which submits a paste one line at a
time). Line editing and history work either way.

```sh
pip install -r requirements.txt   # gnureadline on macOS only
```

Options:

```sh
python -m qw35_client --base-url http://127.0.0.1:8080 --model qwen35-9b
python -m qw35_client --no-stream --max-tokens 256
```

Thinking is hidden by default and aligned with the `qw35-agent` (`qwowl35`):

```sh
python -m qw35_client --think auto            # defer to the server --mode preset (default)
python -m qw35_client --think on --reasoning-effort high
python -m qw35_client --think off             # disable thinking (Qwen3.5 thinks by default)
python -m qw35_client --no-show-thinking      # hide the reasoning (spinner only)
```

Reasoning is shown by default (dimmed, under a `thinking>` header); inline
`<think>` tags are always stripped from the visible answer and stored history.

Sampling (`--temperature`, `--top-p`) is owned by the server's `--mode` preset;
the client only sends an override when you pass the flag explicitly.

Commands inside chat:

- `/exit` or `/quit` exits.
- `/clear` clears the conversation history.
- `/system <text>` replaces the system prompt.
- `/history` prints the current transcript.
- `/think <auto|on|off>` switches thinking mode mid-session.
- `/show-thinking` toggles whether reasoning is shown or hidden.

This client is a one-shot / REPL chat — it has no tools or agent loop. For an
agentic coding experience (bash + file tools, tool-call loop, mascot) use the
`qw35-agent` (`qwowl35`).
