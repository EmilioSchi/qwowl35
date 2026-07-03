# qwowl35

A minimal terminal coding agent for the local **qw35-server**. It streams the
model's response and calls tools in a loop — a safety-aware **bash** tool and
anchor-backed **file** read/edit tools — with an animated owl mascot pinned
top-left that reflects the agent's live state.

```
   _   z
 {-,-}        ← the owl stays put while the chat scrolls
 /)_)
  " "
```

## Install & run

```bash
pip install -r ../requirements.txt       # textual, rich, httpx, xxhash (at the qw35-agent root)
python -m qwowl35                         # qw35-server must listen on 127.0.0.1:8080
python -m qwowl35 --base-url http://127.0.0.1:8080 --reasoning-effort xhigh
```

Configuration is CLI-only — no environment-variable overrides. The bash analyzer
gains an AST mode if the optional `tree-sitter-language-pack` is installed, but
falls back to substring matching without it.

The `/theme` picker is the one exception: the last committed theme is remembered
across launches (written to `theme.json` in the OS config dir). Set
`QWOWL35_THEME=<name>` (or `<name>:<mode>`, e.g. `tokyonight:light`) to override
that saved choice for a session.

## Design

- **Tools upfront.** The system prompt describes `bash` and the anchored file
  tools directly (no skill calling); the server parses Qwen3 XML tool calls back
  into structured calls.
- **Anchored edits.** `read_file` returns lines as `line:hash|content`; the
  mutators (`edit/insert/delete_lines_if_file_exists`) operate on anchored
  lines/ranges, never whole files. `bash` handles search, tests, and file
  create/delete.
- **Mascot state machine.** idle→WAITING, submit→WAKEUP, prefill→PREFILL,
  reasoning→THINKING, generating→INFERENCE, bash→BASH, file→EDIT, done→OK.
  Real prefill % comes from the server's `qw35_prefill` SSE chunks (OpenAI
  clients ignore them; falls back to an animated bar).
- **Readline-style input.** Auto-growing multiline prompt; Enter submits,
  Shift/Alt+Enter newline; Up/Down recall history in `~/.qwowl35/history`; large
  pastes collapse to `[paste #N +M lines]`.
- **Streaming, expandable tools.** Tool boxes grow as args stream, tint by state,
  truncate large results; **Ctrl+O** expands all.
- **Gated risky bash.** Keyboard approval: `1. Accept · 2. Deny · 3. Write to do
  differently (Tab)`.

## Keys

`Enter` send · `Shift+Enter` newline · `Up/Down` history · `Ctrl+O` expand tools ·
`Ctrl+C` quit. Approval: `1/2/3` or arrows + `Enter`, `Tab` to write an alternative.

## Headless runners (`debug/`)

Non-interactive drivers that exercise the *same* agent/client/tools as the TUI,
inside an isolated scratch dir, and dump a full transcript (`transcript.jsonl` +
`messages.json`) for debugging why a run behaved as it did. The server must be
running.

```bash
# one task, one turn (defaults to the bundled benchmark/cal_task.md)
python qwowl35/debug/headless.py --task benchmark/cal_task.md --timeout 300
python qwowl35/debug/headless.py --prompt "write hello.py" --restricted-bash

# several steps through ONE persistent session — tests incremental editing
python qwowl35/debug/headless_steps.py \
    --steps-file benchmark/solve_real_root_steps.md \
    --target solve_real_root.py --timeout 360
```

Pass `--help` to either for the full flag list.

## Layout

| Path | Role |
|------|------|
| `app.py` / `agent.py` / `client.py` | TUI app · agent loop + mascot states · httpx SSE client |
| `tools/bash/`, `tools/files/` | Bash tool · anchored file tools (`adapter.py` + `hashline/` core) |
| `tools_registry.py` / `prompts.py` / `config.py` | Tool schemas + dispatch · system prompt · defaults |
| `widgets/` | `mascot_widget`, `chat_log`, `prompt_input`, `approval`, `status_panel` |
| `debug/` | Headless runners (`headless.py`, `headless_steps.py`) |
| `tests/` | All tests (incl. `hashline_parity_test.py`, an optional upstream audit) |
