# qwowl35

A terminal coding agent for the local **qw35-server**. It streams the model's
response and calls tools in a loop, with an animated owl mascot pinned
top-left that reflects the agent's live state.

The user picks the operating mode BEFORE sending a prompt — a Vim-style
inverted label in the bottom-left corner shows the active mode, **Shift+Tab**
(or the `/mode` command) cycles it, and it locks while the model is
generating. Every new conversation starts in NORMAL.

- **NORMAL (default).** The freestyle executor: a safety-aware
  `run_shell_command` plus `edit`, where `edit` delegates the change to a
  dedicated **editor** sub-agent (hashline `edit`/`insert`/`delete`, no
  bash) — the label flips to **INSERT** while the editor runs.
- **PLAN.** The **planner** (`plan`, `ask_user_question`, `explore`) designs
  an ordered todo plan behind an approval modal. It cannot read code itself:
  its `explore` tool spawns a stateless read-only **explorer** sub-agent
  (`list_directory`/`glob`/`grep_search`/`inspect_file` + a restricted
  shell) that searches freely and reports back through one `resume` call —
  only that findings summary returns to the planner (label: **VISUAL** while
  it runs). After approval, todos execute one at a time on ONE persistent
  executor conversation (each todo appends a slim directive, so consecutive
  executors inherit the full context of their predecessors and the server's
  checkpoint stack prefills only the new message), with a bounded planner
  review ping after each todo (label: NORMAL while executors run, PLAN
  during reviews).
- **WEB.** A web agent restricted to `search_engine` + `web_fetch`: finds
  and fetches what the request needs and answers with findings + URLs.
- **CHAT.** A lightweight, tool-less conversational agent on one persistent
  conversation.

Every agent is segregated: it starts from a fresh context with its OWN
system prompt, advertises only its OWN tools, and receives exactly the data
handed over to it. The editor and explorer sub-agents run on the server's
scratch GPU session (`qw35_session: scratch`), the planner persists on its
own `plan` session, so nothing disturbs a stage in progress. Agents live
one-per-module under `agents/`; the mode dispatch is `orchestrator.py`.

Every conversation is persisted as a session under the user cache dir
(`sessions/` package): one `sessions/{session_hash}/turns/{NNNN}/` directory
per turn holding the stage artifacts (exploration reports, plan, per-task
results), a `meta.json` (goal, mode, outcome, session-path tallies), and a
`transcript.jsonl` with the raw model I/O — the exact requests sent, the raw
SSE chunks received (malformed tool-call XML included), parsed assistant
turns, and tool results — the on-disk source of truth for debugging what the
TUI renders. `/sessions` lists past sessions and restores one: the display
replays, the turn log and CHAT conversation rehydrate verbatim, and the
server re-primes its KV cache with a normal full prefill on the next
request. Stale sessions are garbage-collected by age and count at app
startup/exit.

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
python -m qwowl35 --ui webgui             # same UI at http://localhost:8000 (pip install textual-serve)
python -m qwowl35 --ui gui                # …wrapped in a desktop window (pip install pywebview)
```

`--ui` picks the render target without changing the UI itself: `webgui` serves
the app through textual-serve, which relaunches this same entry point (all
other flags forwarded) as one subprocess per browser tab and streams its render
output — colors, layout, mouse, and themes are identical to the terminal.
`gui` starts that server on a free port (`--ui-port` overrides either mode) and
wraps the page in a pywebview desktop window, falling back to the default
browser when pywebview is missing; closing the window stops the server.

The web page renders in **Mononoki Nerd Font Mono** (SIL OFL 1.1), vendored as
woff2 under `webui/fonts/` and served locally — no Google Fonts request. The
stock textual-serve page hardcodes "Roboto Mono" in its JS bundle, so
`webui/app_index.html` (a vendored template with the remote font link removed)
re-binds that family name to the Mononoki files via `@font-face`. To swap the
font, drop replacement woff2 files in `webui/fonts/` and update the
`@font-face` `src` names in the template.

Configuration is CLI-only — no environment-variable overrides. The bash analyzer
gains an AST mode if the optional `tree-sitter-language-pack` is installed, but
falls back to substring matching without it.

Read/edit results carry a validation block with two layers. The primary layer
(`tools/lsp/`, on by default, `--no-lsp` to disable) runs real language servers
through the optional `multilspy` package for semantic diagnostics — unresolved
symbols, type errors — and needs the per-language binary on `PATH`
(`jedi-language-server` for Python — syntax-level only, jedi is not a type
checker — `rust-analyzer`, `gopls`, …). Whenever LSP cannot answer (package or
binary missing, server still warming up, diagnostics timeout, unsupported
language), the check silently falls back to the second layer: the tree-sitter
syntax checker, which needs only `tree-sitter-language-pack`. Without either
package the block simply disappears; edits never fail because validation is
unavailable.

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
`Shift+Tab` (or `/mode [normal|plan|web|chat]`) cycle mode · `Ctrl+C` quit.
Approval: `1/2/3` or arrows + `Enter`, `Tab` to write an alternative.

## Headless runners (`debug/`)

Non-interactive drivers that exercise the *same* agent/client/tools as the TUI,
inside an isolated scratch dir, and dump a full transcript (`transcript.jsonl` +
`messages.json`) for debugging why a run behaved as it did. The server must be
running.

```bash
# one task, one turn (defaults to the bundled benchmark/cal_task.md)
python qwowl35/debug/headless.py --task benchmark/cal_task.md --timeout 300
python qwowl35/debug/headless.py --prompt "write hello.py" --restricted-bash
python qwowl35/debug/headless.py --prompt "refactor foo" --mode plan   # planner pipeline, scripted approvals

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
| `tools/lsp/`, `tools/syntax/` | LSP semantic diagnostics (primary edit validation) · tree-sitter checker (fallback) + the `validate.py` router |
| `tools_registry.py` / `prompts.py` / `config.py` | Tool schemas + dispatch · system prompt · defaults |
| `widgets/` | `chat/` (transcript: `chat_view`, `tool_block`, `thinking_block`, `card`, `renderers/`), `prompt_input`, `approval_modal`, `status_bar`, `mascot` |
| `debug/` | Headless runners (`headless.py`, `headless_steps.py`) |
| `tests/` | All tests (incl. `hashline_parity_test.py`, an optional upstream audit) |
