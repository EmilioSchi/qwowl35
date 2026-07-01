# Anchor-grammar token-efficiency decision

Measured with `tools/token_report.py` against the model's own vocab+merges
(`Qwen3.5-9B-Q4_K_M.gguf`, gpt2/qwen35, vocab 248,320). The Python BPE port is
validated 50/50 against `vocab-research/camelcase_single_after_function.tsv` and
the README's hard facts (beginTransaction/getName single; getContext/getRow not).

## 1. Tool rename `read` → `beginTransaction`
- `<function=read>` = **4 tokens** (`< | function | =read | >` — `=read` fuses).
- `<function=beginTransaction>` = **5 tokens** (`< | function | = | beginTransaction | >`).
- Finding: `beginTransaction` *is* a single token for the name (the original
  observation is correct), but the whole call is **+1 token vs `read`**, because
  `=read` fuses into one token. **The rename is a behavioral/semantics choice
  (edit-intent + dropping the `anchor`/`context` view params), not a token win.**
  Adopted anyway per the approved plan; the +1/call is negligible vs read-output.

## 2. Anchor encoding — ADOPTED: `{line}{hh}|{content}` (drop the `:`)
Per-line prefix cost over a real 1,218-line corpus (tool_calling.py + anchor.py +
output.py), total tokens vs the `N:hh|C` baseline (20,183):

| scheme                       | tokens | delta |
|------------------------------|-------:|------:|
| `N word C`  (word + spaces)  | 17,028 | −15.6% |
| `Nhh C`     (drop `:` + sp)  | 18,006 | −10.8% |
| `Nword|C`   (word, keep `|`) | 18,107 | −10.3% |
| `N hh C`    (hex + spaces)   | 18,864 | −6.5% |
| `N:hh C`    (space join)     | 18,983 | −5.9% |
| **`Nhh|C`  (drop `:`, keep `|`) — ADOPTED** | **19,206** | **−4.8%** |
| `N:word|C`  (word, colon)    | 19,274 | −4.5% |
| `NxHH|C`    (x, UPPER)       | 20,064 | −0.6% |
| `N:hh|C`    (baseline)       | 20,183 |  0.0% |
| `N:HH|C`    (UPPER hex)      | 20,183 |  0.0% |

Decision rationale:
- **Drop the `:`** — a clean −4.8% with ZERO change to the cross-check (same low
  byte, same 256-bucket collision profile) and a trivial, unambiguous parse
  (hash = fixed last 2 chars, line = the leading digits). Range join stays `..`
  → `12af..189c`.
- **Keep the `|`** content delimiter even though space-join saves a further ~6%:
  the read output is display-only, and the crisp `|` boundary lets the model
  reproduce exact leading indentation when it echoes replacement content — load
  bearing for a Python editor. Worth the tokens.
- **Reject `x`/UPPER hex** — measured as ~0% (the user's `{line}x{HASH}` idea is
  not efficient here).
- **Word-table (−10.3% keeping `|`, −15.6% with space-join): measured, NOT
  adopted.** It roughly doubles the saving but costs a pinned 256-word table
  coupled to the tokenizer version, and rendering the hash as a real-looking word
  (`12cat|…`) risks the model treating it as content or copying it less reliably
  than an opaque `af`. Reproducible via the harness if we later want to push
  further.

## 3. Concept name — ADOPTED: `id` (replaces param `anchor` + the word "hash")
`<parameter=NAME>` cost: 4-token (fuse with `=`): `id key line name node num path
pos row val`. 5-token: `anchor at hash loc ref tag mark span …`.
- Chose **`id`** — cheapest tier (4 tokens), opaque (the model copies `12af`
  whole, no bias toward passing a bare line number the way `line` would), and one
  canonical word for the concept. Guidance calls the `12af` token the "line id".
- Per-call only, so this is a small win; the name is chosen mostly for a single
  clean vocabulary (no more `anchor` vs `hash` split).

## Confirmed on live tool output (post-implementation)
- `beginTransaction` of the real 357-line `anchor.py`: **5526 → 5230 tokens
  (−5.4%)** read-output.
- One edit-call id (`<parameter=id>12af</parameter>` vs
  `<parameter=anchor>12:af</parameter>`): **12 → 10 tokens** (shorter name +
  dropped colon).
- `<function=read>` is 4 tokens, `<function=beginTransaction>` is 5 — the rename
  is **+1 token/call** (behavioral choice, not a token win). Read-output savings
  dominate total cost.
- All 18 test files green; `verify_tool_template.py` confirms the tool-call
  wrapper the model was trained on is unchanged.

## 4. Aliases removed
`file`/`path` and `content`/`text` silent fallbacks dropped — one canonical
parameter name per concept.
