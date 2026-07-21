# qw35.app packaging

Builds a double-clickable macOS app — the Rust `qw35` server, the Python
`qwowl35` agent, and a first-run GUI that downloads the GGUF models with
explicit consent and visible progress — and wraps it in a distributable DMG.

```sh
make dmg          # → dist/qw35-<version>-arm64.dmg  (runs everything below)
make app          # just the .app bundle (PyInstaller + ad-hoc codesign)
make icns         # just packaging/qw35.icns from assets/app_icon.png
make packaging-venv
```

Requires: Xcode CLT (codesign/iconutil/hdiutil), python3, and a completed
`make release` (the `app` target depends on it). Apple silicon only, macOS 14+.

## What the app does

Double-clicking `qw35.app` opens a native window (pywebview) with a setup
page:

1. **Check** — looks for the models in `~/Library/Application Support/qw35/gguf`.
2. **Consent** — if missing, shows each file, its real size (HTTP HEAD),
   the free-disk margin, and rough time estimates; nothing downloads until
   the user clicks Download.
3. **Download** — per-file progress bar, live speed, ETA. Interruptions are
   safe: partials are kept as `<name>.part` and resumed with HTTP ranges
   (same convention as `download_model.sh`). The reranker is optional: if
   its Hugging Face repo refuses anonymous access, it is skipped with a
   visible notice and the agent's web rerank falls back to BM25.
4. **Start** — launches the bundled `qw35` server on a free port, waits for
   `/health` to report `decoder_ready`, starts textual-serve for the agent,
   and swaps the same window over to the agent UI.

Closing the window (or Cmd-Q, or SIGTERM) reaps the whole process tree.
Logs land in `~/Library/Logs/qw35/` (`launcher.log`, `server.log`,
`agent.log`).

## Architecture notes

The frozen executable is a multi-mode dispatcher (`launcher/qw35_launcher.py`):

```
qw35                                  GUI launcher (Finder entry point)
qw35 --qw35-dispatch agent-serve …    textual-serve host (child of the GUI)
qw35 --qw35-dispatch agent-tui …      one qwowl35 instance per browser tab
```

This exists because `qwowl35/__main__.py` re-invokes
`sys.executable __main__.py` for its `--ui gui` children, which cannot work
from a PyInstaller bundle. The launcher never modifies `qwowl35/` — the
package ships as data (preserving its sys.path-insertion design and
`webui/` assets) and all frozen-specific logic lives here.

Because `qwowl35` ships as data, PyInstaller cannot trace its imports;
every runtime dependency is force-collected in `qw35.spec` (`collect_all`).
If a frozen agent tab dies on an import error, run the dispatch mode in a
terminal to see it immediately:

```sh
dist/qw35.app/Contents/MacOS/qw35 --qw35-dispatch agent-tui --base-url http://127.0.0.1:8080
```

Not bundled: `jedi-language-server` (multilspy launches it from PATH; inside
the app the LSP check degrades to tree-sitter diagnostics — install it with
`pipx install jedi-language-server` to restore LSP for terminal use).

## Testing

```sh
# dev-mode launcher (no freeze), isolated model dir:
QW35_APP_SUPPORT_DIR=$(mktemp -d)/qw35 packaging/.venv/bin/python packaging/launcher/qw35_launcher.py

# fresh-user simulation against the built app:
QW35_APP_SUPPORT_DIR=$(mktemp -d)/qw35 dist/qw35.app/Contents/MacOS/qw35

# resume: quit mid-download, relaunch — consent shows "Resume (… already on disk)".
# reap: close the window, then `pgrep -f qw35` must print nothing.
```

## Gatekeeper (no Developer ID)

The app is ad-hoc signed, so the first open triggers a warning. The DMG
ships `How to open (unsigned app).txt` covering macOS 14 (right-click →
Open) and macOS 15+ (System Settings → Privacy & Security → "Open Anyway"),
plus `xattr -dr com.apple.quarantine /Applications/qw35.app`.
