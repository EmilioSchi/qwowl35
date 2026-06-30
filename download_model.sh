#!/bin/sh
set -e

# qw35 model downloader.
#
# Fetches GGUFs from Hugging Face into ./.gguf/. The canonical unified model the
# server loads by default — Qwowl3.5-9B.gguf: the base GGUF with its FFN baked as
# GF4 (type-id 100) and an AWQ fold in the norms — can either be downloaded
# directly or cooked locally from the base GGUF.
# The base GGUF is kept as the cook input and the quality-comparison reference.

REPO="unsloth/Qwen3.5-9B-GGUF"
MODEL_FILE="Qwen3.5-9B-Q4_K_M.gguf"
# Hugging Face repo hosting the prebuilt canonical unified model.
CANON_REPO="EmilioSchi/Qwowl3.5-9B-GGUF"
# Canonical unified model (cooked with the AWQ-GF4 approach, the winner of the
# awq-gf4 vs gf4 quality comparison) and
# the AWQ activation stats it folds in (captured from the base model).
CANON_FILE="Qwowl3.5-9B.gguf"
ACT_STATS_FILE="act-stats.bin"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${QW35_GGUF_DIR:-"$ROOT/.gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}
COOK="$ROOT/tools/cook_qw35_awq_gf4.py"

usage() {
    cat <<EOF
qw35 model downloader

Usage:
  ./download_model.sh [download|model|cook|all] [--token TOKEN]

Targets:

  download  (default) Download the prebuilt canonical unified model
            $CANON_FILE from $CANON_REPO into the model
            directory. This is the file the server loads by default — no
            cooking needed.

  model     Download the base GGUF
            $MODEL_FILE (~5.3 GB) into the model directory. This is the
            cook input and the quality-comparison reference.

  cook      Cook the canonical unified model $CANON_FILE from the
            downloaded base GGUF using tools (FFN baked as GF4 + AWQ norm
            fold). An alternative to 'download'. Needs the AWQ activation stats
            $ACT_STATS_FILE; if absent, capture them first with:
              cargo test -p qw35-server --lib real_model_capture_activations -- --ignored
            CPU-heavy; takes several minutes. Requires python3 + numpy + gguf.

  all       Run model, then cook.

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF token
                 cache (~/.cache/huggingface/token) is used if present.

Environment:
  QW35_GGUF_DIR  Directory used for the base GGUF and cooked unified model.
                 Default: ./.gguf

After downloading, the default server command just works:
  ./target/release/qw35      (or: make run)
EOF
}

TARGET=download
case "${1:-}" in
    download|unified|model|cook|gf4|all) TARGET=$1; shift ;;
    -h|--help|help) usage; exit 0 ;;
    "" ) ;;
    --token) ;;  # no target given, options follow
    *) echo "Unknown target: $1" >&2; echo >&2; usage >&2; exit 1 ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            [ $# -gt 0 ] || { echo "Missing value after --token" >&2; exit 1; }
            TOKEN=$1
            ;;
        *)
            echo "Unknown option: $1" >&2; exit 1
            ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

download_model() {
    out="$OUT_DIR/$MODEL_FILE"
    part="$out.part"
    url="https://huggingface.co/$REPO/resolve/main/$MODEL_FILE"

    mkdir -p "$OUT_DIR"

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    echo "Downloading $MODEL_FILE"
    echo "from https://huggingface.co/$REPO"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --progress-meter -C - -o "$part" "$url"
    fi

    mv "$part" "$out"
    echo "Saved $out"
}

download_unified() {
    out="$OUT_DIR/$CANON_FILE"
    part="$out.part"
    url="https://huggingface.co/$CANON_REPO/resolve/main/$CANON_FILE"

    mkdir -p "$OUT_DIR"

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    echo "Downloading $CANON_FILE"
    echo "from https://huggingface.co/$CANON_REPO"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --progress-meter -C - -o "$part" "$url"
    fi

    mv "$part" "$out"
    echo "Saved $out"
}

cook_unified() {
    model="$OUT_DIR/$MODEL_FILE"
    canon="$OUT_DIR/$CANON_FILE"
    act_stats="$OUT_DIR/$ACT_STATS_FILE"

    if [ ! -s "$model" ]; then
        echo "Base GGUF not found: $model" >&2
        echo "Run './download_model.sh model' first." >&2
        exit 1
    fi

    if [ -s "$canon" ]; then
        echo "Already cooked: $canon"
        return
    fi

    if [ ! -s "$act_stats" ]; then
        echo "AWQ activation stats not found: $act_stats" >&2
        echo "Capture them from the base model first:" >&2
        echo "  cargo test -p qw35-server --lib real_model_capture_activations -- --ignored" >&2
        exit 1
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Cooking the unified model requires python3." >&2
        exit 1
    fi
    if ! python3 -c "import numpy, gguf" >/dev/null 2>&1; then
        echo "Cooking the unified model requires the numpy and gguf packages." >&2
        echo "Install them with:" >&2
        echo "  python3 -m pip install -U numpy gguf" >&2
        exit 1
    fi

    echo "Cooking unified $CANON_FILE from $MODEL_FILE (CPU-heavy, takes several minutes)..."
    python3 "$COOK" "$model" "$canon" --awq "$act_stats"
    echo "Cooked $canon"
}

case "$TARGET" in
    download|unified)
        download_unified
        ;;
    model)
        download_model
        echo
        echo "Tip: './download_model.sh cook' cooks the canonical unified $CANON_FILE,"
        echo "or './download_model.sh download' fetches the prebuilt one."
        ;;
    cook|gf4)
        cook_unified
        ;;
    all)
        download_model
        cook_unified
        ;;
esac

echo
echo "Done."
