#!/bin/sh
set -e

# qw35 model downloader.
#
# Fetches the base GGUF from Hugging Face into ./.gguf/ (the cwd-relative path
# the server loads by default) and can cook the optional GF4 decode sidecar with
# the project tool. The GF4 sidecar is NOT downloadable: it is generated locally
# from the GGUF.

REPO="unsloth/Qwen3.5-9B-GGUF"
MODEL_FILE="Qwen3.5-9B-Q4_K_M.gguf"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${QW35_GGUF_DIR:-"$ROOT/.gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}
COOK="$ROOT/qw35-tool/qw35_tool/cook_qw35_ffn_gf4_sidecar.py"

usage() {
    cat <<EOF
qw35 model downloader

Usage:
  ./download_model.sh [model|gf4|all] [--token TOKEN]

Targets:

  model   (default) Download the base GGUF
          $MODEL_FILE (~5.3 GB) into the model directory. This is all the
          server needs to run.

  gf4     Cook the GF4 decode sidecar from the downloaded GGUF using
          qw35-tool. Optional: the server runs without it (just slower decode,
          ~13.7 vs ~19.8 tok/s) and picks it up automatically when present.
          CPU-heavy; takes several minutes. Requires python3 + numpy + gguf.

  all     Run model, then gf4.

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF token
                 cache (~/.cache/huggingface/token) is used if present.

Environment:
  QW35_GGUF_DIR  Directory used for the GGUF and sidecar.
                 Default: ./.gguf

After downloading, the default server command just works:
  ./target/release/qw35      (or: make run)
EOF
}

TARGET=model
case "${1:-}" in
    model|gf4|all) TARGET=$1; shift ;;
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

cook_gf4() {
    model="$OUT_DIR/$MODEL_FILE"
    sidecar="$OUT_DIR/${MODEL_FILE%.gguf}.gf4.bin"

    if [ ! -s "$model" ]; then
        echo "GGUF not found: $model" >&2
        echo "Run './download_model.sh model' first." >&2
        exit 1
    fi

    if [ -s "$sidecar" ]; then
        echo "Already cooked: $sidecar"
        return
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Cooking the GF4 sidecar requires python3." >&2
        exit 1
    fi
    if ! python3 -c "import numpy, gguf" >/dev/null 2>&1; then
        echo "Cooking the GF4 sidecar requires the numpy and gguf packages." >&2
        echo "Install them with:" >&2
        echo "  python3 -m pip install -U numpy gguf" >&2
        exit 1
    fi

    echo "Cooking GF4 sidecar from $MODEL_FILE (CPU-heavy, takes several minutes)..."
    python3 "$COOK" "$model" --only full
    echo "Cooked $sidecar"
}

case "$TARGET" in
    model)
        download_model
        echo
        echo "Tip: './download_model.sh gf4' cooks the optional GF4 speed sidecar."
        ;;
    gf4)
        cook_gf4
        ;;
    all)
        download_model
        cook_gf4
        ;;
esac

echo
echo "Done."
