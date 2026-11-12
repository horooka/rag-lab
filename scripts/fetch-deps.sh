#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD="$ROOT/third_party"
ORT_VERSION="1.31.0"
MODEL_REPO="Xenova/bge-small-en-v1.5"
ORT_DIR="$THIRD/onnxruntime"
MODEL_DIR="$ROOT/models/bge-small-en-v1.5"

FORCE=0
LANGUAGES_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=1
            shift
            ;;
        --languages)
            LANGUAGES_ARG="${2:?--languages needs a value}"
            shift 2
            ;;
        --languages=*)
            LANGUAGES_ARG="${1#*=}"
            shift
            ;;
        *)
            echo "usage: $0 [--force] [--languages c,cpp,python,pascal,luau,rust]" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$THIRD" "$MODEL_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

EXTRACTED=""

fetch_archive() {
    local name="$1"
    local url="$2"
    echo "Fetching ${name}..." >&2
    local archive="$TMP/${name}.tar.gz"
    curl -fL --retry 3 --silent --show-error "$url" -o "$archive"
    mkdir -p "$TMP/${name}"
    tar xzf "$archive" -C "$TMP/${name}"
    EXTRACTED="$(find "$TMP/${name}" -mindepth 1 -maxdepth 1 -type d -print -quit)"
    if [[ -z "$EXTRACTED" ]]; then
        echo "error: archive ${name} has no top-level directory" >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# tree-sitter runtime
# ---------------------------------------------------------------------------
if [[ "$FORCE" -eq 1 || ! -f "$THIRD/tree-sitter/lib/src/lib.c" ]]; then
    fetch_archive "tree-sitter-0.27.1" \
        "https://github.com/tree-sitter/tree-sitter/archive/refs/tags/v0.27.1.tar.gz"
    rm -rf "$THIRD/tree-sitter"
    mkdir -p "$THIRD/tree-sitter/lib"
    cp -r "$EXTRACTED/lib/src" "$THIRD/tree-sitter/lib/"
    cp -r "$EXTRACTED/lib/include" "$THIRD/tree-sitter/"
fi

# ---------------------------------------------------------------------------
# tree-sitter grammars
# ---------------------------------------------------------------------------
fetch_grammar() {
    local name="$1"
    local version="$2"
    local directory="$3"
    local owner="${4:-tree-sitter}"
    local dest="$THIRD/${directory}"
    local marker="${dest}/src/parser.c"
    if [[ "$FORCE" -eq 0 && -f "$marker" ]]; then
        return
    fi
    fetch_archive "${directory}-${version}" \
        "https://github.com/${owner}/${directory}/archive/refs/tags/v${version}.tar.gz"
    rm -rf "$dest"
    mkdir -p "${dest}/src/tree_sitter" "${dest}/bindings"
    cp "$EXTRACTED/src/parser.c" "${dest}/src/"
    find "$EXTRACTED/src/tree_sitter" -maxdepth 1 -name '*.h' \
        -exec cp {} "${dest}/src/tree_sitter/" \;
    if [[ -f "$EXTRACTED/src/scanner.c" ]]; then
        cp "$EXTRACTED/src/scanner.c" "${dest}/src/"
    fi
    local header
    header="$(find "$EXTRACTED/bindings/c" -name "tree-sitter-${name}.h" -print -quit)"
    if [[ -z "$header" ]]; then
        echo "error: missing tree-sitter-${name}.h in archive" >&2
        exit 1
    fi
    cp "$header" "${dest}/bindings/"
}

languages=()
if [[ -n "$LANGUAGES_ARG" ]]; then
    read -ra languages <<< "${LANGUAGES_ARG//,/ }"
elif [[ -n "${RAG_TS_LANGUAGES:-}" ]]; then
    read -ra languages <<< "${RAG_TS_LANGUAGES//,/ }"
else
    languages=(c cpp python pascal luau rust)
fi

if [[ ${#languages[@]} -eq 0 ]]; then
    echo "error: no tree-sitter languages enabled" >&2
    exit 1
fi

for language in "${languages[@]}"; do
    case "$language" in
        c)
            fetch_grammar "c" "0.24.2" "tree-sitter-c"
            ;;
        cpp)
            fetch_grammar "cpp" "0.23.4" "tree-sitter-cpp"
            ;;
        python)
            fetch_grammar "python" "0.25.0" "tree-sitter-python"
            ;;
        pascal)
            fetch_grammar "pascal" "0.10.2" "tree-sitter-pascal" "Isopod"
            ;;
        luau)
            fetch_grammar "luau" "1.2.0" "tree-sitter-luau" \
                "tree-sitter-grammars"
            ;;
        rust)
            fetch_grammar "rust" "0.24.2" "tree-sitter-rust"
            ;;
        *)
            echo "error: unknown tree-sitter language: $language" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# sqlite-vec
# ---------------------------------------------------------------------------
if [[ "$FORCE" -eq 1 || ! -f "$THIRD/sqlite-vec/sqlite-vec.h" ]]; then
    fetch_archive "sqlite-vec-0.1.9" \
        "https://github.com/asg017/sqlite-vec/archive/refs/tags/v0.1.9.tar.gz"
    rm -rf "$THIRD/sqlite-vec"
    mkdir -p "$THIRD/sqlite-vec"
    cp "$EXTRACTED/sqlite-vec.c" "$EXTRACTED/sqlite-vec.h.tmpl" \
        "$THIRD/sqlite-vec/"
    VERSION="$(cat "$EXTRACTED/VERSION")"
    VERSION_MAJOR="${VERSION%%.*}"
    REST="${VERSION#*.}"
    VERSION_MINOR="${REST%%.*}"
    VERSION_PATCH="${REST#*.}"
    VERSION_PATCH="${VERSION_PATCH%%-*}"
    DATE="$(date -u +'%FT%TZ%z')"
    sed -e "s/\${VERSION}/${VERSION}/g" \
        -e "s/\${SOURCE}/fetched/g" \
        -e "s/\${DATE}/${DATE}/g" \
        -e "s/\${VERSION_MAJOR}/${VERSION_MAJOR}/g" \
        -e "s/\${VERSION_MINOR}/${VERSION_MINOR}/g" \
        -e "s/\${VERSION_PATCH}/${VERSION_PATCH}/g" \
        "$EXTRACTED/sqlite-vec.h.tmpl" > "$THIRD/sqlite-vec/sqlite-vec.h"
fi

# ---------------------------------------------------------------------------
# CLI11 and nlohmann/json
# ---------------------------------------------------------------------------
if [[ "$FORCE" -eq 1 || ! -f "$THIRD/CLI11/CLI/CLI.hpp" ]]; then
    fetch_archive "CLI11-2.7.2" \
        "https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.7.2.tar.gz"
    rm -rf "$THIRD/CLI11"
    mkdir -p "$THIRD/CLI11"
    if [[ -d "$EXTRACTED/single_include/CLI" ]]; then
        cp -r "$EXTRACTED/single_include/CLI" "$THIRD/CLI11/"
    else
        cp -r "$EXTRACTED/include/CLI" "$THIRD/CLI11/"
    fi
fi

if [[ "$FORCE" -eq 1 || ! -f "$THIRD/nlohmann/nlohmann/json.hpp" ]]; then
    fetch_archive "json-3.12.0" \
        "https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz"
    rm -rf "$THIRD/nlohmann"
    mkdir -p "$THIRD/nlohmann/nlohmann"
    if [[ -f "$EXTRACTED/single_include/nlohmann/json.hpp" ]]; then
        cp "$EXTRACTED/single_include/nlohmann/json.hpp" \
            "$THIRD/nlohmann/nlohmann/"
    else
        cp "$EXTRACTED/include/nlohmann/json.hpp" \
            "$THIRD/nlohmann/nlohmann/"
    fi
fi

# ---------------------------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------------------------
if [[ "$FORCE" -eq 1 || ! -f "$ORT_DIR/include/onnxruntime_cxx_api.h" ]]; then
    echo "Fetching ONNX Runtime ${ORT_VERSION}..." >&2
    curl -fL --retry 3 --silent --show-error \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
        -o "$TMP/ort.tgz"
    tar xzf "$TMP/ort.tgz" -C "$TMP"
    rm -rf "$ORT_DIR"
    mv "$TMP/onnxruntime-linux-x64-${ORT_VERSION}" "$ORT_DIR"
fi

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
fetch_file() {
    local url="$1"
    local dest="$2"
    if [[ ! -f "$dest" ]]; then
        echo "Fetching $(basename "$dest")..." >&2
        mkdir -p "$(dirname "$dest")"
        curl -fL --retry 3 --silent --show-error "$url" -o "$dest"
    fi
}

fetch_file "https://huggingface.co/${MODEL_REPO}/resolve/main/onnx/model_quantized.onnx" \
    "$MODEL_DIR/model_quantized.onnx"
fetch_file "https://huggingface.co/${MODEL_REPO}/resolve/main/vocab.txt" \
    "$MODEL_DIR/vocab.txt"

if [[ ! -f "$MODEL_DIR/model.json" ]]; then
    cat > "$MODEL_DIR/model.json" <<'EOF'
{
  "name": "bge-small-en-v1.5",
  "onnx_file": "model_quantized.onnx",
  "vocab_file": "vocab.txt",
  "dim": 384,
  "max_length": 512,
  "pooling": "cls",
  "normalize": true,
  "query_prefix": "Represent this sentence for searching relevant passages: ",
  "doc_prefix": ""
}
EOF
fi

echo "Dependencies ready."
