#!/usr/bin/env bash
set -euo pipefail

RAG="${1:?usage: smoke.sh /path/to/rag}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/src"
cat > "$TMP/src/hello.cpp" <<'EOF'
#include <string>

// Archive an RTP packet to disk.
bool archive_packet(const std::string& packet) {
    return packet.size() > 0;
}
EOF

cat > "$TMP/src/app.py" <<'EOF'
def load_configuration(path):
    with open(path) as handle:
        return handle.read()
EOF

MODEL_ARGS=()
if [[ -n "${RAG_MODEL_DIR:-}" ]]; then
    MODEL_ARGS=(--model-dir "$RAG_MODEL_DIR")
fi

"$RAG" index "$TMP" --db "$TMP/.rag/index.db" --quiet "${MODEL_ARGS[@]}" >/dev/null

"$RAG" search "archive RTP packet" --root "$TMP" --db "$TMP/.rag/index.db" \
    --json "${MODEL_ARGS[@]}" > "$TMP/result.json"
python3 - "$TMP/result.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
paths = [item["path"] for item in data["results"]]
assert "src/hello.cpp" in paths, paths
assert data["results"], "no results"
PY

"$RAG" search "parse_config" --root "$TMP" --db "$TMP/.rag/index.db" \
    --mode bm25 --json "${MODEL_ARGS[@]}" > "$TMP/bm25.json"
python3 - "$TMP/bm25.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data["results"] == [], "unexpected bm25 match for absent identifier"
PY

# Search should still find the Python function via dense retrieval.
"$RAG" search "read configuration file" --root "$TMP" --db "$TMP/.rag/index.db" \
    --mode dense --json "${MODEL_ARGS[@]}" > "$TMP/dense.json"
python3 - "$TMP/dense.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
paths = [item["path"] for item in data["results"][:3]]
assert "src/app.py" in paths, paths
PY

# Recency filters operate on the filesystem mtime captured at index time.
"$RAG" search "archive RTP packet" --root "$TMP" --db "$TMP/.rag/index.db" \
    --after 2000-01-01 --json "${MODEL_ARGS[@]}" > "$TMP/after.json"
python3 - "$TMP/after.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert any(item["path"] == "src/hello.cpp" for item in data["results"]), data["results"]
PY

"$RAG" search "archive RTP packet" --root "$TMP" --db "$TMP/.rag/index.db" \
    --after 2200-01-01 --json "${MODEL_ARGS[@]}" > "$TMP/future.json"
python3 - "$TMP/future.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data["results"] == [], data["results"]
PY

# Workspace querying across two independent indexes.
mkdir -p "$TMP/ws/cpp" "$TMP/ws/pascal"
cat > "$TMP/ws/cpp/hello.cpp" <<'EOF'
int archive_packet() { return 1; }
EOF
cat > "$TMP/ws/pascal/StationID.pas" <<'EOF'
unit StationID;

interface

procedure FillStationsList;

implementation

procedure FillStationsList;
begin
end;

end.
EOF
"$RAG" index "$TMP/ws/cpp" --db "$TMP/ws/cpp.db" --quiet "${MODEL_ARGS[@]}" >/dev/null
"$RAG" index "$TMP/ws/pascal" --db "$TMP/ws/pascal.db" --quiet "${MODEL_ARGS[@]}" >/dev/null
cat > "$TMP/ws/workspace.json" <<'JSON'
{
  "indexes": [
    {"name": "cpp", "db": "cpp.db"},
    {"name": "pascal", "db": "pascal.db"}
  ]
}
JSON

"$RAG" search "FillStationsList" --workspace "$TMP/ws/workspace.json" \
    --source pascal --json "${MODEL_ARGS[@]}" > "$TMP/ws/pascal-result.json"
python3 - "$TMP/ws/pascal-result.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data["results"], "workspace returned no Pascal results"
assert any(item.get("source") == "pascal" for item in data["results"]), data["results"]
PY

"$RAG" search "archive_packet" --workspace "$TMP/ws/workspace.json" \
    --json "${MODEL_ARGS[@]}" > "$TMP/ws/cpp-result.json"
python3 - "$TMP/ws/cpp-result.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert any(item.get("source") == "cpp" for item in data["results"]), data["results"]
PY

"$RAG" resolve FillStationsList --workspace "$TMP/ws/workspace.json" \
    --json "${MODEL_ARGS[@]}" > "$TMP/ws/resolve.json"
python3 - "$TMP/ws/resolve.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert any(item.get("source") == "pascal" for item in data["definitions"]), data["definitions"]
PY

"$RAG" stats --workspace "$TMP/ws/workspace.json" > "$TMP/ws/stats.txt"
grep -q "\[cpp\]" "$TMP/ws/stats.txt"
grep -q "\[pascal\]" "$TMP/ws/stats.txt"

echo "smoke test passed"
