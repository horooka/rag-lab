#!/usr/bin/env python3
import json
import os
import pathlib
import subprocess
import sys
import tempfile

mcp_bin = pathlib.Path(sys.argv[1])
rag_bin = pathlib.Path(sys.argv[2])
model_dir = os.environ.get("RAG_MODEL_DIR", "")
if not model_dir:
    raise SystemExit("RAG_MODEL_DIR is required")

with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp)
    (root / "cpp").mkdir()
    (root / "pascal").mkdir()
    (root / "cpp" / "hello.cpp").write_text(
        "int archive_packet() { return 1; }\n", encoding="utf-8"
    )
    (root / "pascal" / "StationID.pas").write_text(
        "unit StationID;\ninterface\nprocedure FillStationsList;\n"
        "implementation\nprocedure FillStationsList;\nbegin\nend;\nend.\n",
        encoding="utf-8",
    )

    def run_index(source, db):
        subprocess.run(
            [
                str(rag_bin),
                "index",
                str(root / source),
                "--db",
                str(db),
                "--model-dir",
                model_dir,
                "--quiet",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    run_index("cpp", root / "cpp.db")
    run_index("pascal", root / "pascal.db")
    workspace = root / "workspace.json"
    workspace.write_text(
        json.dumps(
            {
                "indexes": [
                    {"name": "cpp", "db": "cpp.db"},
                    {"name": "pascal", "db": "pascal.db"},
                ]
            }
        ),
        encoding="utf-8",
    )

    process = subprocess.Popen(
        [str(mcp_bin), "--workspace", str(workspace), "--model-dir", model_dir],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def send(message):
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    def receive():
        line = process.stdout.readline()
        if not line:
            raise AssertionError("MCP server closed stdout: " + process.stderr.read())
        return json.loads(line)

    def call(request_id, method, params=None):
        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        send(message)
        response = receive()
        assert response["id"] == request_id, response
        return response

    init = call(
        1,
        "initialize",
        {
            "protocolVersion": "2026-07-28",
            "capabilities": {},
            "clientInfo": {"name": "mcp-smoke", "version": "0"},
        },
    )
    assert "result" in init, init
    assert init["result"]["capabilities"]["tools"] is not None

    send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    tools = call(2, "tools/list")
    names = {tool["name"] for tool in tools["result"]["tools"]}
    assert {
        "search_codebase",
        "read_chunk",
        "resolve_symbol",
        "index_stats",
        "list_sources",
    } <= names, names

    search = call(
        3,
        "tools/call",
        {
            "name": "search_codebase",
            "arguments": {"query": "FillStationsList", "top_k": 5},
        },
    )
    search_payload = json.loads(search["result"]["content"][0]["text"])
    assert search_payload["results"], search_payload
    pascal_hit = next(
        item for item in search_payload["results"] if item["source"] == "pascal"
    )
    assert pascal_hit["path"] == "StationID.pas"

    chunk = call(
        4,
        "tools/call",
        {
            "name": "read_chunk",
            "arguments": {
                "source": "pascal",
                "path": "StationID.pas",
                "start_line": 1,
                "end_line": 20,
            },
        },
    )
    chunk_payload = json.loads(chunk["result"]["content"][0]["text"])
    assert chunk_payload["chunks"], chunk_payload

    resolved = call(
        5,
        "tools/call",
        {
            "name": "resolve_symbol",
            "arguments": {"symbol": "FillStationsList", "top_k": 5},
        },
    )
    resolve_payload = json.loads(resolved["result"]["content"][0]["text"])
    assert any(item["source"] == "pascal" for item in resolve_payload["definitions"]), (
        resolve_payload
    )

    stats = call(6, "tools/call", {"name": "index_stats", "arguments": {}})
    stats_payload = json.loads(stats["result"]["content"][0]["text"])
    assert stats_payload["totals"]["files"] == 2, stats_payload

    sources = call(7, "tools/call", {"name": "list_sources", "arguments": {}})
    sources_payload = json.loads(sources["result"]["content"][0]["text"])
    assert {item["name"] for item in sources_payload} == {"cpp", "pascal"}

    process.stdin.close()
    process.wait(timeout=10)
    if process.returncode != 0:
        raise AssertionError(process.stderr.read())

print("mcp smoke test passed")
