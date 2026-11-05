# Third-party notices

`rag-lab` uses the following third-party components. Their source is fetched by
`scripts/fetch-deps.sh` rather than vendored in this repository; the versions
are pinned in that script.

| component | version | license | upstream |
|---|---|---|---|
| tree-sitter | 0.27.1 | MIT | https://github.com/tree-sitter/tree-sitter |
| tree-sitter-c | 0.24.2 | MIT | https://github.com/tree-sitter/tree-sitter-c |
| tree-sitter-cpp | 0.23.4 | MIT | https://github.com/tree-sitter/tree-sitter-cpp |
| tree-sitter-python | 0.25.0 | MIT | https://github.com/tree-sitter/tree-sitter-python |
| tree-sitter-pascal | 0.10.2 | MIT | https://github.com/Isopod/tree-sitter-pascal |
| sqlite-vec | 0.1.9 | MIT OR Apache-2.0 | https://github.com/asg017/sqlite-vec |
| CLI11 | 2.7.2 | BSD-3-Clause | https://github.com/CLIUtils/CLI11 |
| nlohmann/json | 3.12.0 | MIT | https://github.com/nlohmann/json |
| ONNX Runtime | 1.31.0 | MIT | https://github.com/microsoft/onnxruntime |
| bge-small-en-v1.5 | BAAI / Xenova ONNX | MIT | https://huggingface.co/BAAI/bge-small-en-v1.5 |

SQLite is provided by the system and is in the public domain.

Fetching sources at build time does not remove license obligations for
distributed binaries: keep this file and the upstream license texts with any
binary or container release. The full license texts live in the respective
upstream repositories linked above.
