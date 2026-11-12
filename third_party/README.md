# Third-party sources

The actual sources are not versioned in this repository. CMake invokes
`scripts/fetch-deps.sh` during configure, with the tree-sitter language set
coming from `RAG_TS_LANGUAGES`. Versions are pinned in the script:

- tree-sitter runtime and the grammars listed in
  `config/tree-sitter-languages.txt` (currently C, C++, Python, Pascal, Luau,
  Rust)
- sqlite-vec
- CLI11
- nlohmann/json
- ONNX Runtime

See [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for licenses and
upstream links.
