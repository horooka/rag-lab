# Third-party sources

The actual sources are not versioned in this repository. They are downloaded
into this directory by `scripts/fetch-deps.sh` using the versions pinned there:

- tree-sitter runtime and the C, C++ and Python grammars
- sqlite-vec
- CLI11
- nlohmann/json
- ONNX Runtime

See [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for licenses and
upstream links.
