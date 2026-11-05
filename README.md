# rag-lab

A lab for tuning code RAG pipelines: chunkers, embeddings, retrieval and
evaluation. It includes semantic code search over local source trees, plus
in-memory semantic grouping. C++17 core, CPU-only embeddings, SQLite index,
hybrid dense + BM25 retrieval, and a small retrieval-evaluation harness.

## License

Original code is dedicated to the public domain under CC0 1.0 Universal; see
`LICENSE`. Third-party components keep their own licenses and attribution
requirements; see `THIRD_PARTY_NOTICES.md`.

## Requirements

- Linux, CMake ≥ 3.22, a C++17 compiler (tested with GCC 16)
- SQLite ≥ 3.35 with FTS5 (`libsqlite3-dev`)
- `curl`, `tar`, `iconv` (glibc provides it)
- ~50 MB of fetched sources, ONNX Runtime and the model; ~15 GB RAM is plenty

All third-party sources (tree-sitter, sqlite-vec, CLI11, nlohmann/json, ONNX
Runtime) and the embedding model are fetched by `scripts/fetch-deps.sh` into
gitignored directories. See `THIRD_PARTY_NOTICES.md`.

## Build

```bash
./scripts/fetch-deps.sh          # sources, ONNX Runtime and the model
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

The binaries are `build/rag`, `build/grouper` and `build/coverage`.
`scripts/fetch-deps.sh` is idempotent; `--force` re-downloads everything. After
one successful fetch, CMake builds work offline. Dependency licenses are listed
in `THIRD_PARTY_NOTICES.md`.

The local project is now `rag-lab`. The git remote still points at the old
GitHub name until the repository is renamed there:

```bash
git remote set-url origin https://github.com/horooka/rag-lab.git
```

## Quick start

Index one project first. The target path is required, and keeping the index
outside the source tree avoids writing untracked files into someone else's
repository:

```bash
./build/rag index /path/to/project \
    --db indexes/project.db \
    --model-dir models/bge-small-en-v1.5 \
    --threads 8 --batch-size 16

./build/rag stats --db indexes/project.db \
    --model-dir models/bge-small-en-v1.5

./build/rag search "read controllers from a binary file" \
    --db indexes/project.db \
    --model-dir models/bge-small-en-v1.5 -k 5

./build/rag search "GetStationNameById" \
    --db indexes/project.db \
    --model-dir models/bge-small-en-v1.5 -k 5
```

If `--db` is omitted the index is placed at `<root>/.rag/index.db` and
`<root>/.rag/` is where an optional `config.json` is read from. There is no
default corpus: `rag index` without a path is an error.

A larger tree can be indexed the same way once the per-project workflow is
trusted:

```bash
./build/rag index /path/to/large-tree --db indexes/tree.db \
    --model-dir models/bge-small-en-v1.5 --threads 8 --batch-size 16
```

## Workspaces: multiple indexes

Keep one index per root and let the query layer merge them. This avoids path
collisions, cross-root stale deletion, and schema migrations, while still
allowing different encoding/chunker settings per project.

```bash
./build/rag index /path/to/delphi --db indexes/delphi.db \
    --model-dir models/bge-small-en-v1.5

./build/rag index /path/to/cpp --db indexes/cpp.db \
    --model-dir models/bge-small-en-v1.5
```

`examples/workspace.json`:

```json
{
  "indexes": [
    {"name": "delphi", "db": "../indexes/delphi.db"},
    {"name": "cpp", "db": "../indexes/cpp.db"}
  ]
}
```

```bash
./build/rag search "FillStationsList" --workspace examples/workspace.json
./build/rag stats --workspace examples/workspace.json
```

Each hit is tagged with its source, so two indexes containing
`src/foo.cpp` stay distinguishable:

```
1. [delphi] StationID.pas:20-22  TStation.FillStationsList
2. [cpp] src/stations_ini.cpp:44-58  parse_stations_ini
```

Details:

- `db` paths are resolved relative to the workspace file.
- The compact map form `{"indexes": {"delphi": "delphi.db"}}` is also accepted.
- All indexes must use the same embedding model and dimension for dense/hybrid
  search; the workspace validates this before querying.
- `--source NAME` restricts a workspace query to one index.
- `--db` and `--workspace` are mutually exclusive.
- `stats --workspace` prints per-index statistics and aggregate totals.
- A future `rag port <symbol>` will use the same workspace list.
- There is intentionally no `source_id` in the index schema: isolation is per
  database, merging is a query-layer concern.

## Utils

### Semantic grouper

`grouper` reuses the same tokenizer and ONNX embedder through `rag_embed`, but
keeps everything in memory: no SQLite, no tree-sitter, no index. Inputs are
embedded and clustered with deterministic agglomerative clustering (average
linkage by default).

```bash
printf '%s\n' \
  "parse a configuration file" \
  "read settings from disk" \
  "load the config" \
  "train a neural network" \
  "fit a machine learning model" \
  "build a deep learning classifier" \
  | ./build/grouper --threads 8
```

```
2 groups from 6 items

Group 1  (3 items, cohesion 0.920)
  build a deep learning classifier
  train a neural network
  fit a machine learning model

Group 2  (3 items, cohesion 0.898)
  parse a configuration file
  load the config
  read settings from disk
```

Keywords can be passed directly, optionally with file inputs:

```bash
./build/grouper --groups 3 apple pear banana carrot potato beef chicken fish
./build/grouper --threshold 0.30 --file keywords.txt --json
```

| option | meaning |
|---|---|
| `-n, --groups N` | exact number of groups |
| `-t, --threshold D` | merge while cosine distance is at most `D` |
| `--max-groups N` | upper bound for automatic group selection |
| `--linkage average\|complete\|single` | agglomerative linkage |
| `--keep-duplicates` | keep exact duplicates |
| `-d, --delimiter C` | split each input line further |
| `--model-dir DIR` | another compatible ONNX model directory |
| `--threads N`, `--batch-size N` | ONNX Runtime tuning |
| `--json`, `-v, --verbose` | machine-readable / diagnostic output |

The model is resolved from `--model-dir`, `$GROUPER_MODEL_DIR`,
`$RAG_MODEL_DIR`, `./models/bge-small-en-v1.5`, then the path compiled into the
binary. The standalone project's tests were not carried over; retrieval tests
remain in `tests/`.

### Coverage analysis

`coverage` reads an existing index and assigns every chunk to the most similar
supplied group. It answers "how much of this codebase is networking / database /
UI code?" without re-indexing or generating text.

Groups are defined by keywords and embedded through the same model:

```bash
./build/coverage --db indexes/project.db \
    --group "network=socket,packet,rtp,connection" \
    --group "database=sql,query,transaction,table" \
    --group "ui=button,dialog,window,widget" \
    --details
```

```
group                        chunks     lines     share   mean_sim
network                         120       980     34.2%      0.581
database                        145      1320     41.3%      0.563
ui                               52       410     14.8%      0.612
unassigned                       34       260      9.7%      0.411
```

Groups can also come from a JSON file:

```json
[
  {"label": "network", "keywords": ["socket", "packet", "rtp"]},
  {"label": "database", "keywords": ["sql", "query", "transaction"]}
]
```

or the compact map form `{"network": ["socket", "packet"], "ui": "button"}`.
A starter set is provided in `eval/groups-example.json`:

```bash
./build/coverage --db indexes/project.db \
    --groups-file eval/groups-example.json \
    --unit lines --threshold 0.45 --details --json
```

| option | meaning |
|---|---|
| `--db PATH` | index database (required) |
| `-g, --group LABEL=KW1,KW2` | inline group, repeatable |
| `--groups-file FILE` | JSON array/object of groups |
| `--unit chunks\|lines` | count chunks (default) or de-duplicated line spans |
| `--threshold S` | minimum cosine similarity; lower values become `unassigned` |
| `--strategy max\|centroid` | best keyword (default) or keyword-centroid similarity |
| `--details` | show the best-matching chunks per group |
| `--top N` | matches per group with `--details` |
| `--model-dir DIR` | override the embedding model |
| `--threads N` | ONNX Runtime threads |
| `--json` | machine-readable output |

`--unit chunks` is fast and is the right default. `--unit lines` de-duplicates
overlapping chunk ranges by `(path, line)`, giving true coverage of the lines
represented by the index at the cost of more memory. Lines/blocks that were
never chunked (for example some global declarations) are outside the
denominator; the tool reports coverage of indexed content, not raw file lines.
Line spans include blank lines inside a chunk.

## Commands

### `rag index [path]`

| option | meaning |
|---|---|
| `--db PATH` | index database (default `<root>/.rag/index.db`) |
| `--model-dir DIR` | embedding model directory |
| `--full` | drop and rebuild instead of incremental update |
| `--exclude GLOB` | additional exclude pattern, repeatable |
| `--threads N` | ONNX Runtime intra-op threads (0 = auto) |
| `--batch-size N` | embedding batch size |
| `--window-lines N`, `--overlap-lines N` | fallback text windows |
| `--max-tokens N` | approximate chunk budget (default 512) |
| `--encoding NAME` | fallback encoding for non-UTF-8 files (`auto` by default) |
| `--quiet` | no per-file progress |

Default exclusions cover VCS metadata, build trees, virtualenvs,
`node_modules`, object files and binaries. Dotfiles are ignored except known
build/configuration files. Files over 5 MB and files containing NUL bytes are
skipped.

### `rag search <query>`

| option | meaning |
|---|---|
| `--mode hybrid\|dense\|bm25` | retrieval strategy (default `hybrid`) |
| `-k, --top-k N` | result count (default 10) |
| `--repo NAME` | repository filter |
| `--lang cpp\|python\|c\|pascal\|text` | language filter |
| `--path PREFIX` | relative path prefix filter |
| `--workspace FILE` | query multiple named indexes |
| `--source NAME` | restrict a workspace query to one index |
| `--after ISO` | only files modified at/after this UTC time |
| `--before ISO` | only files modified at/before this UTC time |
| `--dense-weight W`, `--bm25-weight W` | override adaptive RRF weights |
| `--no-auto-weights` | use the configured weights verbatim |
| `--json` | machine-readable output including full chunk text |

Each result reports `path:start-end`, symbol, fused score, and the dense/BM25
ranks that contributed to it.

```bash
./build/rag search "packet handling" --db indexes/project.db \
    --model-dir models/bge-small-en-v1.5 --after 2026-10-01
```

`--after`/`--before` accept `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM` or
`YYYY-MM-DDTHH:MM:SSZ` and compare against the filesystem mtime stored for each
file at index time. Date-only values mean 00:00 UTC. Indexes created before mtime was normalized to Unix
nanoseconds must be rebuilt with `--full` once. Filtered dense retrieval scans
the matching subset and ranks it directly, so a narrow date window cannot
silently lose candidates to KNN pre-filtering.

### `rag resolve <symbol>`

Resolves a symbol across one index or a workspace. It partitions the hits into:

- **definitions** — exact symbol matches, including qualified forms
  (`TStation.FillStationsList` when resolving `FillStationsList`);
- **references** — chunks whose text contains the symbol (approximate call
  sites/declarations);
- **candidates** — related chunks from other sources/languages, useful for
  finding an analogue such as `fill_project_stations`.

```bash
./build/rag resolve FillStationsList \
    --workspace examples/workspace.json --source delphi

./build/rag resolve FillStationsList \
    --workspace examples/workspace.json --json
```

| option | meaning |
|---|---|
| `--workspace FILE` | resolve across named indexes |
| `--db PATH` | resolve in one index |
| `--source NAME` | workspace index filter |
| `--lang`, `--repo`, `--path` | metadata filters |
| `--after`, `--before` | file mtime filters |
| `-k, --top-k N` | hits per section (default 20) |
| `--json` | structured output |

Manifest-aware status (`done` / `partial` / `todo`) will be added once the
manifest TSV chunker exists.

### `rag stats`

Files, chunks, repositories, languages, model, index timestamp and database
size. `--json` for scripting.

### `rag eval <queries.jsonl>`

JSONL ground truth, one object per line:

```json
{"query": "read station names from Stations.ini", "relevant": ["src/StationNames.cpp"]}
{"query": "GetDynParam", "relevant": ["src/BinDataStruct.cpp::GetDynParam"]}
```

References may be a full relative path, a path suffix, or `path::symbol`.
Metrics are recall@k (distinct relevant chunks retrieved / min(relevant, k)),
MRR and hit@k. `--mode all` compares dense, BM25 and hybrid; `--details` prints
the rank of the first relevant hit per query. A starter set lives in
`eval/queries-fastparam.jsonl`.

## Configuration

Optional `<root>/.rag/config.json`:

```json
{
  "db_path": ".rag/index.db",
  "model_dir": <path>,
  "mode": "hybrid",
  "top_k": 10,
  "batch_size": 8,
  "threads": 8,
  "include_docs": true,
  "exclude": ["include/tinyxml2/*", "src/tinyxml2.cpp"],
  "chunk": {
    "max_tokens": 512,
    "window_lines": 60,
    "overlap_lines": 12,
    "include_header": true
  },
  "chunkers": [
    {"match": "*.md", "type": "markdown", "max_tokens": 300},
    {"match": "*.cpp", "type": "treesitter", "max_tokens": 512},
    {"match": "*", "type": "window", "window_lines": 60, "overlap_lines": 12}
  ]
}
```

Chunker profiles are evaluated in order and matched against both the relative
path and the bare file name, so `*.md` and `CMakeLists.txt` both work. The
built-in types are:

- `treesitter` — AST chunks for C, C++, Python and Pascal (also the fallback
  for unsupported languages inside that chunker);
- `markdown` — heading sections with hierarchical symbols such as
  `Intro > Setup`, ignoring `#` inside fenced code blocks;
- `window` — overlapping line windows for scripts, build files and unknown
  formats.

When `chunkers` is omitted, the defaults map C/C++/Python/Pascal files to
tree-sitter, Markdown/RST to the heading chunker, and everything else to line
windows.
Profiles can override `max_tokens`, `window_lines` and `overlap_lines`
individually. `exclude` patterns without `/` match any path component; patterns 
with `/` match the relative path. Excluding vendored XML parsers is a good first
move on projects that embed third-party headers.

### Source encoding

Decoding happens in the `Source` layer, before chunking or embedding. Every
source returns UTF-8; downstream code never sees cp1251/UTF-16 bytes.

- Valid UTF-8 passes through unchanged.
- A UTF-8 BOM is stripped.
- UTF-16 LE/BE files with a BOM are converted.
- Otherwise the first matching `encodings` rule is used; the default rules
  treat Delphi/Pascal extensions (`.pas`, `.dpr`, `.dpk`, `.lpr`, `.dfm`,
  `.inc`) as cp1251.
- If no rule matches and the file is not UTF-8, indexing reports the file and
  suggests configuring an encoding.

```json
{
  "encoding": "auto",
  "encodings": [
    {"match": "*.pas", "encoding": "cp1251"},
    {"match": "*.dpr", "encoding": "cp1251"},
    {"match": "*.cpp", "encoding": "utf-8"}
  ]
}
```

`--encoding cp1251` sets the global fallback for a single `rag index` run.
Encoding rules are evaluated in order and matched against the relative path and
file name. `auto` means “do not guess”: a non-UTF-8 file without a matching
rule is an error rather than silent mojibake.

## Embedding models

The model directory must contain `model.json`, an ONNX file and a BERT
WordPiece `vocab.txt`. The shipped spec is:

```json
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
```

`pooling` supports `cls` and `mean`; `dim`, prefixes and normalization are read
at runtime. Swapping in another BERT-family ONNX embedding model (for example a
code-specific 768-dimensional model) is a directory swap plus `--full`
reindexing.

## Index schema

```
meta(key, value)                       model, dim, root, indexed_at, chunk params
files(id, path, repo, lang, mtime_ns, size, hash, chunk_count, indexed_at)
chunks(id, file_id, seq, path, repo, lang, symbol, kind, start_line, end_line, text)
vec_chunks(rowid, embedding float[384])   sqlite-vec virtual table
fts_chunks(rowid, text, symbol, path)     FTS5 virtual table (BM25 weights 1/4/2)
```

A single file is replaced transactionally; stale chunk ids are removed from
both virtual tables before the new rows are inserted. WAL mode is enabled.
