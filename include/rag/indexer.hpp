#pragma once

#include <cstdint>
#include <string>

#include "rag/chunker.hpp"
#include "rag/config.hpp"
#include "rag/embedder.hpp"
#include "rag/source.hpp"
#include "rag/store.hpp"

namespace rag {

struct IndexResult {
    int64_t scanned = 0;
    int64_t added = 0;
    int64_t updated = 0;
    int64_t unchanged = 0;
    int64_t removed = 0;
    int64_t chunks_written = 0;
    int64_t failed = 0;
    double seconds = 0.0;
};

class Indexer {
public:
    Indexer(const Config& config, Source& source, Chunker& chunker,
            Embedder& embedder, Store& store);

    IndexResult run(bool full, bool quiet = false);

private:
    std::string embed_text(const SourceFile& file, const RawChunk& chunk) const;

    Config config_;
    Source& source_;
    Chunker& chunker_;
    Embedder& embedder_;
    Store& store_;
};

std::string make_embed_text(const std::string& rel_path,
                            const RawChunk& chunk);

}  // namespace rag
