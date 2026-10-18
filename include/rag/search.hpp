#pragma once

#include <string>
#include <vector>

#include "rag/config.hpp"
#include "rag/embedder.hpp"
#include "rag/store.hpp"

namespace rag {

struct SearchOptions {
    std::string mode = "hybrid";
    int k = 10;
    int candidate_k = 50;
    double dense_weight = 1.0;
    double bm25_weight = 0.10;
    bool auto_weights = true;
    SearchFilters filters;
};

struct SearchResult {
    std::vector<SearchHit> hits;
    double embed_ms = 0.0;
    double dense_ms = 0.0;
    double bm25_ms = 0.0;
    double total_ms = 0.0;
};

class Searcher {
public:
    Searcher(Store& store, Embedder& embedder, SearchOptions options);

    SearchResult search(const std::string& query);
    SearchOptions& options() { return options_; }

    // Reciprocal-rank fusion. Ranks are 1-based; missing ranks are ignored.
    static std::vector<std::pair<int64_t, double>> rrf_fuse(
        const std::vector<RankedId>& dense, const std::vector<RankedId>& bm25,
        double k = 60.0, double dense_weight = 1.0,
        double bm25_weight = 1.0);

private:
    Store& store_;
    Embedder& embedder_;
    SearchOptions options_;
};

bool chunk_matches(const Chunk& chunk, const SearchFilters& filters);

// True for queries that look like a symbol, path or identifier rather than
// natural language. Such queries get a stronger BM25 contribution.
bool looks_like_identifier(const std::string& query);

}  // namespace rag
