#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rag/embedder.hpp"
#include "rag/types.hpp"

namespace coverage {

struct GroupSpec {
    std::string label;
    std::vector<std::string> keywords;
};

struct CoverageOptions {
    // Similarity cutoff for assignment. Everything below it is "unassigned".
    double threshold = 0.0;
    // Count de-duplicated source lines instead of chunks. Slower and more
    // memory-hungry, but gives true line coverage across overlapping chunks.
    bool line_unit = false;
    // "max" compares a chunk against every keyword and takes the best one;
    // "centroid" compares against the normalized keyword centroid.
    bool max_similarity = true;
    bool details = false;
    std::size_t details_limit = 5;
};

struct GroupStat {
    std::string label;
    int64_t chunks = 0;
    int64_t lines = 0;
    double share = 0.0;
    double mean_similarity = 0.0;
};

struct ChunkMatch {
    std::string path;
    std::string symbol;
    std::string label;
    std::string keyword;
    int start_line = 0;
    int end_line = 0;
    double similarity = 0.0;
};

struct CoverageResult {
    std::vector<GroupStat> groups;  // sorted by share, descending
    GroupStat unassigned;
    int64_t total_chunks = 0;
    int64_t total_lines = 0;
    int64_t assigned_chunks = 0;
    int64_t assigned_lines = 0;
    std::vector<ChunkMatch> matches;  // populated when options.details
};

// Builds group embeddings and assigns chunks. `embedder` is used only for the
// group keywords; chunk embeddings are read from the index.
CoverageResult analyze(const std::vector<rag::ChunkEmbedding>& chunks,
                       const std::vector<GroupSpec>& groups,
                       rag::Embedder& embedder,
                       const CoverageOptions& options);

// Parses "label=keyword1,keyword2" and falls back to a single-keyword group
// when no '=' is present.
GroupSpec parse_group(const std::string& text);

}  // namespace coverage
