#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rag {

enum class Lang { Unknown, C, Cpp, Python, Pascal, Luau, Rust, Text };

const char* lang_name(Lang lang);
Lang lang_from_path(const std::string& path);
// Empty -> nullopt (infer from extension). Unknown names also return nullopt;
// the chunker registry turns that into a configuration error.
std::optional<Lang> lang_from_name(std::string_view name);

struct SourceFile {
    std::string abs_path;
    std::string rel_path;
    std::string repo;
    Lang lang = Lang::Unknown;
    int64_t size = 0;
    int64_t mtime_ns = 0;
};

struct RawChunk {
    std::string symbol;
    std::string kind;  // function, method, class, header, window
    Lang lang = Lang::Unknown;  // effective grammar, if overridden
    std::string text;
    int start_line = 0;
    int end_line = 0;
};

struct Chunk {
    int64_t id = 0;
    int64_t file_id = 0;
    std::string path;
    std::string repo;
    std::string lang;
    std::string symbol;
    std::string kind;
    int start_line = 0;
    int end_line = 0;
    int64_t mtime_ns = 0;  // filesystem modification time (Unix ns)
    std::string text;
};

struct ChunkEmbedding {
    Chunk chunk;
    std::vector<float> embedding;
};

struct SearchHit {
    Chunk chunk;
    std::string source;  // workspace index name; empty for single-db search
    double score = 0.0;
    double dense_score = 0.0;
    double bm25_score = 0.0;
    int dense_rank = 0;
    int bm25_rank = 0;
};

struct SearchFilters {
    std::string repo;
    std::string lang;
    std::string path_prefix;
    std::string source;  // workspace index name
    std::optional<int64_t> after_ns;   // inclusive file mtime lower bound
    std::optional<int64_t> before_ns;  // inclusive file mtime upper bound
};

struct IndexStats {
    int64_t files = 0;
    int64_t chunks = 0;
    std::vector<std::pair<std::string, int64_t>> repos;
    std::vector<std::pair<std::string, int64_t>> languages;
    std::string model;
    std::string root;
    int64_t indexed_at = 0;
    uint64_t db_bytes = 0;
};

}  // namespace rag
