#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "rag/types.hpp"

namespace rag {

struct FileState {
    int64_t id = 0;
    int64_t mtime_ns = 0;
    int64_t size = 0;
    std::string hash;
};

struct FileRecord {
    std::string path;
    std::string repo;
    std::string lang;
    int64_t mtime_ns = 0;
    int64_t size = 0;
    std::string hash;
    int chunk_count = 0;
};

struct RankedId {
    int64_t id = 0;
    double score = 0.0;
};

class Store {
public:
    virtual ~Store() = default;

    virtual void initialize(int dim) = 0;
    virtual void reset() = 0;

    virtual void begin() = 0;
    virtual void commit() = 0;
    virtual void rollback() = 0;

    virtual std::optional<FileState> file_state(
        const std::string& path) = 0;
    virtual void touch_file(int64_t file_id, int64_t mtime_ns, int64_t size) = 0;
    virtual int64_t upsert_file(const FileRecord& file) = 0;
    virtual void delete_file_chunks(int64_t file_id) = 0;
    virtual void insert_chunk(int64_t file_id, int seq, const Chunk& chunk,
                              const std::vector<float>& embedding) = 0;

    virtual int64_t delete_missing(
        const std::unordered_set<std::string>& paths) = 0;

    virtual std::vector<RankedId> dense_search(
        const std::vector<float>& query, int k,
        const SearchFilters& filters = {}) = 0;
    virtual std::vector<RankedId> bm25_search(
        const std::string& query, int k,
        const SearchFilters& filters = {}) = 0;
    virtual std::vector<Chunk> chunks_by_ids(
        const std::vector<int64_t>& ids) = 0;
    virtual std::vector<ChunkEmbedding> all_chunks() = 0;

    virtual IndexStats stats() = 0;
    virtual std::string meta(const std::string& key,
                             const std::string& fallback = {}) = 0;
    virtual void set_meta(const std::string& key, const std::string& value) = 0;

    virtual void optimize() = 0;
};

std::unique_ptr<Store> open_sqlite_store(const std::filesystem::path& db_path);

}  // namespace rag
