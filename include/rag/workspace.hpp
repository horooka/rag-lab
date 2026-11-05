#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "rag/embedder.hpp"
#include "rag/search.hpp"
#include "rag/store.hpp"
#include "rag/types.hpp"

namespace rag {

struct WorkspaceEntry {
    std::string name;
    std::filesystem::path db_path;
    std::unique_ptr<Store> store;
    IndexStats stats;
};

// A named group of independent indexes. Each index keeps its own model,
// chunker and encoding setup; queries are merged at the result level.
class Workspace {
public:
    Workspace() = default;
    Workspace(Workspace&&) = default;
    Workspace& operator=(Workspace&&) = default;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    // JSON shape:
    //   {"model_dir": "...", "indexes": [{"name": "a", "db": "a.db"}]}
    // or {"indexes": {"a": "a.db", "b": "b.db"}}.
    // Relative paths are resolved against the workspace file's directory.
    static Workspace load(const std::filesystem::path& path);

    const std::vector<WorkspaceEntry>& entries() const { return entries_; }
    std::vector<WorkspaceEntry>& entries() { return entries_; }
    const std::filesystem::path& model_dir() const { return model_dir_; }

    // Fails when an index was built with a different model or embedding dim.
    void validate_embedder(const Embedder& embedder) const;

    IndexStats aggregate_stats() const;

private:
    std::filesystem::path path_;
    std::filesystem::path model_dir_;
    std::vector<WorkspaceEntry> entries_;
};

// Runs the query against every index and merges the ranked lists. Each hit is
// tagged with its WorkspaceEntry::name. Requires a single embedder shared by
// all indexes.
SearchResult search_workspace(Workspace& workspace, Embedder& embedder,
                              const SearchOptions& options,
                              const std::string& query);

}  // namespace rag
