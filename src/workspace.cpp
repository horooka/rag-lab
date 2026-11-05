#include "rag/workspace.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "rag/util.hpp"

namespace rag {

namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

std::filesystem::path resolve_path(const std::filesystem::path& base,
                                   const std::string& value) {
    std::filesystem::path path(value);
    if (path.is_absolute()) {
        return path;
    }
    return std::filesystem::absolute(base / path);
}

std::string default_name(const std::filesystem::path& db_path) {
    std::string name = db_path.stem().string();
    if (name.empty()) {
        name = "index";
    }
    return name;
}

}  // namespace

Workspace Workspace::load(const std::filesystem::path& path) {
    const std::filesystem::path absolute_path =
        std::filesystem::absolute(path);
    if (!std::filesystem::exists(absolute_path)) {
        throw std::runtime_error("workspace file not found: " +
                                 absolute_path.string());
    }
    const nlohmann::json json =
        nlohmann::json::parse(read_file(absolute_path));
    const std::filesystem::path base = absolute_path.parent_path();

    Workspace workspace;
    workspace.path_ = absolute_path;
    if (json.is_object() && json.contains("model_dir")) {
        workspace.model_dir_ =
            resolve_path(base, json["model_dir"].get<std::string>());
    }

    auto add_entry = [&workspace](const std::string& name,
                                  const std::string& db) {
        WorkspaceEntry entry;
        entry.name = name.empty() ? default_name(db) : name;
        entry.db_path = db;
        workspace.entries_.push_back(std::move(entry));
    };

    if (json.is_array()) {
        for (const auto& item : json) {
            if (!item.is_object() || !item.contains("db")) {
                throw std::runtime_error(
                    "workspace array entries need a 'db' field");
            }
            add_entry(item.value("name", ""), item["db"].get<std::string>());
        }
    } else if (json.is_object() && json.contains("indexes")) {
        const auto& indexes = json["indexes"];
        if (indexes.is_array()) {
            for (const auto& item : indexes) {
                if (!item.is_object() || !item.contains("db")) {
                    throw std::runtime_error(
                        "workspace index entries need a 'db' field");
                }
                add_entry(item.value("name", ""),
                          item["db"].get<std::string>());
            }
        } else if (indexes.is_object()) {
            for (auto it = indexes.begin(); it != indexes.end(); ++it) {
                if (!it.value().is_string()) {
                    throw std::runtime_error(
                        "workspace index map values must be database paths");
                }
                add_entry(it.key(), it.value().get<std::string>());
            }
        } else {
            throw std::runtime_error(
                "workspace 'indexes' must be an array or object");
        }
    } else {
        throw std::runtime_error(
            "workspace file needs an 'indexes' array/object");
    }

    if (workspace.entries_.empty()) {
        throw std::runtime_error("workspace has no indexes");
    }

    std::unordered_set<std::string> names;
    for (auto& entry : workspace.entries_) {
        entry.db_path = resolve_path(base, entry.db_path.string());
        if (!std::filesystem::exists(entry.db_path)) {
            throw std::runtime_error("workspace index not found: " +
                                     entry.db_path.string() + " (" +
                                     entry.name + ")");
        }
        if (!names.insert(entry.name).second) {
            throw std::runtime_error("duplicate workspace index name: " +
                                     entry.name);
        }
        entry.store = open_sqlite_store(entry.db_path);
        entry.stats = entry.store->stats();
    }
    return workspace;
}

void Workspace::validate_embedder(const Embedder& embedder) const {
    for (const auto& entry : entries_) {
        if (!entry.stats.model.empty() &&
            entry.stats.model != embedder.spec().name) {
            throw std::runtime_error(
                "workspace index '" + entry.name + "' uses model '" +
                entry.stats.model + "', current model is '" +
                embedder.spec().name + "'");
        }
        const std::string dim = entry.store->meta("dim");
        if (!dim.empty() && std::stoi(dim) != embedder.dim()) {
            throw std::runtime_error(
                "workspace index '" + entry.name + "' has dim=" + dim +
                ", current model has dim=" +
                std::to_string(embedder.dim()));
        }
    }
}

IndexStats Workspace::aggregate_stats() const {
    IndexStats total;
    total.root = "workspace";
    total.model = entries_.empty() ? "" : entries_.front().stats.model;
    for (const auto& entry : entries_) {
        total.files += entry.stats.files;
        total.chunks += entry.stats.chunks;
        total.db_bytes += entry.stats.db_bytes;
        total.indexed_at =
            std::max(total.indexed_at, entry.stats.indexed_at);
        if (entry.stats.model != total.model) {
            total.model = "mixed";
        }
        total.repos.insert(total.repos.end(), entry.stats.repos.begin(),
                           entry.stats.repos.end());
        total.languages.insert(total.languages.end(),
                               entry.stats.languages.begin(),
                               entry.stats.languages.end());
    }
    return total;
}

SearchResult search_workspace(Workspace& workspace, Embedder& embedder,
                              const SearchOptions& options,
                              const std::string& query) {
    SearchResult result;
    const auto total_start = std::chrono::steady_clock::now();
    if (trim(query).empty()) {
        result.total_ms = elapsed_ms(total_start);
        return result;
    }
    const bool use_dense = options.mode != "bm25";
    if (use_dense) {
        workspace.validate_embedder(embedder);
    }
    if (!options.filters.source.empty()) {
        bool found = false;
        for (const auto& entry : workspace.entries()) {
            if (entry.name == options.filters.source) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("workspace has no index named: " +
                                     options.filters.source);
        }
    }
    std::vector<float> embedding;
    if (use_dense) {
        const auto start = std::chrono::steady_clock::now();
        embedding = embedder.embed_query(query);
        result.embed_ms = elapsed_ms(start);
    }

    for (auto& entry : workspace.entries()) {
        if (!options.filters.source.empty() &&
            entry.name != options.filters.source) {
            continue;
        }
        Searcher searcher(*entry.store, embedder, options);
        SearchResult one =
            searcher.search_with_embedding(query, embedding);
        result.dense_ms += one.dense_ms;
        result.bm25_ms += one.bm25_ms;
        for (auto& hit : one.hits) {
            hit.source = entry.name;
            result.hits.push_back(std::move(hit));
        }
    }

    std::sort(result.hits.begin(), result.hits.end(),
              [](const SearchHit& a, const SearchHit& b) {
                  if (a.score != b.score) {
                      return a.score > b.score;
                  }
                  if (a.source != b.source) {
                      return a.source < b.source;
                  }
                  if (a.chunk.path != b.chunk.path) {
                      return a.chunk.path < b.chunk.path;
                  }
                  return a.chunk.start_line < b.chunk.start_line;
              });

    std::vector<SearchHit> unique;
    unique.reserve(result.hits.size());
    std::unordered_set<std::string> seen;
    for (auto& hit : result.hits) {
        const std::string key =
            hit.source + "\n" + hit.chunk.path + "\n" +
            std::to_string(hit.chunk.start_line) + "-" +
            std::to_string(hit.chunk.end_line);
        if (!seen.insert(key).second) {
            continue;
        }
        unique.push_back(std::move(hit));
        if (static_cast<int>(unique.size()) >= options.k) {
            break;
        }
    }
    result.hits = std::move(unique);
    result.total_ms = elapsed_ms(total_start);
    return result;
}

}  // namespace rag
