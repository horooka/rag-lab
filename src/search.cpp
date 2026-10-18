#include "rag/search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>

#include "rag/util.hpp"

namespace rag {

namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

}  // namespace

Searcher::Searcher(Store& store, Embedder& embedder, SearchOptions options)
    : store_(store), embedder_(embedder), options_(std::move(options)) {}

SearchResult Searcher::search(const std::string& query) {
    SearchResult result;
    const auto total_start = std::chrono::steady_clock::now();
    if (trim(query).empty()) {
        result.total_ms = elapsed_ms(total_start);
        return result;
    }

    const bool use_dense = options_.mode != "bm25";
    const bool use_bm25 = options_.mode != "dense";
    const bool has_filters = !options_.filters.repo.empty() ||
                             !options_.filters.lang.empty() ||
                             !options_.filters.path_prefix.empty() ||
                             options_.filters.after_ns.has_value() ||
                             options_.filters.before_ns.has_value();
    const int candidate_k =
        std::max(options_.candidate_k, options_.k) * (has_filters ? 5 : 1);

    std::vector<float> query_embedding;
    if (use_dense) {
        const auto embed_start = std::chrono::steady_clock::now();
        query_embedding = embedder_.embed_query(query);
        result.embed_ms = elapsed_ms(embed_start);
    }

    std::vector<RankedId> dense;
    if (use_dense) {
        const auto start = std::chrono::steady_clock::now();
        dense = store_.dense_search(query_embedding, candidate_k,
                                   options_.filters);
        result.dense_ms = elapsed_ms(start);
    }

    std::vector<RankedId> bm25;
    if (use_bm25) {
        const auto start = std::chrono::steady_clock::now();
        bm25 = store_.bm25_search(query, candidate_k, options_.filters);
        result.bm25_ms = elapsed_ms(start);
    }

    double dense_weight = options_.dense_weight;
    double bm25_weight = options_.bm25_weight;
    if (options_.auto_weights) {
        if (looks_like_identifier(query)) {
            dense_weight = 0.3;
            bm25_weight = 1.0;
        } else {
            dense_weight = 1.0;
            bm25_weight = 0.02;
        }
    }

    std::vector<std::pair<int64_t, double>> fused;
    if (use_dense && use_bm25) {
        fused = rrf_fuse(dense, bm25, 60.0, dense_weight, bm25_weight);
    } else if (use_dense) {
        for (size_t i = 0; i < dense.size(); ++i) {
            fused.emplace_back(dense[i].id, 1.0 / (60.0 + i + 1.0));
        }
    } else {
        for (size_t i = 0; i < bm25.size(); ++i) {
            fused.emplace_back(bm25[i].id, 1.0 / (60.0 + i + 1.0));
        }
    }

    std::unordered_map<int64_t, int> dense_rank;
    std::unordered_map<int64_t, double> dense_score;
    for (size_t i = 0; i < dense.size(); ++i) {
        dense_rank[dense[i].id] = static_cast<int>(i + 1);
        dense_score[dense[i].id] = dense[i].score;
    }
    std::unordered_map<int64_t, int> bm25_rank;
    std::unordered_map<int64_t, double> bm25_score;
    for (size_t i = 0; i < bm25.size(); ++i) {
        bm25_rank[bm25[i].id] = static_cast<int>(i + 1);
        bm25_score[bm25[i].id] = bm25[i].score;
    }

    std::vector<int64_t> ids;
    ids.reserve(fused.size());
    for (const auto& [id, score] : fused) {
        (void)score;
        ids.push_back(id);
    }
    auto chunks = store_.chunks_by_ids(ids);
    std::unordered_map<int64_t, Chunk> by_id;
    by_id.reserve(chunks.size());
    for (auto& chunk : chunks) {
        by_id.emplace(chunk.id, std::move(chunk));
    }

    for (const auto& [id, score] : fused) {
        auto it = by_id.find(id);
        if (it == by_id.end()) {
            continue;
        }
        if (!chunk_matches(it->second, options_.filters)) {
            continue;
        }
        SearchHit hit;
        hit.chunk = it->second;
        hit.score = score;
        auto dense_it = dense_score.find(id);
        if (dense_it != dense_score.end()) {
            hit.dense_score = dense_it->second;
        }
        auto bm25_it = bm25_score.find(id);
        if (bm25_it != bm25_score.end()) {
            hit.bm25_score = bm25_it->second;
        }
        auto dr = dense_rank.find(id);
        if (dr != dense_rank.end()) {
            hit.dense_rank = dr->second;
        }
        auto br = bm25_rank.find(id);
        if (br != bm25_rank.end()) {
            hit.bm25_rank = br->second;
        }
        result.hits.push_back(std::move(hit));
        if (static_cast<int>(result.hits.size()) >= options_.k) {
            break;
        }
    }

    result.total_ms = elapsed_ms(total_start);
    return result;
}

std::vector<std::pair<int64_t, double>> Searcher::rrf_fuse(
    const std::vector<RankedId>& dense, const std::vector<RankedId>& bm25,
    double k, double dense_weight, double bm25_weight) {
    std::unordered_map<int64_t, double> scores;
    scores.reserve(dense.size() + bm25.size());
    for (size_t i = 0; i < dense.size(); ++i) {
        scores[dense[i].id] +=
            dense_weight / (k + static_cast<double>(i + 1));
    }
    for (size_t i = 0; i < bm25.size(); ++i) {
        scores[bm25[i].id] += bm25_weight / (k + static_cast<double>(i + 1));
    }
    std::vector<std::pair<int64_t, double>> fused;
    fused.reserve(scores.size());
    for (const auto& [id, score] : scores) {
        fused.emplace_back(id, score);
    }
    std::sort(fused.begin(), fused.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) {
                      return a.second > b.second;
                  }
                  return a.first < b.first;
              });
    return fused;
}

bool looks_like_identifier(const std::string& query) {
    std::string text = trim(query);
    while (!text.empty() &&
           (text.front() == '"' || text.front() == '`' || text.front() == '\'')) {
        text.erase(text.begin());
    }
    while (!text.empty() &&
           (text.back() == '"' || text.back() == '`' || text.back() == '\'')) {
        text.pop_back();
    }
    if (text.empty()) {
        return false;
    }
    auto parts = split(text, ' ');
    for (const auto& part : parts) {
        if (part.empty()) {
            continue;
        }
        if (part.find('_') != std::string::npos ||
            part.find("::") != std::string::npos ||
            part.find('.') != std::string::npos ||
            part.find("->") != std::string::npos) {
            return true;
        }
        for (size_t i = 1; i < part.size(); ++i) {
            if (part[i - 1] >= 'a' && part[i - 1] <= 'z' &&
                part[i] >= 'A' && part[i] <= 'Z') {
                return true;
            }
        }
    }
    return text.find(' ') == std::string::npos;
}

bool chunk_matches(const Chunk& chunk, const SearchFilters& filters) {
    if (!filters.repo.empty() && chunk.repo != filters.repo) {
        return false;
    }
    if (!filters.lang.empty() && chunk.lang != filters.lang) {
        return false;
    }
    if (!filters.path_prefix.empty() &&
        !starts_with(chunk.path, filters.path_prefix)) {
        return false;
    }
    if (filters.after_ns.has_value() &&
        chunk.mtime_ns < *filters.after_ns) {
        return false;
    }
    if (filters.before_ns.has_value() &&
        chunk.mtime_ns > *filters.before_ns) {
        return false;
    }
    return true;
}

}  // namespace rag
