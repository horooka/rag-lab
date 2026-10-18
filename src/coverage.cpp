#include "coverage/coverage.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "rag/util.hpp"

namespace coverage {

namespace {

struct GroupVectors {
    GroupSpec spec;
    std::vector<float> centroid;
    std::vector<std::vector<float>> keyword_vectors;
};

double dot(const std::vector<float>& a, const std::vector<float>& b) {
    double value = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        value += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return value;
}

void normalize(std::vector<float>& value) {
    double norm = 0.0;
    for (float component : value) {
        norm += static_cast<double>(component) * component;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0) {
        for (float& component : value) {
            component = static_cast<float>(component / norm);
        }
    }
}

std::vector<float> normalize_copy(std::vector<float> value) {
    normalize(value);
    return value;
}

size_t line_span(const rag::Chunk& chunk) {
    if (chunk.end_line < chunk.start_line || chunk.start_line <= 0) {
        return 0;
    }
    return static_cast<size_t>(chunk.end_line - chunk.start_line + 1);
}

struct LineState {
    int group = -2;  // -2 = not covered, -1 = unassigned
    double similarity = -2.0;
};

void finalize_shares(CoverageResult& result, bool line_unit) {
    const int64_t denominator =
        line_unit ? result.total_lines : result.total_chunks;
    auto set_share = [denominator](GroupStat& stat, int64_t count) {
        stat.share = denominator > 0
                         ? static_cast<double>(count) /
                               static_cast<double>(denominator)
                         : 0.0;
    };
    for (auto& group : result.groups) {
        set_share(group, line_unit ? group.lines : group.chunks);
    }
    set_share(result.unassigned,
              line_unit ? result.unassigned.lines : result.unassigned.chunks);

    std::sort(result.groups.begin(), result.groups.end(),
              [](const GroupStat& a, const GroupStat& b) {
                  if (a.share != b.share) {
                      return a.share > b.share;
                  }
                  return a.label < b.label;
              });
}

}  // namespace

GroupSpec parse_group(const std::string& text) {
    GroupSpec group;
    const size_t separator = text.find('=');
    if (separator == std::string::npos) {
        group.label = rag::trim(text);
        if (!group.label.empty()) {
            group.keywords.push_back(group.label);
        }
    } else {
        group.label = rag::trim(text.substr(0, separator));
        const std::string values = text.substr(separator + 1);
        std::unordered_set<std::string> seen;
        for (const auto& piece : rag::split(values, ',')) {
            const std::string keyword = rag::trim(piece);
            if (!keyword.empty() && seen.insert(keyword).second) {
                group.keywords.push_back(keyword);
            }
        }
    }
    if (group.label.empty()) {
        throw std::invalid_argument("group label must not be empty");
    }
    if (group.keywords.empty()) {
        throw std::invalid_argument("group '" + group.label +
                                    "' has no keywords");
    }
    return group;
}

CoverageResult analyze(const std::vector<rag::ChunkEmbedding>& chunks,
                       const std::vector<GroupSpec>& groups,
                       rag::Embedder& embedder,
                       const CoverageOptions& options) {
    if (groups.empty()) {
        throw std::invalid_argument("at least one group is required");
    }

    std::vector<GroupVectors> vectors;
    vectors.reserve(groups.size());
    int dim = 0;
    for (const auto& group : groups) {
        if (group.keywords.empty()) {
            throw std::invalid_argument("group '" + group.label +
                                        "' has no keywords");
        }
        GroupVectors entry;
        entry.spec = group;
        for (const auto& keyword : group.keywords) {
            auto embedding = normalize_copy(embedder.embed_query(keyword));
            if (dim == 0) {
                dim = static_cast<int>(embedding.size());
            } else if (static_cast<int>(embedding.size()) != dim) {
                throw std::runtime_error("embedding dimension mismatch");
            }
            for (size_t i = 0; i < embedding.size(); ++i) {
                if (entry.centroid.size() <= i) {
                    entry.centroid.resize(embedding.size(), 0.0f);
                }
                entry.centroid[i] += embedding[i];
            }
            entry.keyword_vectors.push_back(std::move(embedding));
        }
        normalize(entry.centroid);
        vectors.push_back(std::move(entry));
    }

    CoverageResult result;
    result.groups.resize(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        result.groups[i].label = groups[i].label;
    }
    result.unassigned.label = "unassigned";
    result.total_chunks = static_cast<int64_t>(chunks.size());

    std::unordered_map<std::string, std::vector<LineState>> lines;
    if (options.line_unit) {
        lines.reserve(64);
    }

    std::vector<double> similarity_sums(groups.size(), 0.0);
    std::vector<std::vector<ChunkMatch>> group_matches(groups.size());
    double unassigned_similarity_sum = 0.0;

    for (const auto& item : chunks) {
        if (item.embedding.empty()) {
            continue;
        }
        if (dim != 0 && static_cast<int>(item.embedding.size()) != dim) {
            throw std::runtime_error("index/model embedding dimension mismatch");
        }

        int best_group = -1;
        double best_similarity = -2.0;
        std::string best_keyword;
        for (size_t g = 0; g < vectors.size(); ++g) {
            double similarity = -2.0;
            std::string keyword;
            if (options.max_similarity) {
                for (size_t k = 0; k < vectors[g].keyword_vectors.size(); ++k) {
                    const double value =
                        dot(item.embedding, vectors[g].keyword_vectors[k]);
                    if (value > similarity) {
                        similarity = value;
                        keyword = vectors[g].spec.keywords[k];
                    }
                }
            } else {
                similarity = dot(item.embedding, vectors[g].centroid);
            }
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_group = static_cast<int>(g);
                best_keyword = std::move(keyword);
            }
        }

        const bool assigned = best_group >= 0 &&
                              best_similarity >= options.threshold;
        if (assigned) {
            ++result.assigned_chunks;
            ++result.groups[best_group].chunks;
            similarity_sums[best_group] += best_similarity;
        } else {
            ++result.unassigned.chunks;
            unassigned_similarity_sum += best_similarity;
            best_group = -1;
        }

        const size_t span = line_span(item.chunk);
        if (options.line_unit) {
            auto& file_lines = lines[item.chunk.path];
            if (file_lines.size() <=
                static_cast<size_t>(item.chunk.end_line)) {
                file_lines.resize(
                    static_cast<size_t>(item.chunk.end_line) + 1);
            }
            const int begin = std::max(1, item.chunk.start_line);
            for (int line = begin; line <= item.chunk.end_line; ++line) {
                LineState& state = file_lines[static_cast<size_t>(line)];
                if (best_similarity > state.similarity) {
                    state.group = best_group;
                    state.similarity = best_similarity;
                }
            }
        } else {
            result.total_lines += static_cast<int64_t>(span);
            if (assigned) {
                result.groups[best_group].lines += static_cast<int64_t>(span);
            } else {
                result.unassigned.lines += static_cast<int64_t>(span);
            }
        }

        if (options.details && assigned) {
            ChunkMatch match;
            match.path = item.chunk.path;
            match.symbol = item.chunk.symbol;
            match.label = groups[best_group].label;
            match.keyword = best_keyword;
            match.start_line = item.chunk.start_line;
            match.end_line = item.chunk.end_line;
            match.similarity = best_similarity;
            group_matches[best_group].push_back(std::move(match));
        }
    }

    if (options.line_unit) {
        for (const auto& [path, file_lines] : lines) {
            (void)path;
            for (const auto& state : file_lines) {
                if (state.group == -2) {
                    continue;
                }
                ++result.total_lines;
                if (state.group >= 0) {
                    ++result.assigned_lines;
                    ++result.groups[static_cast<size_t>(state.group)].lines;
                } else {
                    ++result.unassigned.lines;
                }
            }
        }
    } else {
        result.assigned_lines = 0;
        for (const auto& group : result.groups) {
            result.assigned_lines += group.lines;
        }
    }

    for (size_t g = 0; g < result.groups.size(); ++g) {
        if (result.groups[g].chunks > 0) {
            result.groups[g].mean_similarity =
                similarity_sums[g] / static_cast<double>(result.groups[g].chunks);
        }
    }
    if (result.unassigned.chunks > 0) {
        result.unassigned.mean_similarity =
            unassigned_similarity_sum /
            static_cast<double>(result.unassigned.chunks);
    }

    finalize_shares(result, options.line_unit);

    if (options.details) {
        for (auto& matches : group_matches) {
            std::sort(matches.begin(), matches.end(),
                      [](const ChunkMatch& a, const ChunkMatch& b) {
                          if (a.similarity != b.similarity) {
                              return a.similarity > b.similarity;
                          }
                          return a.path < b.path;
                      });
            if (matches.size() > options.details_limit) {
                matches.resize(options.details_limit);
            }
            for (auto& match : matches) {
                result.matches.push_back(std::move(match));
            }
        }
        std::sort(result.matches.begin(), result.matches.end(),
                  [](const ChunkMatch& a, const ChunkMatch& b) {
                      if (a.similarity != b.similarity) {
                          return a.similarity > b.similarity;
                      }
                      return a.path < b.path;
                  });
    }
    return result;
}

}  // namespace coverage
