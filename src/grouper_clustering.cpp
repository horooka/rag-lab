#include "grouper/clustering.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace grouper {
namespace {

using Embedding = std::vector<float>;

double dot_product(const Embedding& a, const Embedding& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

double vector_norm(const Embedding& value) {
    return std::sqrt(dot_product(value, value));
}

void normalize_in_place(Embedding& value) {
    const double norm = vector_norm(value);
    if (norm <= 0.0) {
        return;
    }
    for (float& component : value) {
        component = static_cast<float>(component / norm);
    }
}

Embedding normalized_copy(Embedding value) {
    normalize_in_place(value);
    return value;
}

// Fast path for vectors that have already been normalized. Both callers in
// cluster_embeddings pass normalized embeddings.
double normalized_cosine_distance(const Embedding& a, const Embedding& b) {
    double similarity = dot_product(a, b);
    similarity = std::max(-1.0, std::min(1.0, similarity));
    return 1.0 - similarity;
}

void validate_embeddings(const std::vector<Embedding>& embeddings) {
    if (embeddings.empty()) {
        throw std::invalid_argument("cannot cluster an empty embedding list");
    }
    const std::size_t dim = embeddings.front().size();
    if (dim == 0) {
        throw std::invalid_argument("embeddings must not be empty");
    }
    for (std::size_t i = 0; i < embeddings.size(); ++i) {
        if (embeddings[i].size() != dim) {
            throw std::invalid_argument(
                "embedding dimension mismatch at item " + std::to_string(i));
        }
    }
}

struct Merge {
    int keep = 0;
    int drop = 0;
    double distance = 0.0;
};

// Greedy agglomerative merge sequence. Every entry records the two current
// component roots so any prefix can be replayed to reconstruct a partition.
std::vector<Merge> merge_sequence(const std::vector<Embedding>& embeddings,
                                  Linkage linkage) {
    const std::size_t n = embeddings.size();
    std::vector<double> distance(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double value = normalized_cosine_distance(embeddings[i],
                                                            embeddings[j]);
            distance[i * n + j] = value;
            distance[j * n + i] = value;
        }
    }

    std::vector<int> sizes(n, 1);
    std::vector<int> roots(n);
    std::iota(roots.begin(), roots.end(), 0);
    std::vector<char> alive(n, 1);

    std::vector<Merge> merges;
    merges.reserve(n > 0 ? n - 1 : 0);
    std::size_t active = n;

    while (active > 1) {
        double best = std::numeric_limits<double>::infinity();
        std::size_t keep = 0;
        std::size_t drop = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (!alive[i]) {
                continue;
            }
            for (std::size_t j = i + 1; j < n; ++j) {
                if (!alive[j]) {
                    continue;
                }
                const double value = distance[i * n + j];
                if (value < best) {
                    best = value;
                    keep = i;
                    drop = j;
                }
            }
        }
        if (!std::isfinite(best)) {
            break;
        }

        merges.push_back(Merge{roots[keep], roots[drop], best});

        const int keep_size = sizes[keep];
        const int drop_size = sizes[drop];
        for (std::size_t c = 0; c < n; ++c) {
            if (!alive[c] || c == keep || c == drop) {
                continue;
            }
            const double to_keep = distance[keep * n + c];
            const double to_drop = distance[drop * n + c];
            double merged = 0.0;
            switch (linkage) {
                case Linkage::Complete:
                    merged = std::max(to_keep, to_drop);
                    break;
                case Linkage::Single:
                    merged = std::min(to_keep, to_drop);
                    break;
                case Linkage::Average:
                default:
                    merged = (static_cast<double>(keep_size) * to_keep +
                              static_cast<double>(drop_size) * to_drop) /
                             static_cast<double>(keep_size + drop_size);
                    break;
            }
            distance[keep * n + c] = merged;
            distance[c * n + keep] = merged;
        }

        sizes[keep] += sizes[drop];
        alive[drop] = 0;
        --active;
    }
    return merges;
}

// Rebuild the partition produced by the first `count` merges.
std::vector<std::vector<std::size_t>> groups_after(
    std::size_t n, const std::vector<Merge>& merges, std::size_t count) {
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);

    auto find_root = [&parent](int node) {
        int root = node;
        while (parent[root] != root) {
            root = parent[root];
        }
        while (parent[node] != node) {
            const int next = parent[node];
            parent[node] = root;
            node = next;
        }
        return root;
    };

    const std::size_t limit = std::min(count, merges.size());
    for (std::size_t i = 0; i < limit; ++i) {
        const int keep = find_root(merges[i].keep);
        const int drop = find_root(merges[i].drop);
        if (keep != drop) {
            parent[drop] = keep;
        }
    }

    std::unordered_map<int, std::size_t> positions;
    std::vector<std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < n; ++i) {
        const int root = find_root(static_cast<int>(i));
        auto it = positions.find(root);
        if (it == positions.end()) {
            positions.emplace(root, groups.size());
            groups.emplace_back();
            groups.back().push_back(i);
        } else {
            groups[it->second].push_back(i);
        }
    }
    for (auto& group : groups) {
        std::sort(group.begin(), group.end());
    }
    return groups;
}

double silhouette_score(const std::vector<Embedding>& embeddings,
                        const std::vector<std::vector<std::size_t>>& groups) {
    const std::size_t n = embeddings.size();
    if (n == 0 || groups.size() < 2) {
        return -1.0;
    }

    std::vector<std::size_t> group_of(n, 0);
    for (std::size_t g = 0; g < groups.size(); ++g) {
        for (std::size_t index : groups[g]) {
            group_of[index] = g;
        }
    }

    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t own = group_of[i];
        if (groups[own].size() <= 1) {
            // Standard convention: a singleton silhouette is zero.
            continue;
        }

        double a = 0.0;
        for (std::size_t other : groups[own]) {
            if (other != i) {
                a += normalized_cosine_distance(embeddings[i],
                                                embeddings[other]);
            }
        }
        a /= static_cast<double>(groups[own].size() - 1);

        double b = std::numeric_limits<double>::infinity();
        for (std::size_t g = 0; g < groups.size(); ++g) {
            if (g == own || groups[g].empty()) {
                continue;
            }
            double mean = 0.0;
            for (std::size_t other : groups[g]) {
                mean += normalized_cosine_distance(embeddings[i],
                                                   embeddings[other]);
            }
            mean /= static_cast<double>(groups[g].size());
            b = std::min(b, mean);
        }
        if (!std::isfinite(b)) {
            continue;
        }

        const double denominator = std::max(a, b);
        if (denominator > 1e-12) {
            total += (b - a) / denominator;
        }
    }
    return total / static_cast<double>(n);
}

int automatic_group_count(const std::vector<Embedding>& embeddings,
                          const std::vector<Merge>& merges,
                          const ClusterOptions& options) {
    const std::size_t n = embeddings.size();
    if (n <= 1) {
        return 1;
    }
    if (n == 2) {
        // Silhouette is degenerate for two points, so use a semantic pair
        // cutoff: cosine similarity >= 0.70 stays together. This is stricter
        // than the multi-item path because there is no cluster structure to
        // learn from.
        const double distance = normalized_cosine_distance(embeddings[0],
                                                           embeddings[1]);
        return distance <= 0.30 ? 1 : 2;
    }

    std::size_t cap = std::min<std::size_t>(n, 32);
    if (options.max_groups > 0) {
        cap = std::min<std::size_t>(n, static_cast<std::size_t>(
                                            options.max_groups));
    }
    if (cap < 2) {
        return 1;
    }

    int best_k = 1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t k = 2; k <= cap; ++k) {
        const auto groups = groups_after(n, merges, n - k);
        const double score = silhouette_score(embeddings, groups);
        if (score > best_score + 1e-12) {
            best_score = score;
            best_k = static_cast<int>(k);
        }
    }

    // Silhouette happily splits near-identical items into separate groups.
    // Do not split across a boundary that is semantically negligible.
    while (best_k > 1) {
        const std::size_t next_merge = n - static_cast<std::size_t>(best_k);
        if (next_merge >= merges.size()) {
            break;
        }
        if (merges[next_merge].distance >= 0.01) {
            break;
        }
        --best_k;
    }
    return best_k;
}

ClusterResult make_result(const std::vector<Embedding>& embeddings,
                          std::vector<std::vector<std::size_t>> groups,
                          const std::string& mode,
                          const std::vector<Merge>& merges,
                          std::size_t merge_count) {
    ClusterResult result;
    result.mode = mode;
    result.merges = static_cast<int>(merge_count);
    if (merge_count < merges.size()) {
        result.cut_distance = merges[merge_count].distance;
    }

    const std::size_t dim = embeddings.front().size();
    for (auto& members : groups) {
        Cluster cluster;
        std::vector<float> centroid(dim, 0.0f);
        for (std::size_t index : members) {
            const auto& embedding = embeddings[index];
            for (std::size_t d = 0; d < dim; ++d) {
                centroid[d] += embedding[d];
            }
        }
        normalize_in_place(centroid);

        double cohesion = 0.0;
        for (std::size_t index : members) {
            cohesion += dot_product(embeddings[index], centroid);
        }
        cluster.cohesion =
            members.empty()
                ? 0.0
                : cohesion / static_cast<double>(members.size());
        cluster.centroid = std::move(centroid);
        cluster.members = std::move(members);

        std::sort(cluster.members.begin(), cluster.members.end(),
                  [&embeddings, &cluster](std::size_t left,
                                          std::size_t right) {
                      const double left_score =
                          dot_product(embeddings[left], cluster.centroid);
                      const double right_score =
                          dot_product(embeddings[right], cluster.centroid);
                      if (left_score != right_score) {
                          return left_score > right_score;
                      }
                      return left < right;
                  });
        cluster.representative = cluster.members.front();
        result.clusters.push_back(std::move(cluster));
    }

    std::sort(result.clusters.begin(), result.clusters.end(),
              [](const Cluster& left, const Cluster& right) {
                  if (left.members.size() != right.members.size()) {
                      return left.members.size() > right.members.size();
                  }
                  if (left.cohesion != right.cohesion) {
                      return left.cohesion > right.cohesion;
                  }
                  return left.representative < right.representative;
              });
    return result;
}

}  // namespace

double cosine_distance(const std::vector<float>& a,
                       const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("embedding dimension mismatch");
    }
    const double norm_a = vector_norm(a);
    const double norm_b = vector_norm(b);
    if (norm_a <= 0.0 || norm_b <= 0.0) {
        return 1.0;
    }
    double similarity = dot_product(a, b) / (norm_a * norm_b);
    similarity = std::max(-1.0, std::min(1.0, similarity));
    return 1.0 - similarity;
}

const char* linkage_name(Linkage linkage) {
    switch (linkage) {
        case Linkage::Complete:
            return "complete";
        case Linkage::Single:
            return "single";
        case Linkage::Average:
        default:
            return "average";
    }
}

ClusterResult cluster_embeddings(
    const std::vector<std::vector<float>>& embeddings,
    const ClusterOptions& options) {
    if (embeddings.empty()) {
        return ClusterResult{};
    }
    validate_embeddings(embeddings);
    if (options.groups < 0) {
        throw std::invalid_argument("group count must not be negative");
    }
    if (options.groups > 0 && options.threshold >= 0.0) {
        throw std::invalid_argument(
            "--groups and --threshold are mutually exclusive");
    }
    if (options.groups > static_cast<int>(embeddings.size())) {
        throw std::invalid_argument(
            "cannot create more groups than input items");
    }

    std::vector<Embedding> normalized;
    normalized.reserve(embeddings.size());
    for (const auto& embedding : embeddings) {
        normalized.push_back(normalized_copy(embedding));
    }

    const std::size_t n = normalized.size();
    if (n == 1) {
        auto groups = groups_after(n, {}, 0);
        return make_result(normalized, std::move(groups), "single", {}, 0);
    }
    if (options.groups == static_cast<int>(n)) {
        auto groups = groups_after(n, {}, 0);
        return make_result(normalized, std::move(groups), "fixed", {}, 0);
    }

    const auto merges = merge_sequence(normalized, options.linkage);
    std::size_t merge_count = 0;
    std::string mode;

    if (options.groups > 0) {
        merge_count = n - static_cast<std::size_t>(options.groups);
        mode = "fixed";
    } else if (options.threshold >= 0.0) {
        while (merge_count < merges.size() &&
               merges[merge_count].distance <= options.threshold) {
            ++merge_count;
        }
        mode = "threshold";
    } else {
        const int groups = automatic_group_count(normalized, merges, options);
        merge_count = n - static_cast<std::size_t>(groups);
        mode = "auto";
    }

    auto groups = groups_after(n, merges, merge_count);
    return make_result(normalized, std::move(groups), mode, merges,
                       merge_count);
}

}  // namespace grouper
