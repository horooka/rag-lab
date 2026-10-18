#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace grouper {

enum class Linkage {
    Average,
    Complete,
    Single,
};

struct ClusterOptions {
    // Exact number of groups. 0 means choose automatically.
    int groups = 0;

    // Cosine-distance cutoff. When >= 0, clusters are formed by merging
    // while the next merge distance is at most this value.
    // Mutually exclusive with `groups`.
    double threshold = -1.0;

    // Upper bound for automatic group selection. 0 means min(item_count, 32).
    int max_groups = 0;

    Linkage linkage = Linkage::Average;
};

struct Cluster {
    // Indices into the input embedding/item vector, ordered by similarity to
    // the centroid (best first).
    std::vector<std::size_t> members;

    // Normalized centroid.
    std::vector<float> centroid;

    // Mean cosine similarity of members to the centroid.
    double cohesion = 0.0;

    // Member closest to the centroid. Useful as a human-readable label.
    std::size_t representative = 0;
};

struct ClusterResult {
    std::vector<Cluster> clusters;
    std::string mode;  // fixed, threshold, auto or single
    int merges = 0;
    double cut_distance = 0.0;
};

// Groups normalized or unnormalized embeddings using agglomerative clustering
// over cosine distance. Members and clusters are returned in deterministic
// order.
ClusterResult cluster_embeddings(
    const std::vector<std::vector<float>>& embeddings,
    const ClusterOptions& options);

// Cosine distance in [0, 2]. Zero-length vectors have distance 1.
double cosine_distance(const std::vector<float>& a,
                       const std::vector<float>& b);

const char* linkage_name(Linkage linkage);

}  // namespace grouper
