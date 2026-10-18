#pragma once

#include <string>
#include <vector>

#include "rag/search.hpp"

namespace rag {

struct EvalQuery {
    std::string query;
    std::vector<std::string> relevant;
};

struct EvalQueryResult {
    std::string query;
    double recall = 0.0;
    double reciprocal_rank = 0.0;
    int first_relevant_rank = 0;
    std::vector<std::string> retrieved;
};

struct EvalSummary {
    int queries = 0;
    int k = 10;
    std::string mode;
    double recall = 0.0;
    double mrr = 0.0;
    double hit_rate = 0.0;
    std::vector<EvalQueryResult> details;
};

std::vector<EvalQuery> load_eval_queries(const std::string& path);

EvalSummary evaluate(Searcher& searcher, const std::vector<EvalQuery>& queries,
                     int k);

// Matches a retrieved path/symbol against a ground-truth reference like
// "repo/src/file.cpp", "src/file.cpp", or "src/file.cpp::symbol".
bool reference_matches(const std::string& reference, const Chunk& chunk);

}  // namespace rag
