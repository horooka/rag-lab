#include "rag/eval.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "rag/util.hpp"

namespace rag {

namespace {

bool path_matches(const std::string& reference, const std::string& path) {
    if (reference == path) {
        return true;
    }
    if (ends_with(path, "/" + reference)) {
        return true;
    }
    if (ends_with(reference, "/" + path)) {
        return true;
    }
    return false;
}

}  // namespace

std::vector<EvalQuery> load_eval_queries(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open eval queries: " + path);
    }
    std::vector<EvalQuery> queries;
    std::string line;
    int line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        try {
            nlohmann::json json = nlohmann::json::parse(trimmed);
            EvalQuery query;
            query.query = json.at("query").get<std::string>();
            for (const auto& item : json.at("relevant")) {
                query.relevant.push_back(item.get<std::string>());
            }
            if (query.relevant.empty()) {
                throw std::runtime_error("empty relevant list");
            }
            queries.push_back(std::move(query));
        } catch (const std::exception& error) {
            throw std::runtime_error(path + ":" + std::to_string(line_number) +
                                     ": " + error.what());
        }
    }
    return queries;
}

bool reference_matches(const std::string& reference, const Chunk& chunk) {
    const std::string ref = reference;
    const size_t separator = ref.find("::");
    if (separator == std::string::npos) {
        return path_matches(ref, chunk.path);
    }
    const std::string path = ref.substr(0, separator);
    const std::string symbol = ref.substr(separator + 2);
    if (!path_matches(path, chunk.path)) {
        return false;
    }
    if (symbol.empty()) {
        return true;
    }
    if (chunk.symbol == symbol) {
        return true;
    }
    if (ends_with(chunk.symbol, "::" + symbol)) {
        return true;
    }
    return chunk.symbol.find(symbol) != std::string::npos;
}

EvalSummary evaluate(Searcher& searcher, const std::vector<EvalQuery>& queries,
                     int k) {
    EvalSummary summary;
    summary.queries = static_cast<int>(queries.size());
    summary.k = k;
    summary.mode = searcher.options().mode;

    for (const auto& query : queries) {
        auto result = searcher.search(query.query);
        EvalQueryResult detail;
        detail.query = query.query;

        std::set<size_t> matched_references;
        int first_rank = 0;
        for (size_t rank = 0; rank < result.hits.size(); ++rank) {
            const auto& hit = result.hits[rank];
            detail.retrieved.push_back(hit.chunk.path + "::" + hit.chunk.symbol);
            bool relevant = false;
            for (size_t r = 0; r < query.relevant.size(); ++r) {
                if (reference_matches(query.relevant[r], hit.chunk)) {
                    matched_references.insert(r);
                    relevant = true;
                }
            }
            if (relevant && first_rank == 0) {
                first_rank = static_cast<int>(rank) + 1;
            }
        }

        const double possible =
            static_cast<double>(std::min<size_t>(query.relevant.size(), k));
        detail.recall =
            possible > 0.0
                ? static_cast<double>(matched_references.size()) / possible
                : 0.0;
        detail.first_relevant_rank = first_rank;
        detail.reciprocal_rank = first_rank > 0 ? 1.0 / first_rank : 0.0;
        summary.recall += detail.recall;
        summary.mrr += detail.reciprocal_rank;
        if (first_rank > 0) {
            summary.hit_rate += 1.0;
        }
        summary.details.push_back(std::move(detail));
    }

    if (summary.queries > 0) {
        summary.recall /= summary.queries;
        summary.mrr /= summary.queries;
        summary.hit_rate /= summary.queries;
    }
    return summary;
}

}  // namespace rag
