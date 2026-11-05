#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "rag/types.hpp"

namespace rag {

struct ResolveResult {
    std::vector<SearchHit> definitions;
    std::vector<SearchHit> references;
    std::vector<SearchHit> candidates;
};

// Lowercases, removes whitespace and normalizes Pascal/object dots to "::".
std::string normalize_symbol(std::string_view symbol);

// Partitions search hits into exact definitions, text references and other
// candidates. Qualified symbols such as TStation.FillStationsList match the
// target FillStationsList.
ResolveResult classify_symbol_hits(const std::vector<SearchHit>& hits,
                                   const std::string& symbol);

}  // namespace rag
