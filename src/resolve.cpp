#include "rag/resolve.hpp"

#include <cctype>

#include "rag/util.hpp"

namespace rag {

namespace {

bool contains_case_insensitive(const std::string& text,
                               const std::string& needle) {
    if (needle.empty()) {
        return false;
    }
    return lower_ascii(text).find(lower_ascii(needle)) != std::string::npos;
}

}  // namespace

std::string normalize_symbol(std::string_view symbol) {
    std::string out;
    for (char c : symbol) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            continue;
        }
        if (c == '.') {
            out += "::";
        } else {
            out.push_back(static_cast<char>(
                std::tolower(static_cast<unsigned char>(c))));
        }
    }
    return out;
}

ResolveResult classify_symbol_hits(const std::vector<SearchHit>& hits,
                                   const std::string& symbol) {
    ResolveResult result;
    const std::string target = normalize_symbol(symbol);
    for (const auto& hit : hits) {
        const std::string normalized = normalize_symbol(hit.chunk.symbol);
        const bool is_definition =
            !normalized.empty() &&
            (normalized == target ||
             ends_with(normalized, "::" + target));
        if (is_definition) {
            result.definitions.push_back(hit);
        } else if (contains_case_insensitive(hit.chunk.text, symbol)) {
            result.references.push_back(hit);
        } else {
            result.candidates.push_back(hit);
        }
    }
    return result;
}

}  // namespace rag
