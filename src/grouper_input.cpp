#include "grouper/input.hpp"

#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace grouper {
namespace {

std::string trim_copy(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end &&
           std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

void add_item(const std::string& raw, const InputOptions& options,
              std::unordered_set<std::string>& seen,
              std::vector<std::string>& items) {
    std::string item = trim_copy(raw);
    if (item.empty()) {
        return;
    }
    if (options.unique && !seen.insert(item).second) {
        return;
    }
    items.push_back(std::move(item));
}

void add_line(std::string_view line, const InputOptions& options,
              std::unordered_set<std::string>& seen,
              std::vector<std::string>& items) {
    if (options.delimiter == '\0') {
        add_item(std::string(line), options, seen, items);
        return;
    }
    std::size_t start = 0;
    while (start <= line.size()) {
        const std::size_t position = line.find(options.delimiter, start);
        const std::string_view piece =
            position == std::string_view::npos
                ? line.substr(start)
                : line.substr(start, position - start);
        add_item(std::string(piece), options, seen, items);
        if (position == std::string_view::npos) {
            break;
        }
        start = position + 1;
    }
}

void read_stream(std::istream& stream, const InputOptions& options,
                 std::unordered_set<std::string>& seen,
                 std::vector<std::string>& items) {
    std::string line;
    while (std::getline(stream, line)) {
        add_line(line, options, seen, items);
    }
}

}  // namespace

std::vector<std::string> read_inputs(
    const std::vector<std::string>& positional,
    const std::vector<std::string>& files,
    const InputOptions& options) {
    std::unordered_set<std::string> seen;
    std::vector<std::string> items;

    for (const auto& item : positional) {
        add_line(item, options, seen, items);
    }

    if (files.empty() && positional.empty()) {
        read_stream(std::cin, options, seen, items);
        return items;
    }

    for (const auto& file : files) {
        if (file == "-") {
            read_stream(std::cin, options, seen, items);
            continue;
        }
        std::ifstream input(file);
        if (!input) {
            throw std::runtime_error("cannot open input file: " + file);
        }
        read_stream(input, options, seen, items);
    }
    return items;
}

}  // namespace grouper
