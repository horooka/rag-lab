#include "rag/chunker.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "rag/util.hpp"

namespace rag {

namespace {

std::vector<std::string> split_lines(std::string_view text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find('\n', start);
        if (end == std::string_view::npos) {
            lines.emplace_back(text.substr(start));
            break;
        }
        lines.emplace_back(text.substr(start, end - start));
        start = end + 1;
    }
    return lines;
}

std::string join_lines(const std::vector<std::string>& lines, size_t begin,
                       size_t end) {
    std::string out;
    for (size_t i = begin; i < end && i < lines.size(); ++i) {
        if (!out.empty()) {
            out.push_back('\n');
        }
        out += lines[i];
    }
    return out;
}

struct Heading {
    size_t line = 0;
    int level = 0;
    std::string title;
};

bool heading_at(const std::string& line, Heading& heading) {
    size_t position = 0;
    int spaces = 0;
    while (position < line.size() &&
           (line[position] == ' ' || line[position] == '\t') && spaces < 4) {
        ++position;
        ++spaces;
    }
    if (position >= line.size() || line[position] != '#') {
        return false;
    }
    int level = 0;
    while (position + static_cast<size_t>(level) < line.size() &&
           line[position + static_cast<size_t>(level)] == '#') {
        ++level;
    }
    if (level < 1 || level > 6) {
        return false;
    }
    const size_t after = position + static_cast<size_t>(level);
    if (after < line.size() && line[after] != ' ' && line[after] != '\t') {
        return false;
    }
    heading.level = level;
    heading.title = trim(line.substr(after));
    return !heading.title.empty();
}

bool fence_marker(const std::string& line, char& marker) {
    size_t position = line.find_first_not_of(" \t");
    if (position == std::string::npos) {
        return false;
    }
    if (line.compare(position, 3, "```") == 0) {
        marker = '`';
        return true;
    }
    if (line.compare(position, 3, "~~~") == 0) {
        marker = '~';
        return true;
    }
    return false;
}

std::string heading_path(
    const std::vector<std::pair<int, std::string>>& stack) {
    std::string path;
    for (const auto& [level, title] : stack) {
        (void)level;
        if (!path.empty()) {
            path += " > ";
        }
        path += title;
    }
    return path;
}

}  // namespace

MarkdownChunker::MarkdownChunker(ChunkOptions options)
    : options_(std::move(options)) {}

std::vector<RawChunk> MarkdownChunker::chunk(const SourceFile& file,
                                             std::string_view content) {
    auto lines = split_lines(content);
    if (lines.empty()) {
        return {};
    }

    std::vector<Heading> headings;
    bool in_fence = false;
    char fence = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        char marker = 0;
        if (fence_marker(lines[i], marker)) {
            if (!in_fence) {
                in_fence = true;
                fence = marker;
            } else if (marker == fence) {
                in_fence = false;
            }
            continue;
        }
        if (in_fence) {
            continue;
        }
        Heading heading;
        if (heading_at(lines[i], heading)) {
            heading.line = i;
            headings.push_back(std::move(heading));
        }
    }

    if (headings.empty()) {
        return chunk_lines(content, options_.window_lines,
                           options_.overlap_lines, file.rel_path);
    }

    std::vector<RawChunk> chunks;
    std::vector<std::pair<int, std::string>> stack;
    const size_t max_chars =
        static_cast<size_t>(std::max(64, options_.max_tokens * 4));

    auto emit_section = [&](size_t begin, size_t end,
                            const std::string& symbol,
                            const std::string& kind) {
        std::string text = trim(join_lines(lines, begin, end));
        if (text.size() < 8) {
            return;
        }
        const int start_line = static_cast<int>(begin) + 1;
        const int end_line = static_cast<int>(end);
        if (text.size() <= max_chars) {
            RawChunk chunk;
            chunk.symbol = symbol;
            chunk.kind = kind;
            chunk.text = std::move(text);
            chunk.start_line = start_line;
            chunk.end_line = end_line;
            chunks.push_back(std::move(chunk));
            return;
        }
        auto windows = chunk_lines(text, options_.window_lines,
                                   options_.overlap_lines, symbol);
        for (auto& window : windows) {
            window.start_line += start_line - 1;
            window.end_line += start_line - 1;
            window.kind = kind;
            chunks.push_back(std::move(window));
        }
    };

    size_t section_begin = 0;
    if (headings.front().line > 0) {
        emit_section(0, headings.front().line, file.rel_path,
                     "markdown-preamble");
    }

    for (size_t i = 0; i < headings.size(); ++i) {
        const Heading& heading = headings[i];
        while (!stack.empty() && stack.back().first >= heading.level) {
            stack.pop_back();
        }
        stack.emplace_back(heading.level, heading.title);
        const size_t section_end =
            i + 1 < headings.size() ? headings[i + 1].line : lines.size();
        emit_section(heading.line, section_end, heading_path(stack),
                     "markdown");
        section_begin = section_end;
    }
    (void)section_begin;

    if (chunks.empty()) {
        return chunk_lines(content, options_.window_lines,
                           options_.overlap_lines, file.rel_path);
    }
    return chunks;
}

}  // namespace rag
