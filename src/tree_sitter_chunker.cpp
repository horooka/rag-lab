#include "rag/chunker.hpp"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <cctype>

#include <tree_sitter/api.h>

#include "tree-sitter-c.h"
#include "tree-sitter-cpp.h"
#include "tree-sitter-pascal.h"
#include "tree-sitter-python.h"
#include "rag/util.hpp"

namespace rag {

namespace {

const TSLanguage* language_for(Lang lang) {
    switch (lang) {
        case Lang::C:
            return tree_sitter_c();
        case Lang::Cpp:
            return tree_sitter_cpp();
        case Lang::Python:
            return tree_sitter_python();
        case Lang::Pascal:
            return tree_sitter_pascal();
        default:
            return nullptr;
    }
}

bool is_unit_type(Lang lang, const char* type) {
    const std::string_view t(type);
    if (lang == Lang::Python) {
        return t == "function_definition" || t == "class_definition" ||
               t == "decorated_definition";
    }
    if (lang == Lang::C) {
        return t == "function_definition" || t == "struct_specifier" ||
               t == "union_specifier" || t == "enum_specifier" ||
               t == "type_definition";
    }
    if (lang == Lang::Cpp) {
        return t == "function_definition" || t == "class_specifier" ||
               t == "struct_specifier" || t == "union_specifier" ||
               t == "enum_specifier" || t == "template_declaration" ||
               t == "type_definition";
    }
    if (lang == Lang::Pascal) {
        return t == "defProc" || t == "declProc" || t == "declType" ||
               t == "declClass" || t == "declIntf" || t == "declConst" ||
               t == "declUses";
    }
    return false;
}

bool is_comment_type(const char* type) {
    const std::string_view t(type);
    return t == "comment" || t == "line_comment" || t == "block_comment";
}

std::string node_text(std::string_view content, TSNode node) {
    const uint32_t start = ts_node_start_byte(node);
    const uint32_t end = ts_node_end_byte(node);
    if (start >= content.size() || end > content.size() || end <= start) {
        return {};
    }
    return std::string(content.substr(start, end - start));
}


std::string declarator_name(std::string_view content, TSNode node,
                            int depth = 0) {
    if (ts_node_is_null(node) || depth > 12) {
        return {};
    }
    const std::string_view type(ts_node_type(node));
    if (type == "identifier" || type == "field_identifier" ||
        type == "type_identifier" || type == "qualified_identifier" ||
        type == "destructor_name" || type == "operator_name" ||
        type == "namespace_identifier") {
        return collapse_ws(node_text(content, node));
    }
    TSNode inner = ts_node_child_by_field_name(node, "declarator", 10);
    if (!ts_node_is_null(inner)) {
        auto name = declarator_name(content, inner, depth + 1);
        if (!name.empty()) {
            return name;
        }
    }
    inner = ts_node_child_by_field_name(node, "name", 4);
    if (!ts_node_is_null(inner)) {
        auto name = declarator_name(content, inner, depth + 1);
        if (!name.empty()) {
            return name;
        }
    }
    const uint32_t count = ts_node_named_child_count(node);
    for (uint32_t i = 0; i < count; ++i) {
        TSNode child = ts_node_named_child(node, i);
        const std::string_view child_type(ts_node_type(child));
        if (child_type == "identifier" || child_type == "field_identifier" ||
            child_type == "qualified_identifier" ||
            child_type == "destructor_name" ||
            child_type == "operator_name") {
            return collapse_ws(node_text(content, child));
        }
    }
    // Last resort: keep a short signature so the chunk still has a label.
    std::string text = collapse_ws(node_text(content, node));
    if (text.size() > 120) {
        text = truncate_utf8(text, 120);
    }
    return text;
}

std::string symbol_for(std::string_view content, TSNode node) {
    const std::string_view type(ts_node_type(node));
    if (type == "defProc") {
        TSNode header = ts_node_child_by_field_name(node, "header", 6);
        if (!ts_node_is_null(header)) {
            return symbol_for(content, header);
        }
    }
    if (type == "declUses") {
        return "uses";
    }
    if (type == "declClass" || type == "declIntf") {
        return {};
    }
    if (type == "decorated_definition" || type == "template_declaration") {
        const uint32_t count = ts_node_named_child_count(node);
        for (uint32_t i = 0; i < count; ++i) {
            TSNode child = ts_node_named_child(node, i);
            const std::string_view child_type(ts_node_type(child));
            if (child_type == "function_definition" ||
                child_type == "class_definition" ||
                child_type == "class_specifier" ||
                child_type == "struct_specifier") {
                return symbol_for(content, child);
            }
        }
    }
    TSNode name = ts_node_child_by_field_name(node, "name", 4);
    if (!ts_node_is_null(name)) {
        auto text = collapse_ws(node_text(content, name));
        if (!text.empty()) {
            return text;
        }
    }
    TSNode declarator = ts_node_child_by_field_name(node, "declarator", 10);
    if (!ts_node_is_null(declarator)) {
        auto text = declarator_name(content, declarator);
        if (!text.empty()) {
            return text;
        }
    }
    return {};
}

std::string kind_for(std::string_view content, TSNode node) {
    const std::string_view type(ts_node_type(node));
    if (type == "defProc") {
        TSNode header = ts_node_child_by_field_name(node, "header", 6);
        const std::string text =
            lower_ascii(node_text(content,
                                  ts_node_is_null(header) ? node : header));
        if (text.find("constructor") != std::string::npos) return "constructor";
        if (text.find("destructor") != std::string::npos) return "destructor";
        if (text.find("function") != std::string::npos) return "function";
        return "procedure";
    }
    if (type == "declProc") return "declaration";
    if (type == "declType") return "type";
    if (type == "declClass") return "class";
    if (type == "declIntf") return "interface";
    if (type == "declConst") return "const";
    if (type == "declUses") return "uses";
    if (type == "function_definition") return "function";
    if (type == "class_specifier" || type == "class_definition") return "class";
    if (type == "struct_specifier") return "struct";
    if (type == "union_specifier") return "union";
    if (type == "enum_specifier") return "enum";
    if (type == "type_definition") return "typedef";
    if (type == "template_declaration") return "template";
    if (type == "decorated_definition") return "definition";
    return "declaration";
}

std::string normalize_pascal_symbol(const std::string& symbol) {
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

std::vector<std::string> split_lines(std::string_view text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start <= text.size()) {
        size_t end = text.find('\n', start);
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

std::vector<RawChunk> window_chunks(std::string_view text, int first_line,
                                    const ChunkOptions& options,
                                    const std::string& symbol,
                                    const std::string& kind) {
    std::vector<RawChunk> chunks;
    auto lines = split_lines(text);
    if (lines.empty()) {
        return chunks;
    }
    const int window = std::max(4, options.window_lines);
    const int overlap = std::max(0, std::min(options.overlap_lines, window - 1));
    const int step = std::max(1, window - overlap);

    for (int start = 0; start < static_cast<int>(lines.size()); start += step) {
        int end = std::min(static_cast<int>(lines.size()), start + window);
        std::string body = trim(join_lines(lines, start, end));
        if (body.size() < 8) {
            if (end == static_cast<int>(lines.size())) {
                break;
            }
            continue;
        }
        RawChunk chunk;
        chunk.symbol = symbol;
        chunk.kind = kind;
        chunk.text = std::move(body);
        chunk.start_line = first_line + start;
        chunk.end_line = first_line + end - 1;
        chunks.push_back(std::move(chunk));
        if (end == static_cast<int>(lines.size())) {
            break;
        }
    }
    return chunks;
}

class AstCollector {
public:
    AstCollector(std::string_view content, Lang lang, ChunkOptions options)
        : content_(content), lang_(lang), options_(options) {}

    std::vector<RawChunk> collect() {
        // Handled by caller with a parsed tree.
        return {};
    }

    void prepare(TSNode root) {
        if (lang_ == Lang::Pascal) {
            collect_pascal_definitions(root);
        }
    }

    void walk(TSNode node, const std::string& parent_symbol) {
        if (is_unit_type(lang_, ts_node_type(node))) {
            const uint32_t bytes = ts_node_end_byte(node) - ts_node_start_byte(node);
            const size_t max_chars =
                static_cast<size_t>(std::max(64, options_.max_tokens * 4));
            if (bytes >= 8 && bytes <= max_chars) {
                emit(node, parent_symbol);
                return;
            }
            const std::string symbol = qualified(parent_symbol, node);
            bool descended = false;
            const uint32_t count = ts_node_named_child_count(node);
            for (uint32_t i = 0; i < count; ++i) {
                TSNode child = ts_node_named_child(node, i);
                if (is_unit_type(lang_, ts_node_type(child))) {
                    walk(child, symbol);
                    descended = true;
                }
            }
            if (!descended) {
                const uint32_t start = ts_node_start_byte(node);
                const uint32_t end = ts_node_end_byte(node);
                if (start < content_.size() && end <= content_.size()) {
                    auto windows = window_chunks(
                        content_.substr(start, end - start),
                        static_cast<int>(ts_node_start_point(node).row) + 1,
                        options_, parent_symbol, "window");
                    for (auto& chunk : windows) {
                        chunks_.push_back(std::move(chunk));
                    }
                }
            }
            return;
        }

        const uint32_t count = ts_node_named_child_count(node);
        for (uint32_t i = 0; i < count; ++i) {
            walk(ts_node_named_child(node, i), parent_symbol);
        }
    }

    std::vector<RawChunk> take() { return std::move(chunks_); }

private:
    void collect_pascal_definitions(TSNode node) {
        if (std::strcmp(ts_node_type(node), "defProc") == 0) {
            const std::string symbol = symbol_for(content_, node);
            if (!symbol.empty()) {
                defined_pascal_symbols_.insert(
                    normalize_pascal_symbol(symbol));
            }
        }
        const uint32_t count = ts_node_named_child_count(node);
        for (uint32_t i = 0; i < count; ++i) {
            collect_pascal_definitions(ts_node_named_child(node, i));
        }
    }
    std::string qualified(const std::string& parent, TSNode node) const {
        std::string name = symbol_for(content_, node);
        if (name.empty()) {
            return parent;
        }
        if (!parent.empty() && name.rfind(parent + "::", 0) != 0) {
            return parent + "::" + name;
        }
        return name;
    }

    void emit(TSNode node, const std::string& parent_symbol) {
        uint32_t start = ts_node_start_byte(node);
        const uint32_t end = ts_node_end_byte(node);
        const int start_row = static_cast<int>(ts_node_start_point(node).row);

        // Attach an immediately preceding comment block.
        TSNode prev = ts_node_prev_named_sibling(node);
        int comment_start_row = start_row;
        while (!ts_node_is_null(prev) && is_comment_type(ts_node_type(prev))) {
            const int prev_end_row =
                static_cast<int>(ts_node_end_point(prev).row);
            if (comment_start_row - prev_end_row > 2) {
                break;
            }
            start = ts_node_start_byte(prev);
            comment_start_row = static_cast<int>(ts_node_start_point(prev).row);
            prev = ts_node_prev_named_sibling(prev);
        }

        if (start >= content_.size() || end > content_.size() || end <= start) {
            return;
        }
        std::string body = trim(content_.substr(start, end - start));
        if (body.size() < 8) {
            return;
        }

        RawChunk chunk;
        chunk.symbol = qualified(parent_symbol, node);
        chunk.kind = kind_for(content_, node);
        if (lang_ == Lang::Pascal && chunk.kind == "declaration" &&
            defined_pascal_symbols_.count(
                normalize_pascal_symbol(chunk.symbol)) > 0) {
            return;
        }
        chunk.text = std::move(body);
        chunk.start_line = comment_start_row + 1;
        chunk.end_line = static_cast<int>(ts_node_end_point(node).row) + 1;
        chunks_.push_back(std::move(chunk));
    }

    std::string_view content_;
    Lang lang_;
    ChunkOptions options_;
    std::vector<RawChunk> chunks_;
    std::unordered_set<std::string> defined_pascal_symbols_;
};

}  // namespace

struct TreeSitterChunker::Impl {
    explicit Impl(ChunkOptions options) : options(options) {}

    ~Impl() {
        for (auto& [lang, parser] : parsers) {
            (void)lang;
            ts_parser_delete(parser);
        }
    }

    TSParser* parser_for(Lang lang) {
        auto it = parsers.find(lang);
        if (it != parsers.end()) {
            return it->second;
        }
        const TSLanguage* language = language_for(lang);
        if (!language) {
            return nullptr;
        }
        TSParser* parser = ts_parser_new();
        if (!parser) {
            return nullptr;
        }
        if (!ts_parser_set_language(parser, language)) {
            ts_parser_delete(parser);
            return nullptr;
        }
        parsers.emplace(lang, parser);
        return parser;
    }

    ChunkOptions options;
    std::unordered_map<Lang, TSParser*> parsers;
};

TreeSitterChunker::TreeSitterChunker(ChunkOptions options)
    : impl_(std::make_unique<Impl>(options)) {}

TreeSitterChunker::~TreeSitterChunker() = default;

std::vector<RawChunk> TreeSitterChunker::chunk(const SourceFile& file,
                                               std::string_view content) {
    if (file.lang == Lang::Text || file.lang == Lang::Unknown) {
        return chunk_lines(content, impl_->options.window_lines,
                           impl_->options.overlap_lines, file.rel_path);
    }

    TSParser* parser = impl_->parser_for(file.lang);
    if (!parser) {
        return chunk_lines(content, impl_->options.window_lines,
                           impl_->options.overlap_lines, file.rel_path);
    }
    TSTree* tree =
        ts_parser_parse_string(parser, nullptr, content.data(), content.size());
    if (!tree) {
        return chunk_lines(content, impl_->options.window_lines,
                           impl_->options.overlap_lines, file.rel_path);
    }

    TSNode root = ts_tree_root_node(tree);
    AstCollector collector(content, file.lang, impl_->options);
    collector.prepare(root);
    collector.walk(root, "");
    std::vector<RawChunk> chunks = collector.take();
    ts_tree_delete(tree);

    if (chunks.empty()) {
        return chunk_lines(content, impl_->options.window_lines,
                           impl_->options.overlap_lines, file.rel_path);
    }

    std::sort(chunks.begin(), chunks.end(),
              [](const RawChunk& a, const RawChunk& b) {
                  if (a.start_line != b.start_line) {
                      return a.start_line < b.start_line;
                  }
                  return a.end_line < b.end_line;
              });

    if (impl_->options.include_header && chunks.front().start_line > 1) {
        auto lines = split_lines(content);
        const int header_end =
            std::min<int>(static_cast<int>(lines.size()),
                          chunks.front().start_line - 1);
        if (header_end >= 1) {
            std::string header = trim(join_lines(lines, 0, header_end));
            if (header.size() >= 8) {
                auto header_chunks = window_chunks(
                    header, 1, impl_->options, file.rel_path, "header");
                chunks.insert(chunks.begin(),
                              std::make_move_iterator(header_chunks.begin()),
                              std::make_move_iterator(header_chunks.end()));
            }
        }
    }

    // Drop exact duplicates produced by overlapping windows.
    std::vector<RawChunk> deduped;
    for (auto& chunk : chunks) {
        bool duplicate = false;
        for (const auto& kept : deduped) {
            if (kept.start_line == chunk.start_line &&
                kept.end_line == chunk.end_line &&
                kept.text == chunk.text) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            deduped.push_back(std::move(chunk));
        }
    }
    return deduped;
}

std::vector<RawChunk> chunk_lines(std::string_view content, int window_lines,
                                  int overlap_lines,
                                  const std::string& symbol) {
    ChunkOptions options;
    options.window_lines = window_lines;
    options.overlap_lines = overlap_lines;
    return window_chunks(content, 1, options, symbol, "window");
}

}  // namespace rag
