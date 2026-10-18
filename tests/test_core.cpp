#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "coverage/coverage.hpp"
#include "rag/chunker.hpp"
#include "rag/eval.hpp"
#include "rag/search.hpp"
#include "rag/tokenizer.hpp"
#include "rag/types.hpp"
#include "rag/util.hpp"

#include <string_view>

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++failures;
    }
}

void test_tokenizer() {
    const std::string model_dir =
        std::getenv("RAG_MODEL_DIR") ? std::getenv("RAG_MODEL_DIR")
                                     : "models/bge-small-en-v1.5";
    rag::BertTokenizer tokenizer(model_dir + "/vocab.txt");

    auto check = [&](const std::string& text,
                     const std::vector<int32_t>& expected) {
        auto actual = tokenizer.encode(text, 512);
        if (actual != expected) {
            std::cerr << "tokenizer mismatch for: " << text << "\n  expected:";
            for (int id : expected) std::cerr << " " << id;
            std::cerr << "\n  actual:  ";
            for (int id : actual) std::cerr << " " << id;
            std::cerr << "\n";
            ++failures;
        }
    };

    check("hello world", {101, 7592, 2088, 102});
    check("parse_config(const std::string& value)",
          {101, 11968, 3366, 1035, 9530, 8873, 2290, 1006, 9530,
           3367, 2358, 2094, 1024, 1024, 5164, 1004, 3643, 1007, 102});
    check("RTP packet archiver",
          {101, 19387, 2361, 14771, 8756, 2099, 102});
    check("Naïve café", {101, 15743, 7668, 102});
    check("kdb_scripts_embed",
          {101, 1047, 18939, 1035, 14546, 1035, 7861, 8270, 102});
}

void test_chunker() {
    rag::ChunkOptions options;
    rag::TreeSitterChunker chunker(options);
    const std::string content = R"(#include <string>

// Parse a configuration value.
int parse_config(const std::string& value) {
    return value.size();
}

class PacketArchiver {
public:
    bool archive_packet(const std::string& packet) {
        return packet.size() > 0;
    }
};

int main() { return 0; }
)";
    rag::SourceFile file;
    file.rel_path = "sample.cpp";
    file.lang = rag::Lang::Cpp;
    auto chunks = chunker.chunk(file, content);
    expect(!chunks.empty(), "chunker returns chunks");
    bool saw_function = false;
    bool saw_class = false;
    bool saw_header = false;
    for (const auto& chunk : chunks) {
        if (chunk.symbol == "parse_config" || chunk.symbol == "main") {
            saw_function = true;
        }
        if (chunk.symbol == "PacketArchiver") {
            saw_class = true;
        }
        if (chunk.kind == "header") {
            saw_header = true;
        }
        expect(chunk.start_line >= 1, "chunk start line");
        expect(chunk.end_line >= chunk.start_line, "chunk line range");
    }
    expect(saw_function, "function symbols detected");
    expect(saw_class, "class symbol detected");
    expect(saw_header, "header chunk detected");

    auto windows = rag::chunk_lines(
        "line one is long enough\nline two is long enough\n"
        "line three is long enough\nline four is long enough\n",
        3, 1);
    expect(!windows.empty(), "line windows produced");
    if (!windows.empty()) {
        expect(windows.front().start_line == 1, "first window starts at line 1");
    }
}

void test_markdown_and_registry() {
    const std::string markdown = R"(# Intro

Some intro text.

## Setup

Install it.

```bash
# not a heading
echo hi
```

# API

Call the endpoint.
)";
    rag::SourceFile markdown_file;
    markdown_file.rel_path = "guide.md";
    markdown_file.lang = rag::Lang::Text;

    rag::ChunkOptions options;
    rag::MarkdownChunker markdown_chunker(options);
    auto markdown_chunks = markdown_chunker.chunk(markdown_file, markdown);
    bool saw_setup = false;
    bool saw_api = false;
    bool saw_fenced_comment = false;
    for (const auto& chunk : markdown_chunks) {
        if (chunk.symbol == "Intro > Setup") {
            saw_setup = true;
        }
        if (chunk.symbol == "API") {
            saw_api = true;
        }
        if (chunk.symbol.find("not a heading") != std::string::npos) {
            saw_fenced_comment = true;
        }
    }
    expect(saw_setup, "markdown heading path is used as symbol");
    expect(saw_api, "markdown top-level heading is detected");
    expect(!saw_fenced_comment, "markdown headings inside code fences ignored");

    std::vector<rag::ChunkerProfile> profiles;
    rag::ChunkerProfile md_profile;
    md_profile.match = "*.md";
    md_profile.type = "markdown";
    profiles.push_back(md_profile);
    rag::ChunkerProfile cpp_profile;
    cpp_profile.match = "*.cpp";
    cpp_profile.type = "treesitter";
    profiles.push_back(cpp_profile);

    auto registry = rag::make_chunker_registry(options, profiles);
    auto registry_markdown = registry->chunk(markdown_file, markdown);
    expect(!registry_markdown.empty(), "registry dispatches markdown");
    bool registry_setup = false;
    for (const auto& chunk : registry_markdown) {
        if (chunk.symbol == "Intro > Setup") registry_setup = true;
    }
    expect(registry_setup, "registry selected markdown chunker");

    rag::SourceFile cpp_file;
    cpp_file.rel_path = "sample.cpp";
    cpp_file.lang = rag::Lang::Cpp;
    auto cpp_chunks = registry->chunk(cpp_file,
                                      "int sample_function() { return 1; }\n");
    bool cpp_function = false;
    for (const auto& chunk : cpp_chunks) {
        if (chunk.symbol == "sample_function") cpp_function = true;
    }
    expect(cpp_function, "registry selected tree-sitter chunker");

    rag::SourceFile text_file;
    text_file.rel_path = "notes.txt";
    text_file.lang = rag::Lang::Text;
    auto text_chunks = registry->chunk(
        text_file,
        "line one is long enough\nline two is long enough\n"
        "line three is long enough\n");
    expect(!text_chunks.empty(), "registry falls back to window chunker");
}

void test_rrf() {
    std::vector<rag::RankedId> dense = {{1, 0.1}, {2, 0.2}};
    std::vector<rag::RankedId> bm25 = {{2, 3.0}, {3, 2.0}};
    auto fused = rag::Searcher::rrf_fuse(dense, bm25);
    expect(fused.size() == 3, "rrf keeps union");
    expect(fused.front().first == 2, "rrf ranks agreement first");
}

void test_reference_matching() {
    rag::Chunk chunk;
    chunk.path = "RtpArchiver-gxx/src/archiver.cpp";
    chunk.repo = "RtpArchiver-gxx";
    chunk.symbol = "PacketArchiver::archive_packet";
    expect(rag::reference_matches(
               "RtpArchiver-gxx/src/archiver.cpp", chunk),
           "full path matches");
    expect(rag::reference_matches("src/archiver.cpp", chunk),
           "path suffix matches");
    expect(rag::reference_matches(
               "src/archiver.cpp::archive_packet", chunk),
           "symbol suffix matches");
    expect(!rag::reference_matches("src/other.cpp", chunk),
           "wrong path rejected");
}

class FakeEmbedder final : public rag::Embedder {
public:
    FakeEmbedder() { spec_.dim = 2; }
    int dim() const override { return 2; }
    const rag::EmbeddingModelSpec& spec() const override { return spec_; }
    std::vector<float> embed_query(std::string_view text) override {
        if (text == "a") return {1.0f, 0.0f};
        if (text == "b") return {0.0f, 1.0f};
        return {0.707f, 0.707f};
    }
    std::vector<std::vector<float>> embed_documents(
        const std::vector<std::string>& texts) override {
        std::vector<std::vector<float>> out;
        for (const auto& text : texts) {
            out.push_back(embed_query(text));
        }
        return out;
    }

private:
    rag::EmbeddingModelSpec spec_;
};

void test_coverage() {
    auto alpha = coverage::parse_group("alpha=a");
    auto beta = coverage::parse_group("beta=b");
    expect(alpha.label == "alpha" && alpha.keywords.size() == 1,
           "group parsed from label=keyword");
    auto duplicate = coverage::parse_group("ui=button, dialog, button");
    expect(duplicate.label == "ui" && duplicate.keywords.size() == 2,
           "group keywords deduplicated");
    auto implicit = coverage::parse_group("network");
    expect(implicit.label == "network" && implicit.keywords.size() == 1,
           "group without '=' becomes a single keyword");

    std::vector<rag::ChunkEmbedding> chunks(2);
    chunks[0].chunk.path = "file.cpp";
    chunks[0].chunk.start_line = 1;
    chunks[0].chunk.end_line = 3;
    chunks[0].embedding = {1.0f, 0.0f};
    chunks[1].chunk.path = "file.cpp";
    chunks[1].chunk.start_line = 3;
    chunks[1].chunk.end_line = 5;
    chunks[1].embedding = {0.0f, 1.0f};

    FakeEmbedder embedder;
    coverage::CoverageOptions options;
    options.threshold = 0.5;
    options.details = true;
    auto result = coverage::analyze(chunks, {alpha, beta}, embedder, options);
    expect(result.total_chunks == 2, "coverage counts chunks");
    expect(result.assigned_chunks == 2, "coverage assigns both chunks");
    expect(result.groups.size() == 2, "coverage reports both groups");
    expect(result.groups[0].label == "alpha", "alpha sorted first by share");
    expect(result.groups[0].chunks == 1 && result.groups[1].chunks == 1,
           "coverage chunk counts");
    expect(result.matches.size() == 2, "coverage detail matches");

    options.line_unit = true;
    auto lines = coverage::analyze(chunks, {alpha, beta}, embedder, options);
    expect(lines.total_lines == 5, "overlapping line ranges de-duplicated");
    expect(lines.assigned_lines == 5, "all covered lines assigned");
    expect(lines.groups[0].lines == 3 && lines.groups[1].lines == 2,
           "line ownership respects similarity and overlap");
}

void test_iso_time() {
    expect(rag::parse_iso8601_utc("1970-01-01") == 0,
           "Unix epoch parses to zero");
    const int64_t first = rag::parse_iso8601_utc("2026-11-04");
    const int64_t second = rag::parse_iso8601_utc("2026-11-05T12:00:00Z");
    expect(second > first, "later ISO timestamp sorts after earlier one");
    expect(second - first == 36LL * 3600 * 1000000000LL,
           "date and timestamp difference is 36 hours");
}

void test_wildcard() {
    expect(rag::wildcard_match("*.cpp", "archiver.cpp"), "* glob");
    expect(rag::wildcard_match("cmake-build-*", "cmake-build-debug"), "prefix glob");
    expect(!rag::wildcard_match("*.cpp", "archiver.py"), "glob negative");
}

}  // namespace

int main() {
    test_tokenizer();
    test_chunker();
    test_markdown_and_registry();
    test_rrf();
    test_reference_matching();
    test_coverage();
    test_iso_time();
    test_wildcard();
    if (failures) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "all tests passed\n";
    return 0;
}
