#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "coverage/coverage.hpp"
#include "rag/chunker.hpp"
#include "rag/encoding.hpp"
#include "rag/eval.hpp"
#include "rag/search.hpp"
#include "rag/source.hpp"
#include "rag/tokenizer.hpp"
#include "rag/types.hpp"
#include "rag/util.hpp"

#include <string_view>
#include <unistd.h>

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

void test_pascal_chunker() {
    const std::string content = R"(unit StationID;

interface

uses SysUtils, IniFiles;

type
  TStation = class
    procedure FillStationsList;
    function GetName: string;
  end;

const
  StationCount = 10;

procedure FillStationsList;

implementation

// comment before the definition
procedure TStation.FillStationsList;
begin
end;

function TStation.GetName: string;
begin
  Result := 'Station';
end;

procedure FillStationsList;
begin
end;

end.
)";
    rag::ChunkOptions options;
    rag::TreeSitterChunker chunker(options);
    rag::SourceFile file;
    file.rel_path = "StationID.pas";
    file.lang = rag::Lang::Pascal;
    auto chunks = chunker.chunk(file, content);
    expect(!chunks.empty(), "Pascal chunker returns chunks");

    bool uses = false;
    bool type = false;
    bool constant = false;
    bool method = false;
    bool function = false;
    int top_level_definitions = 0;
    int declarations = 0;
    for (const auto& chunk : chunks) {
        if (chunk.kind == "uses" && chunk.symbol == "uses") uses = true;
        if (chunk.kind == "type" && chunk.symbol == "TStation") type = true;
        if (chunk.kind == "const" && chunk.symbol == "StationCount") {
            constant = true;
        }
        if (chunk.symbol == "TStation.FillStationsList") method = true;
        if (chunk.symbol == "TStation.GetName" &&
            chunk.kind == "function") {
            function = true;
        }
        if (chunk.symbol == "FillStationsList") ++top_level_definitions;
        if (chunk.kind == "declaration") ++declarations;
    }
    expect(uses, "Pascal uses chunk");
    expect(type, "Pascal type chunk");
    expect(constant, "Pascal const chunk");
    expect(method, "Pascal method definition chunk");
    expect(function, "Pascal function definition chunk");
    expect(top_level_definitions == 1,
           "interface declaration is deduplicated against implementation");
    expect(declarations == 0 || top_level_definitions == 1,
           "no duplicate Pascal declarations");
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

void test_treesitter_language_override() {
    rag::ChunkerProfile profile;
    profile.match = "*.lua";
    profile.type = "treesitter";
    profile.language = "luau";
    rag::ChunkOptions options;
    auto registry = rag::make_chunker_registry(options, {profile});

    rag::SourceFile file;
    file.rel_path = "main.lua";
    file.lang = rag::Lang::Text;
    const std::string content = "type Station = { id: number }\n"
                                "function fillStations() end\n";
    auto chunks = registry->chunk(file, content);
    expect(!chunks.empty(), "language override parses .lua as Luau");
    bool saw_type = false;
    bool saw_function = false;
    for (const auto& chunk : chunks) {
        if (chunk.symbol == "Station" && chunk.lang == rag::Lang::Luau) {
            saw_type = true;
        }
        if (chunk.symbol == "fillStations" &&
            chunk.lang == rag::Lang::Luau) {
            saw_function = true;
        }
    }
    expect(saw_type, "language override tags Luau type chunks");
    expect(saw_function, "language override tags Luau function chunks");

    rag::ChunkerProfile bad;
    bad.match = "*.lua";
    bad.type = "treesitter";
    bad.language = "not-a-language";
    bool threw = false;
    try {
        rag::make_chunker_registry(options, {bad});
    } catch (const std::exception&) {
        threw = true;
    }
    expect(threw, "unknown tree-sitter language is rejected");

    rag::ChunkerProfile auto_language;
    auto_language.match = "*.lua";
    auto_language.type = "treesitter";
    auto_language.language = "auto";
    threw = false;
    try {
        rag::make_chunker_registry(options, {auto_language});
    } catch (const std::exception&) {
        threw = true;
    }
    expect(threw, "'auto' is not a reserved language value");
}

void test_luau_chunker() {
    const std::string content = R"(--!strict
local M = {}

export type Station = {
    id: number,
    name: string,
}

function M.fillStations()
    return true
end

local function helper()
    return 1
end

local handler = function(x: number)
    return x + 1
end

return M
)";
    rag::ChunkOptions options;
    rag::TreeSitterChunker chunker(options);
    rag::SourceFile file;
    file.rel_path = "main.luau";
    file.lang = rag::Lang::Luau;
    auto chunks = chunker.chunk(file, content);
    expect(!chunks.empty(), "Luau chunker returns chunks");

    bool type = false;
    bool method = false;
    bool local_function = false;
    bool assigned_function = false;
    for (const auto& chunk : chunks) {
        if (chunk.kind == "type" && chunk.symbol == "Station") type = true;
        if (chunk.symbol == "M.fillStations" && chunk.kind == "function") {
            method = true;
        }
        if (chunk.symbol == "helper" && chunk.kind == "function") {
            local_function = true;
        }
        if (chunk.symbol == "handler" && chunk.kind == "function") {
            assigned_function = true;
        }
    }
    expect(type, "Luau type definition chunk");
    expect(method, "Luau table method chunk");
    expect(local_function, "Luau local function chunk");
    expect(assigned_function, "Luau assigned anonymous function chunk");
    expect(rag::lang_from_path("main.luau") == rag::Lang::Luau,
           "Luau extension maps to Lang::Luau");
    expect(rag::treesitter_language_available(rag::Lang::Luau),
           "Luau grammar is enabled in the default build");
    expect(rag::treesitter_language_available(rag::Lang::Rust),
           "Rust grammar is enabled in the default build");
}

void test_rust_chunker() {
    const std::string content = R"(use std::fmt;

/// A station identifier.
pub struct Station {
    pub id: u32,
}

pub enum Kind { A, B }

pub fn top_level() -> bool { true }

impl Station {
    pub fn fill_stations(&self) -> bool { true }
}

pub trait Printable {
    fn print(&self);
}

impl Printable for Station {
    fn print(&self) {}
}

pub const LIMIT: usize = 10;

pub mod helpers {
    pub fn help() {}
}
)";
    rag::ChunkOptions options;
    rag::TreeSitterChunker chunker(options);
    rag::SourceFile file;
    file.rel_path = "lib.rs";
    file.lang = rag::Lang::Rust;
    auto chunks = chunker.chunk(file, content);
    expect(!chunks.empty(), "Rust chunker returns chunks");
    bool structure = false;
    bool function = false;
    bool implementation = false;
    bool trait = false;
    bool module = false;
    for (const auto& chunk : chunks) {
        if (chunk.symbol == "Station" && chunk.kind == "struct") structure = true;
        if (chunk.symbol == "top_level" && chunk.kind == "function") {
            function = true;
        }
        if (chunk.symbol == "Station" && chunk.kind == "impl") implementation = true;
        if (chunk.symbol == "Printable" && chunk.kind == "trait") trait = true;
        if (chunk.symbol == "helpers" && chunk.kind == "module") module = true;
    }
    expect(structure, "Rust struct chunk");
    expect(function, "Rust function chunk");
    expect(implementation, "Rust impl chunk");
    expect(trait, "Rust trait chunk");
    expect(module, "Rust module chunk");
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

void test_encoding() {
    expect(rag::is_valid_utf8("hello"), "ASCII is valid UTF-8");
    expect(rag::is_valid_utf8("\xC3\xA9"), "accented UTF-8 is valid");
    expect(!rag::is_valid_utf8(std::string("\xFF", 1)),
           "lone 0xFF is invalid UTF-8");

    rag::EncodingRules rules;
    rules.add("*.pas", "cp1251");
    rules.set_fallback("auto");

    const std::string cp1251("\xCF\xF0\xE8\xE2\xE5\xF2", 6);
    const std::string expected(
        "\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82", 12);
    expect(rag::decode_source(cp1251, "unit.pas", rules) == expected,
           "cp1251 is decoded to UTF-8");
    expect(rag::decode_source("\xEF\xBB\xBFhi", "x.txt", rules) == "hi",
           "UTF-8 BOM is stripped");
    expect(rag::decode_source(std::string("\xFF\xFEh\x00i\x00", 6), "x.txt", rules) == "hi",
           "UTF-16LE BOM is converted");
    expect(rag::decode_source("plain", "x.cpp", rules) == "plain",
           "valid UTF-8 passes through");
    expect(!rag::FilesystemSource::looks_binary(
               std::string("\xFF\xFEh\x00", 4)),
           "UTF-16LE is not treated as binary");
    expect(rag::FilesystemSource::looks_binary(std::string("a\0b", 3)),
           "NUL bytes still mark binary data");

    bool threw = false;
    try {
        rag::decode_source(std::string("\xFF", 1), "x.cpp", rules);
    } catch (const std::exception&) {
        threw = true;
    }
    expect(threw, "undecodable file without a rule is rejected");

    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("rag_encoding_" + std::to_string(::getpid()));
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    {
        std::ofstream out(directory / "unit.pas", std::ios::binary);
        out.write(cp1251.data(), static_cast<std::streamsize>(cp1251.size()));
    }
    rag::FilesystemSource source(directory, {}, true, 5 * 1024 * 1024, rules);
    auto files = source.scan();
    bool saw_pascal = false;
    for (const auto& file : files) {
        if (file.rel_path == "unit.pas") {
            saw_pascal = true;
            expect(file.lang == rag::Lang::Pascal,
                   "Pascal extension maps to Lang::Pascal");
            expect(source.read(file) == expected,
                   "FilesystemSource returns decoded UTF-8");
        }
    }
    expect(saw_pascal, "cp1251 Pascal file is scanned");
    std::filesystem::remove_all(directory);
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
    test_pascal_chunker();
    test_treesitter_language_override();
    test_luau_chunker();
    test_rust_chunker();
    test_rrf();
    test_reference_matching();
    test_coverage();
    test_encoding();
    test_iso_time();
    test_wildcard();
    if (failures) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "all tests passed\n";
    return 0;
}
