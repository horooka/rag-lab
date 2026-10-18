#pragma once

#include <memory>
#include <string>
#include <utility>
#include <string_view>
#include <vector>

#include "rag/config.hpp"
#include "rag/types.hpp"

namespace rag {

class Chunker {
public:
    virtual ~Chunker() = default;
    virtual std::vector<RawChunk> chunk(const SourceFile& file,
                                        std::string_view content) = 0;
};

// AST-aware chunker for C, C++ and Python. Unknown languages (and parse
// failures) fall back to overlapping line windows.
class TreeSitterChunker : public Chunker {
public:
    explicit TreeSitterChunker(ChunkOptions options);
    ~TreeSitterChunker() override;

    std::vector<RawChunk> chunk(const SourceFile& file,
                                std::string_view content) override;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// Plain overlapping line windows; useful for docs, scripts and tests.
std::vector<RawChunk> chunk_lines(std::string_view content,
                                  int window_lines,
                                  int overlap_lines,
                                  const std::string& symbol = {});

// Heading-aware chunker for Markdown. Sections are labelled with their
// heading path, e.g. "Intro > Setup". Oversized sections fall back to line
// windows.
class MarkdownChunker : public Chunker {
public:
    explicit MarkdownChunker(ChunkOptions options);
    std::vector<RawChunk> chunk(const SourceFile& file,
                                std::string_view content) override;

private:
    ChunkOptions options_;
};

// Simple line-window chunker for scripts, build files and unknown formats.
class WindowChunker : public Chunker {
public:
    explicit WindowChunker(ChunkOptions options);
    std::vector<RawChunk> chunk(const SourceFile& file,
                                std::string_view content) override;

private:
    ChunkOptions options_;
};

// Dispatches each file to the first matching rule. A fallback window chunker
// is used when no rule matches.
class ChunkerRegistry : public Chunker {
public:
    explicit ChunkerRegistry(ChunkOptions fallback_options);
    ~ChunkerRegistry() override;

    void add(std::string pattern, std::shared_ptr<Chunker> chunker);
    void set_fallback(std::shared_ptr<Chunker> chunker);

    std::vector<RawChunk> chunk(const SourceFile& file,
                                std::string_view content) override;

private:
    struct Rule {
        std::string pattern;
        std::shared_ptr<Chunker> chunker;
    };
    std::vector<Rule> rules_;
    std::shared_ptr<Chunker> fallback_;
};

// Builds the built-in profile set when `profiles` is empty, otherwise uses the
// configured rules. Rules are evaluated in order; include a catch-all "*"
// rule if you want full control.
std::unique_ptr<Chunker> make_chunker_registry(
    const ChunkOptions& options,
    const std::vector<ChunkerProfile>& profiles);

}  // namespace rag
