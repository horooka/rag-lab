#include "rag/chunker.hpp"

#include <filesystem>
#include <stdexcept>

#include "rag/util.hpp"

namespace rag {

WindowChunker::WindowChunker(ChunkOptions options)
    : options_(std::move(options)) {}

std::vector<RawChunk> WindowChunker::chunk(const SourceFile& file,
                                           std::string_view content) {
    return chunk_lines(content, options_.window_lines,
                       options_.overlap_lines, file.rel_path);
}

ChunkerRegistry::ChunkerRegistry(ChunkOptions fallback_options)
    : fallback_(std::make_shared<WindowChunker>(std::move(fallback_options))) {}

ChunkerRegistry::~ChunkerRegistry() = default;

void ChunkerRegistry::add(std::string pattern,
                          std::shared_ptr<Chunker> chunker) {
    if (pattern.empty() || !chunker) {
        throw std::invalid_argument("invalid chunker rule");
    }
    rules_.push_back({std::move(pattern), std::move(chunker)});
}

void ChunkerRegistry::set_fallback(std::shared_ptr<Chunker> chunker) {
    if (!chunker) {
        throw std::invalid_argument("chunker fallback must not be null");
    }
    fallback_ = std::move(chunker);
}

std::vector<RawChunk> ChunkerRegistry::chunk(const SourceFile& file,
                                             std::string_view content) {
    const std::string path = lower_ascii(file.rel_path);
    const std::string filename =
        lower_ascii(std::filesystem::path(path).filename().string());
    for (const auto& rule : rules_) {
        const std::string pattern = lower_ascii(rule.pattern);
        if (wildcard_match(pattern, path) ||
            wildcard_match(pattern, filename)) {
            return rule.chunker->chunk(file, content);
        }
    }
    return fallback_->chunk(file, content);
}

std::unique_ptr<Chunker> make_chunker_registry(
    const ChunkOptions& options, const std::vector<ChunkerProfile>& profiles) {
    auto registry = std::make_unique<ChunkerRegistry>(options);

    auto make = [&](const ChunkerProfile& profile) -> std::shared_ptr<Chunker> {
        ChunkOptions local = options;
        if (profile.max_tokens > 0) {
            local.max_tokens = profile.max_tokens;
        }
        if (profile.window_lines > 0) {
            local.window_lines = profile.window_lines;
        }
        if (profile.overlap_lines >= 0) {
            local.overlap_lines = profile.overlap_lines;
        }
        if (profile.type == "treesitter") {
            return std::make_shared<TreeSitterChunker>(local);
        }
        if (profile.type == "markdown") {
            return std::make_shared<MarkdownChunker>(local);
        }
        if (profile.type == "window") {
            return std::make_shared<WindowChunker>(local);
        }
        throw std::invalid_argument("unknown chunker type: " + profile.type);
    };

    if (profiles.empty()) {
        const std::vector<std::string> source_extensions = {
            "*.c",   "*.h",   "*.cc",  "*.cpp", "*.cxx", "*.hpp",
            "*.hh",  "*.hxx", "*.inl", "*.ipp", "*.py",  "*.pyi"};
        for (const auto& pattern : source_extensions) {
            ChunkerProfile profile;
            profile.match = pattern;
            profile.type = "treesitter";
            registry->add(profile.match, make(profile));
        }
        for (const char* pattern : {"*.md", "*.markdown", "*.rst"}) {
            ChunkerProfile profile;
            profile.match = pattern;
            profile.type = "markdown";
            registry->add(profile.match, make(profile));
        }
    } else {
        for (const auto& profile : profiles) {
            registry->add(profile.match, make(profile));
        }
    }
    return registry;
}

}  // namespace rag
