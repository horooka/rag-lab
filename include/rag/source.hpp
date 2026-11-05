#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "rag/encoding.hpp"
#include "rag/types.hpp"

namespace rag {

class Source {
public:
    virtual ~Source() = default;
    virtual std::vector<SourceFile> scan() = 0;
    virtual std::string read(const SourceFile& file) = 0;
};

class FilesystemSource : public Source {
public:
    FilesystemSource(std::filesystem::path root,
                     std::vector<std::string> exclude_globs = {},
                     bool include_docs = true,
                     int max_file_bytes = 5 * 1024 * 1024,
                     EncodingRules encodings = {});

    std::vector<SourceFile> scan() override;
    std::string read(const SourceFile& file) override;

    // Exposed for tests.
    static bool looks_binary(const std::string& content);

private:
    bool excluded(const std::string& rel_path, bool is_dir) const;
    std::string repo_for(const std::filesystem::path& dir) const;

    std::filesystem::path root_;
    std::vector<std::string> exclude_globs_;
    bool include_docs_;
    int max_file_bytes_;
    EncodingRules encodings_;
    mutable std::vector<std::pair<std::string, std::string>> repo_cache_;
};

}  // namespace rag
