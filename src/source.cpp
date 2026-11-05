#include "rag/source.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <unordered_set>

#include "rag/encoding.hpp"
#include "rag/util.hpp"

namespace rag {

namespace {

const std::unordered_set<std::string>& ignored_dirs() {
    static const std::unordered_set<std::string> dirs = {
        ".git",       ".hg",       ".svn",      ".rag",
        ".idea",      ".vscode",   ".cache",    ".tox",
        ".venv",      "venv",      "env",       "__pycache__",
        "build",      "Build",     "_build",    "cmake-build-debug",
        "cmake-build-release",     "CMakeFiles", "node_modules",
        "dist",       "target",    "Debug",     "Release",
        "bin",        "obj",       "out",       "install",
        ".deps",      "bower_components",        "vendor_build"};
    return dirs;
}

bool is_source_ext(const std::string& ext) {
    static const std::unordered_set<std::string> exts = {
        ".c",   ".h",    ".cc",   ".cpp",  ".cxx", ".hpp", ".hh",
        ".hxx", ".inl",  ".ipp",  ".py",   ".pyi", ".ts",  ".tsx",
        ".js",  ".jsx",  ".m",    ".mm",   ".rs",  ".go",  ".java",
        ".kt",  ".cs",   ".swift", ".pas",  ".dpr", ".dpk", ".lpr",
        ".dfm", ".inc"};
    return exts.count(ext) > 0;
}

bool is_doc_ext(const std::string& ext) {
    static const std::unordered_set<std::string> exts = {
        ".md",   ".txt",  ".json", ".yaml", ".yml",  ".toml", ".ini",
        ".cfg",  ".conf", ".sh",   ".bash", ".cmake", ".sql", ".proto",
        ".xml",  ".html", ".css",  ".qml",  ".ui",   ".pro",  ".pri",
        ".mk",   ".rst",  ".adoc", ".csv"};
    return exts.count(ext) > 0;
}

bool is_known_filename(const std::string& name) {
    static const std::unordered_set<std::string> names = {
        "CMakeLists.txt", "Makefile", "makefile", "GNUmakefile",
        "Dockerfile",     ".gitignore", ".gitattributes", "meson.build"};
    return names.count(name) > 0;
}

std::string normalized_path(const std::filesystem::path& path) {
    return path.lexically_normal().generic_string();
}

int64_t file_time_to_unix_ns(std::filesystem::file_time_type value) {
    using namespace std::chrono;
    const auto system_now = system_clock::now();
    const auto file_now = std::filesystem::file_time_type::clock::now();
    const auto system_time = time_point_cast<system_clock::duration>(
        value - file_now + system_now);
    return duration_cast<nanoseconds>(system_time.time_since_epoch()).count();
}

}  // namespace

const char* lang_name(Lang lang) {
    switch (lang) {
        case Lang::C:
            return "c";
        case Lang::Cpp:
            return "cpp";
        case Lang::Python:
            return "python";
        case Lang::Pascal:
            return "pascal";
        case Lang::Text:
            return "text";
        default:
            return "unknown";
    }
}

Lang lang_from_path(const std::string& path) {
    std::string lower = lower_ascii(path);
    if (ends_with(lower, ".c")) {
        return Lang::C;
    }
    for (const char* ext : {".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx",
                            ".h", ".inl", ".ipp"}) {
        if (ends_with(lower, ext)) {
            return Lang::Cpp;
        }
    }
    if (ends_with(lower, ".py") || ends_with(lower, ".pyi")) {
        return Lang::Python;
    }
    for (const char* ext : {".pas", ".dpr", ".dpk", ".lpr", ".dfm"}) {
        if (ends_with(lower, ext)) {
            return Lang::Pascal;
        }
    }
    if (is_source_ext(std::filesystem::path(lower).extension().string()) ||
        is_doc_ext(std::filesystem::path(lower).extension().string()) ||
        is_known_filename(std::filesystem::path(path).filename().string())) {
        return Lang::Text;
    }
    return Lang::Unknown;
}

FilesystemSource::FilesystemSource(std::filesystem::path root,
                                   std::vector<std::string> exclude_globs,
                                   bool include_docs, int max_file_bytes,
                                   EncodingRules encodings)
    : root_(std::filesystem::absolute(std::move(root))),
      exclude_globs_(std::move(exclude_globs)),
      include_docs_(include_docs),
      max_file_bytes_(max_file_bytes),
      encodings_(std::move(encodings)) {}

bool FilesystemSource::excluded(const std::string& rel_path,
                                bool is_dir) const {
    const auto parts = split(rel_path, '/');
    for (const auto& part : parts) {
        if (ignored_dirs().count(part)) {
            return true;
        }
    }

    for (const auto& pattern : exclude_globs_) {
        if (pattern.empty()) {
            continue;
        }
        if (pattern.find('/') == std::string::npos) {
            for (const auto& part : parts) {
                if (wildcard_match(pattern, part)) {
                    return true;
                }
            }
        } else if (wildcard_match(pattern, rel_path)) {
            return true;
        }
        if (is_dir && wildcard_match(pattern, rel_path + "/")) {
            return true;
        }
    }
    return false;
}

std::string FilesystemSource::repo_for(const std::filesystem::path& dir) const {
    const std::string key = normalized_path(dir);
    for (const auto& [cached_dir, repo] : repo_cache_) {
        if (cached_dir == key) {
            return repo;
        }
    }

    std::filesystem::path current = dir;
    std::string repo = root_.filename().string();
    std::error_code ec;
    while (true) {
        if (std::filesystem::exists(current / ".git", ec)) {
            repo = current.filename().string();
            break;
        }
        if (current == root_ || current == current.parent_path()) {
            break;
        }
        current = current.parent_path();
    }
    repo_cache_.emplace_back(key, repo);
    return repo;
}

std::vector<SourceFile> FilesystemSource::scan() {
    std::vector<SourceFile> files;
    std::error_code ec;

    std::filesystem::recursive_directory_iterator it(
        root_, std::filesystem::directory_options::skip_permission_denied, ec);
    if (ec) {
        throw std::runtime_error("cannot scan " + root_.string() + ": " +
                                 ec.message());
    }

    std::filesystem::recursive_directory_iterator end;
    for (; it != end; it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }
        const auto& entry = *it;
        const auto rel = normalized_path(entry.path().lexically_relative(root_));
        if (rel.empty() || rel == ".") {
            continue;
        }

        if (entry.is_directory(ec)) {
            if (excluded(rel, true)) {
                it.disable_recursion_pending();
            }
            continue;
        }
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        if (excluded(rel, false)) {
            continue;
        }

        const std::string filename = entry.path().filename().string();
        if (filename.empty() || filename[0] == '.') {
            if (!is_known_filename(filename)) {
                continue;
            }
        }

        const std::string ext = lower_ascii(entry.path().extension().string());
        if (!is_source_ext(ext)) {
            if (!include_docs_ || !is_doc_ext(ext)) {
                if (!is_known_filename(filename)) {
                    continue;
                }
            }
        }

        const auto size = entry.file_size(ec);
        if (ec || size <= 0 || size > static_cast<uintmax_t>(max_file_bytes_)) {
            ec.clear();
            continue;
        }

        std::ifstream probe(entry.path(), std::ios::binary);
        if (!probe) {
            continue;
        }
        std::string head(4096, '\0');
        probe.read(head.data(), static_cast<std::streamsize>(head.size()));
        head.resize(static_cast<size_t>(probe.gcount()));
        if (looks_binary(head)) {
            continue;
        }

        SourceFile file;
        file.abs_path = entry.path().string();
        file.rel_path = rel;
        file.repo = repo_for(entry.path().parent_path());
        file.lang = lang_from_path(rel);
        if (file.lang == Lang::Unknown) {
            file.lang = Lang::Text;
        }
        file.size = static_cast<int64_t>(size);
        const auto write_time = entry.last_write_time(ec);
        if (ec) {
            ec.clear();
            file.mtime_ns = 0;
        } else {
            file.mtime_ns = file_time_to_unix_ns(write_time);
        }
        files.push_back(std::move(file));
    }

    std::sort(files.begin(), files.end(),
              [](const SourceFile& a, const SourceFile& b) {
                  return a.rel_path < b.rel_path;
              });
    return files;
}

std::string FilesystemSource::read(const SourceFile& file) {
    std::string content =
        decode_source(read_file(file.abs_path), file.rel_path, encodings_);
    // Normalize CRLF so tree-sitter byte offsets and line numbers agree.
    std::string out;
    out.reserve(content.size());
    for (size_t i = 0; i < content.size(); ++i) {
        if (content[i] == '\r' && i + 1 < content.size() &&
            content[i + 1] == '\n') {
            continue;
        }
        out.push_back(content[i]);
    }
    return out;
}

bool FilesystemSource::looks_binary(const std::string& content) {
    // UTF-16 has NUL bytes by design; let decode_source handle BOM'd files.
    if (starts_with(content, "\xFF\xFE") ||
        starts_with(content, "\xFE\xFF")) {
        return false;
    }
    return content.find('\0') != std::string::npos;
}

}  // namespace rag
