#include "rag/config.hpp"

#include <cstdlib>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "rag/util.hpp"

#ifndef RAG_DEFAULT_MODEL_DIR
#define RAG_DEFAULT_MODEL_DIR ""
#endif

namespace rag {

namespace {

std::filesystem::path default_model_dir() {
#ifdef RAG_DEFAULT_MODEL_DIR
    if (std::string(RAG_DEFAULT_MODEL_DIR).size() > 0) {
        return std::filesystem::path(RAG_DEFAULT_MODEL_DIR);
    }
#endif
    const char* cache = std::getenv("XDG_CACHE_HOME");
    std::filesystem::path base =
        cache ? std::filesystem::path(cache)
              : std::filesystem::path(std::getenv("HOME") ? std::getenv("HOME")
                                                          : ".")
                    .append(".cache");
    return base / "rag-lab" / "models" / "bge-small-en-v1.5";
}

}  // namespace

Config load_config(const std::filesystem::path& root) {
    Config config;
    config.root = std::filesystem::absolute(root);
    config.db_path = config.root / ".rag" / "index.db";
    config.model_dir = resolve_model_dir(config.root);

    const auto path = config.root / ".rag" / "config.json";
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return config;
    }

    nlohmann::json json = nlohmann::json::parse(read_file(path));
    if (json.contains("db_path")) {
        config.db_path = config.root / json["db_path"].get<std::string>();
    }
    if (json.contains("model_dir")) {
        config.model_dir = config.root / json["model_dir"].get<std::string>();
    }
    if (json.contains("mode")) {
        config.mode = json["mode"].get<std::string>();
    }
    if (json.contains("top_k")) {
        config.top_k = json["top_k"].get<int>();
    }
    if (json.contains("candidate_k")) {
        config.candidate_k = json["candidate_k"].get<int>();
    }
    if (json.contains("batch_size")) {
        config.batch_size = json["batch_size"].get<int>();
    }
    if (json.contains("threads")) {
        config.threads = json["threads"].get<int>();
    }
    if (json.contains("include_docs")) {
        config.include_docs = json["include_docs"].get<bool>();
    }
    if (json.contains("max_file_bytes")) {
        config.max_file_bytes = json["max_file_bytes"].get<int>();
    }
    if (json.contains("exclude") && json["exclude"].is_array()) {
        for (const auto& item : json["exclude"]) {
            config.exclude_globs.push_back(item.get<std::string>());
        }
    }
    if (json.contains("encoding")) {
        config.default_encoding = json["encoding"].get<std::string>();
    }
    if (json.contains("encodings") && json["encodings"].is_array()) {
        for (const auto& item : json["encodings"]) {
            EncodingRule rule;
            rule.match = item.at("match").get<std::string>();
            rule.encoding = item.at("encoding").get<std::string>();
            if (rule.match.empty()) {
                throw std::runtime_error("encoding rule has empty match");
            }
            config.encoding_rules.push_back(std::move(rule));
        }
    }
    if (json.contains("chunkers") && json["chunkers"].is_array()) {
        for (const auto& item : json["chunkers"]) {
            ChunkerProfile profile;
            profile.match = item.at("match").get<std::string>();
            profile.type = item.value("type", "treesitter");
            profile.max_tokens = item.value("max_tokens", 0);
            profile.window_lines = item.value("window_lines", 0);
            profile.overlap_lines = item.value("overlap_lines", -1);
            if (profile.match.empty()) {
                throw std::runtime_error("chunker profile has empty match");
            }
            config.chunker_profiles.push_back(std::move(profile));
        }
    }
    if (json.contains("chunk") && json["chunk"].is_object()) {
        const auto& chunk = json["chunk"];
        if (chunk.contains("max_tokens")) {
            config.chunk.max_tokens = chunk["max_tokens"].get<int>();
        }
        if (chunk.contains("window_lines")) {
            config.chunk.window_lines = chunk["window_lines"].get<int>();
        }
        if (chunk.contains("overlap_lines")) {
            config.chunk.overlap_lines = chunk["overlap_lines"].get<int>();
        }
        if (chunk.contains("include_header")) {
            config.chunk.include_header = chunk["include_header"].get<bool>();
        }
    }
    return config;
}

std::filesystem::path resolve_model_dir(const std::filesystem::path& root,
                                        const std::string& explicit_dir) {
    std::error_code ec;
    if (!explicit_dir.empty()) {
        return std::filesystem::absolute(explicit_dir);
    }
    if (const char* env = std::getenv("RAG_MODEL_DIR");
        env && *env) {
        return std::filesystem::path(env);
    }
    for (const auto& candidate :
         {root / "models" / "bge-small-en-v1.5",
          default_model_dir()}) {
        if (std::filesystem::exists(candidate / "model.json", ec) ||
            std::filesystem::exists(candidate / "vocab.txt", ec)) {
            return candidate;
        }
    }
    return default_model_dir();
}

}  // namespace rag
