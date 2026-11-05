#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "rag/encoding.hpp"
#include "rag/types.hpp"

namespace rag {

struct ChunkOptions {
    int max_tokens = 512;
    int window_lines = 60;
    int overlap_lines = 12;
    bool include_header = true;
};

struct ChunkerProfile {
    // Glob matched against the relative path or the file name.
    std::string match;
    // treesitter, markdown or window.
    std::string type = "treesitter";
    // Zero/negative values inherit the global chunk options.
    int max_tokens = 0;
    int window_lines = 0;
    int overlap_lines = -1;
};

struct Config {
    std::filesystem::path root = std::filesystem::current_path();
    std::filesystem::path db_path;
    std::filesystem::path model_dir;
    std::string mode = "hybrid";
    int top_k = 10;
    int candidate_k = 50;
    int batch_size = 8;
    int threads = 0;  // 0 = ONNX Runtime default
    ChunkOptions chunk;
    std::vector<ChunkerProfile> chunker_profiles;
    std::vector<EncodingRule> encoding_rules;
    std::string default_encoding = "auto";
    std::vector<std::string> exclude_globs;
    bool include_docs = true;
    int max_file_bytes = 5 * 1024 * 1024;
};

// Loads .rag/config.json from the root when present. Unknown keys are ignored.
Config load_config(const std::filesystem::path& root);

// Resolves the model directory in this order:
//   1. explicit override
//   2. $RAG_MODEL_DIR
//   3. <root>/models/<name>
//   4. compiled-in models directory
std::filesystem::path resolve_model_dir(
    const std::filesystem::path& root,
    const std::string& explicit_dir = {});

}  // namespace rag
