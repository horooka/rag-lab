#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rag {

// BERT WordPiece tokenizer (the tokenizer used by bge-* models).
// Not thread-safe to construct, but const encode() calls are safe.
class BertTokenizer {
public:
    explicit BertTokenizer(const std::filesystem::path& vocab_path);

    std::vector<int32_t> encode(std::string_view text,
                                size_t max_length = 512) const;

    size_t vocab_size() const { return vocab_.size(); }
    int32_t id(const std::string& token) const;

private:
    std::vector<std::string> basic_tokenize(std::string_view text) const;
    std::vector<std::string> wordpiece(const std::string& word) const;

    std::unordered_map<std::string, int32_t> vocab_;
    int32_t cls_id_ = 101;
    int32_t sep_id_ = 102;
    int32_t unk_id_ = 100;
    int32_t pad_id_ = 0;
};

}  // namespace rag
