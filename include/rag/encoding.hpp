#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace rag {

// Returns true when `text` is valid UTF-8 (no BOM handling).
bool is_valid_utf8(std::string_view text);

// Normalizes common aliases to names understood by iconv.
std::string normalize_encoding(std::string_view encoding);

// Converts bytes from `encoding` to UTF-8. Throws on unknown encodings or
// invalid input. "auto"/empty is rejected here; callers should detect first.
std::string convert_to_utf8(std::string_view bytes,
                            std::string_view encoding);

struct EncodingRule {
    std::string match;     // glob against relative path or file name
    std::string encoding;  // e.g. cp1251, windows-1252, utf-16le
};

// Selects a fallback encoding by path. UTF-8 is always tried first; the rules
// are only used when the bytes are not valid UTF-8 (or a BOM forces UTF-16).
class EncodingRules {
public:
    EncodingRules() = default;

    void add(std::string match, std::string encoding);
    void set_fallback(std::string encoding);

    // Built-in fallbacks for Delphi/Pascal source files (cp1251). User rules
    // are evaluated first.
    void add_defaults();

    std::string resolve(const std::string& rel_path) const;

private:
    std::vector<EncodingRule> rules_;
    std::string fallback_ = "auto";
};

// Decodes a source file to UTF-8:
//   - strips a UTF-8 BOM;
//   - converts UTF-16 LE/BE when a BOM is present;
//   - returns valid UTF-8 unchanged;
//   - otherwise applies the matching rule / fallback encoding.
// Throws with the path and a configuration hint when decoding is impossible.
std::string decode_source(std::string bytes, const std::string& rel_path,
                          const EncodingRules& rules);

}  // namespace rag
