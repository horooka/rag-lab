#include "rag/encoding.hpp"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <iconv.h>

#include "rag/util.hpp"

namespace rag {

namespace {

bool starts_with_bytes(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() &&
           std::memcmp(text.data(), prefix.data(), prefix.size()) == 0;
}

std::string decode_error(std::string_view rel_path,
                         std::string_view encoding) {
    return "cannot decode " + std::string(rel_path) + " as " +
           std::string(encoding) +
           " (configure \"encodings\" in .rag/config.json or pass "
           "--encoding)";
}


bool path_matches(const std::string& pattern, const std::string& rel_path) {
    const std::string lower_pattern = lower_ascii(pattern);
    const std::string lower_path = lower_ascii(rel_path);
    if (wildcard_match(lower_pattern, lower_path)) {
        return true;
    }
    const std::string filename =
        lower_ascii(std::filesystem::path(rel_path).filename().string());
    return wildcard_match(lower_pattern, filename);
}

}  // namespace

bool is_valid_utf8(std::string_view text) {
    size_t i = 0;
    while (i < text.size()) {
        const auto byte = static_cast<unsigned char>(text[i]);
        if (byte < 0x80) {
            ++i;
            continue;
        }
        int extra = 0;
        uint32_t codepoint = 0;
        if ((byte & 0xE0) == 0xC0) {
            extra = 1;
            codepoint = byte & 0x1F;
        } else if ((byte & 0xF0) == 0xE0) {
            extra = 2;
            codepoint = byte & 0x0F;
        } else if ((byte & 0xF8) == 0xF0) {
            extra = 3;
            codepoint = byte & 0x07;
        } else {
            return false;
        }
        if (i + static_cast<size_t>(extra) >= text.size()) {
            return false;
        }
        for (int j = 1; j <= extra; ++j) {
            const auto next = static_cast<unsigned char>(text[i + j]);
            if ((next & 0xC0) != 0x80) {
                return false;
            }
            codepoint = (codepoint << 6) | (next & 0x3F);
        }
        if (extra == 1 && codepoint < 0x80) return false;
        if (extra == 2 && (codepoint < 0x800 || codepoint > 0xFFFF ||
                           (codepoint >= 0xD800 && codepoint <= 0xDFFF))) {
            return false;
        }
        if (extra == 3 && (codepoint < 0x10000 || codepoint > 0x10FFFF)) {
            return false;
        }
        i += static_cast<size_t>(extra) + 1;
    }
    return true;
}

std::string normalize_encoding(std::string_view encoding) {
    const std::string value = lower_ascii(trim(encoding));
    if (value.empty() || value == "auto") {
        return "auto";
    }
    if (value == "utf8" || value == "utf-8") {
        return "UTF-8";
    }
    if (value == "cp1251" || value == "windows-1251" || value == "win1251") {
        return "CP1251";
    }
    if (value == "cp1252" || value == "windows-1252" || value == "win1252") {
        return "CP1252";
    }
    if (value == "latin1" || value == "latin-1" || value == "iso-8859-1") {
        return "ISO-8859-1";
    }
    if (value == "utf16" || value == "utf-16") {
        return "UTF-16";
    }
    if (value == "utf16le" || value == "utf-16le") {
        return "UTF-16LE";
    }
    if (value == "utf16be" || value == "utf-16be") {
        return "UTF-16BE";
    }
    return std::string(encoding);
}

std::string convert_to_utf8(std::string_view bytes,
                            std::string_view encoding) {
    const std::string name = normalize_encoding(encoding);
    if (name == "auto" || name.empty()) {
        throw std::runtime_error("no fallback encoding configured");
    }
    if (name == "UTF-8") {
        if (!is_valid_utf8(bytes)) {
            throw std::runtime_error("input is not valid UTF-8");
        }
        return std::string(bytes);
    }

    iconv_t converter = iconv_open("UTF-8", name.c_str());
    if (converter == reinterpret_cast<iconv_t>(-1)) {
        throw std::runtime_error("unsupported encoding: " + name);
    }

    std::string output(bytes.size() * 2 + 32, '\0');
    char* input = const_cast<char*>(bytes.data());
    size_t input_left = bytes.size();
    size_t output_used = 0;

    while (input_left > 0) {
        char* output_ptr = output.data() + output_used;
        size_t output_left = output.size() - output_used;
        const size_t result =
            iconv(converter, &input, &input_left, &output_ptr, &output_left);
        output_used = output.size() - output_left;
        if (result == static_cast<size_t>(-1)) {
            if (errno == E2BIG) {
                output.resize(output.size() * 2);
                continue;
            }
            const int saved_errno = errno;
            iconv_close(converter);
            throw std::runtime_error(
                "iconv failed: " + std::string(std::strerror(saved_errno)));
        }
    }
    iconv_close(converter);
    output.resize(output_used);
    if (!is_valid_utf8(output)) {
        throw std::runtime_error("conversion produced invalid UTF-8");
    }
    return output;
}

void EncodingRules::add(std::string match, std::string encoding) {
    if (match.empty()) {
        return;
    }
    rules_.push_back({std::move(match), std::move(encoding)});
}

void EncodingRules::set_fallback(std::string encoding) {
    fallback_ = std::move(encoding);
}

void EncodingRules::add_defaults() {
    for (const char* pattern : {"*.pas", "*.dpr", "*.dpk", "*.lpr", "*.dfm",
                                "*.inc"}) {
        rules_.push_back({pattern, "cp1251"});
    }
}

std::string EncodingRules::resolve(const std::string& rel_path) const {
    for (const auto& rule : rules_) {
        if (path_matches(rule.match, rel_path)) {
            return rule.encoding;
        }
    }
    return fallback_;
}

std::string decode_source(std::string bytes, const std::string& rel_path,
                          const EncodingRules& rules) {
    if (starts_with_bytes(bytes, "\xEF\xBB\xBF")) {
        bytes.erase(0, 3);
        if (!is_valid_utf8(bytes)) {
            throw std::runtime_error(decode_error(rel_path, "UTF-8 BOM"));
        }
        return bytes;
    }
    if (starts_with_bytes(bytes, "\xFF\xFE")) {
        bytes.erase(0, 2);
        try {
            return convert_to_utf8(bytes, "UTF-16LE");
        } catch (const std::exception& error) {
            throw std::runtime_error(decode_error(rel_path, "UTF-16LE") +
                                     ": " + error.what());
        }
    }
    if (starts_with_bytes(bytes, "\xFE\xFF")) {
        bytes.erase(0, 2);
        try {
            return convert_to_utf8(bytes, "UTF-16BE");
        } catch (const std::exception& error) {
            throw std::runtime_error(decode_error(rel_path, "UTF-16BE") +
                                     ": " + error.what());
        }
    }
    if (is_valid_utf8(bytes)) {
        return bytes;
    }

    const std::string encoding = rules.resolve(rel_path);
    if (encoding.empty() || normalize_encoding(encoding) == "auto") {
        throw std::runtime_error(
            decode_error(rel_path, "auto") +
            " (no matching encoding rule and the file is not valid UTF-8)");
    }
    try {
        return convert_to_utf8(bytes, encoding);
    } catch (const std::exception& error) {
        throw std::runtime_error(decode_error(rel_path, encoding) + ": " +
                                 error.what());
    }
}

}  // namespace rag
