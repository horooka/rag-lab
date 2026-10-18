#include "rag/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>

#include "rag/util.hpp"

namespace rag {

namespace {

struct Codepoint {
    uint32_t value = 0;
    size_t length = 0;
};

Codepoint decode_utf8(std::string_view text, size_t pos) {
    const auto byte = static_cast<unsigned char>(text[pos]);
    if (byte < 0x80) {
        return {byte, 1};
    }
    int extra = 0;
    uint32_t value = 0;
    if ((byte & 0xE0) == 0xC0) {
        extra = 1;
        value = byte & 0x1F;
    } else if ((byte & 0xF0) == 0xE0) {
        extra = 2;
        value = byte & 0x0F;
    } else if ((byte & 0xF8) == 0xF0) {
        extra = 3;
        value = byte & 0x07;
    } else {
        return {byte, 1};
    }
    if (pos + static_cast<size_t>(extra) >= text.size()) {
        return {byte, 1};
    }
    for (int i = 1; i <= extra; ++i) {
        const auto next = static_cast<unsigned char>(text[pos + i]);
        if ((next & 0xC0) != 0x80) {
            return {byte, 1};
        }
        value = (value << 6) | (next & 0x3F);
    }
    return {value, static_cast<size_t>(extra + 1)};
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

bool is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == 0x0B || cp == 0x0C || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

bool is_punctuation(uint32_t cp) {
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
        (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
        return true;
    }
    return (cp >= 0x2000 && cp <= 0x206F) ||
           (cp >= 0x2E00 && cp <= 0x2E7F) ||
           (cp >= 0x3001 && cp <= 0x303F) ||
           (cp >= 0xFE10 && cp <= 0xFE1F) ||
           (cp >= 0xFE30 && cp <= 0xFE6F) ||
           (cp >= 0xFF01 && cp <= 0xFF0F) ||
           (cp >= 0xFF1A && cp <= 0xFF20) ||
           (cp >= 0xFF3B && cp <= 0xFF40) ||
           (cp >= 0xFF5B && cp <= 0xFF65);
}

bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0x3040 && cp <= 0x30FF) ||
           (cp >= 0xAC00 && cp <= 0xD7AF);
}

uint32_t strip_accent(uint32_t cp) {
    if (cp < 0xC0) {
        return cp;
    }
    // Latin-1 Supplement.
    if (cp <= 0xFF) {
        switch (cp) {
            case 0xC0: case 0xC1: case 0xC2: case 0xC3: case 0xC4: case 0xC5:
                return 'a';
            case 0xC7: return 'c';
            case 0xC8: case 0xC9: case 0xCA: case 0xCB: return 'e';
            case 0xCC: case 0xCD: case 0xCE: case 0xCF: return 'i';
            case 0xD1: return 'n';
            case 0xD2: case 0xD3: case 0xD4: case 0xD5: case 0xD6:
            case 0xD8: return 'o';
            case 0xD9: case 0xDA: case 0xDB: case 0xDC: return 'u';
            case 0xDD: return 'y';
            case 0xDF: return 's';
            case 0xE0: case 0xE1: case 0xE2: case 0xE3: case 0xE4: case 0xE5:
                return 'a';
            case 0xE7: return 'c';
            case 0xE8: case 0xE9: case 0xEA: case 0xEB: return 'e';
            case 0xEC: case 0xED: case 0xEE: case 0xEF: return 'i';
            case 0xF1: return 'n';
            case 0xF2: case 0xF3: case 0xF4: case 0xF5: case 0xF6:
            case 0xF8: return 'o';
            case 0xF9: case 0xFA: case 0xFB: case 0xFC: return 'u';
            case 0xFD: case 0xFF: return 'y';
            default: return cp;
        }
    }
    // Latin Extended-A: letters are base + diacritic.
    if (cp >= 0x100 && cp <= 0x17F) {
        if (cp <= 0x105) return 'a';
        if (cp <= 0x10D) return 'c';
        if (cp <= 0x111) return 'd';
        if (cp <= 0x11B) return 'e';
        if (cp <= 0x123) return 'g';
        if (cp <= 0x127) return 'h';
        if (cp <= 0x131) return 'i';
        if (cp <= 0x133) return 'i';
        if (cp <= 0x135) return 'j';
        if (cp <= 0x138) return 'k';
        if (cp <= 0x142) return 'l';
        if (cp <= 0x14B) return 'n';
        if (cp <= 0x151) return 'o';
        if (cp <= 0x153) return 'o';
        if (cp <= 0x159) return 'r';
        if (cp <= 0x161) return 's';
        if (cp <= 0x167) return 't';
        if (cp <= 0x173) return 'u';
        if (cp <= 0x175) return 'w';
        if (cp <= 0x17A) return 'y';
        if (cp <= 0x17E) return 'z';
        return 's';
    }
    return cp;
}

uint32_t lowercase_cp(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') {
        return cp - 'A' + 'a';
    }
    if (cp >= 0xC0 && cp <= 0xDE && cp != 0xD7) {
        return cp + 0x20;
    }
    return cp;
}

}  // namespace

BertTokenizer::BertTokenizer(const std::filesystem::path& vocab_path) {
    std::ifstream in(vocab_path);
    if (!in) {
        throw std::runtime_error("cannot open tokenizer vocab: " +
                                 vocab_path.string());
    }
    std::string token;
    int32_t index = 0;
    while (std::getline(in, token)) {
        if (!token.empty() && token.back() == '\r') {
            token.pop_back();
        }
        vocab_.emplace(token, index++);
    }
    if (vocab_.empty()) {
        throw std::runtime_error("empty tokenizer vocab: " +
                                 vocab_path.string());
    }
    if (auto it = vocab_.find("[CLS]"); it != vocab_.end()) {
        cls_id_ = it->second;
    }
    if (auto it = vocab_.find("[SEP]"); it != vocab_.end()) {
        sep_id_ = it->second;
    }
    if (auto it = vocab_.find("[UNK]"); it != vocab_.end()) {
        unk_id_ = it->second;
    }
    if (auto it = vocab_.find("[PAD]"); it != vocab_.end()) {
        pad_id_ = it->second;
    }
}

int32_t BertTokenizer::id(const std::string& token) const {
    auto it = vocab_.find(token);
    return it == vocab_.end() ? unk_id_ : it->second;
}

std::vector<std::string> BertTokenizer::basic_tokenize(
    std::string_view text) const {
    std::vector<std::string> tokens;
    std::string current;

    auto flush = [&]() {
        if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    };

    for (size_t pos = 0; pos < text.size();) {
        Codepoint cp = decode_utf8(text, pos);
        pos += cp.length;
        if (is_space(cp.value)) {
            flush();
            continue;
        }
        cp.value = strip_accent(lowercase_cp(cp.value));
        if (is_punctuation(cp.value) || is_cjk(cp.value)) {
            flush();
            std::string token;
            append_utf8(token, cp.value);
            tokens.push_back(std::move(token));
            continue;
        }
        append_utf8(current, cp.value);
    }
    flush();
    return tokens;
}

std::vector<std::string> BertTokenizer::wordpiece(
    const std::string& word) const {
    std::vector<std::string> pieces;
    if (word.size() > 100) {
        pieces.emplace_back("[UNK]");
        return pieces;
    }
    size_t start = 0;
    while (start < word.size()) {
        size_t end = word.size();
        bool found = false;
        while (start < end) {
            std::string piece = word.substr(start, end - start);
            if (start > 0) {
                piece = "##" + piece;
            }
            if (vocab_.find(piece) != vocab_.end()) {
                pieces.push_back(std::move(piece));
                found = true;
                break;
            }
            --end;
        }
        if (!found) {
            pieces.clear();
            pieces.emplace_back("[UNK]");
            return pieces;
        }
        start = end;
    }
    return pieces;
}

std::vector<int32_t> BertTokenizer::encode(std::string_view text,
                                           size_t max_length) const {
    std::vector<int32_t> ids;
    if (max_length < 2) {
        return ids;
    }
    ids.push_back(cls_id_);
    for (const auto& word : basic_tokenize(text)) {
        for (const auto& piece : wordpiece(word)) {
            auto it = vocab_.find(piece);
            ids.push_back(it == vocab_.end() ? unk_id_ : it->second);
            if (ids.size() + 1 >= max_length) {
                ids.push_back(sep_id_);
                return ids;
            }
        }
    }
    ids.push_back(sep_id_);
    return ids;
}

}  // namespace rag
