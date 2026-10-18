#include "rag/util.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <limits>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rag {

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open file: " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string lower_ascii(std::string_view text) {
    std::string out(text);
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return out;
}

std::string trim(std::string_view text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end &&
           std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

std::string collapse_ws(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool in_space = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!in_space && !out.empty()) {
                out.push_back(' ');
            }
            in_space = true;
        } else {
            out.push_back(c);
            in_space = false;
        }
    }
    return trim(out);
}

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() &&
           text.substr(0, prefix.size()) == prefix;
}

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.substr(text.size() - suffix.size()) == suffix;
}

std::vector<std::string> split(std::string_view text, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= text.size()) {
        size_t pos = text.find(sep, start);
        if (pos == std::string_view::npos) {
            out.emplace_back(text.substr(start));
            break;
        }
        out.emplace_back(text.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

std::string join(const std::vector<std::string>& parts,
                 std::string_view sep) {
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) {
            out += sep;
        }
        out += parts[i];
    }
    return out;
}

uint64_t fnv1a64(std::string_view data, uint64_t seed) {
    uint64_t hash = seed;
    for (unsigned char c : data) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string hex_u64(uint64_t value) {
    static const char* digits = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[i] = digits[value & 0xF];
        value >>= 4;
    }
    return out;
}

int64_t now_unix() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

int64_t parse_iso8601_utc(std::string_view text) {
    std::string value = trim(text);
    if (value.empty()) {
        throw std::runtime_error("empty ISO timestamp");
    }
    if (value.back() == 'Z' || value.back() == 'z') {
        value.pop_back();
    }
    if (value.size() > 10 && value[10] == ' ') {
        value[10] = 'T';
    }

    static const char* formats[] = {
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    };
    std::tm tm{};
    bool parsed = false;
    for (const char* format : formats) {
        std::istringstream stream(value);
        stream >> std::get_time(&tm, format);
        if (stream.fail()) {
            tm = std::tm{};
            continue;
        }
        stream >> std::ws;
        if (stream.peek() == std::char_traits<char>::eof()) {
            parsed = true;
            break;
        }
        tm = std::tm{};
    }
    if (!parsed) {
        throw std::runtime_error(
            "invalid ISO timestamp (use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ): " +
            std::string(text));
    }
    const std::time_t seconds = timegm(&tm);
    if (seconds == static_cast<std::time_t>(-1)) {
        throw std::runtime_error("timestamp out of range: " +
                                 std::string(text));
    }
    constexpr int64_t kNanos = 1000000000LL;
    if (seconds > std::numeric_limits<int64_t>::max() / kNanos ||
        seconds < std::numeric_limits<int64_t>::min() / kNanos) {
        throw std::runtime_error("timestamp out of nanosecond range: " +
                                 std::string(text));
    }
    return static_cast<int64_t>(seconds) * kNanos;
}

std::string format_bytes(uint64_t bytes) {
    static const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream ss;
    if (unit == 0) {
        ss << bytes << ' ' << units[unit];
    } else {
        ss.setf(std::ios::fixed);
        ss.precision(1);
        ss << value << ' ' << units[unit];
    }
    return ss.str();
}

std::string truncate_utf8(std::string_view text, size_t max_bytes) {
    if (text.size() <= max_bytes) {
        return std::string(text);
    }
    size_t end = max_bytes;
    // Do not split a UTF-8 sequence.
    while (end > 0 &&
           (static_cast<unsigned char>(text[end]) & 0xC0) == 0x80) {
        --end;
    }
    return std::string(text.substr(0, end));
}

namespace {

bool wildcard_impl(std::string_view pattern, std::string_view text,
                   bool path_mode) {
    size_t p = 0;
    size_t t = 0;
    size_t star = std::string_view::npos;
    size_t match = 0;

    while (t < text.size()) {
        if (p < pattern.size() && (pattern[p] == '?' ||
                                   pattern[p] == text[t])) {
            ++p;
            ++t;
        } else if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            match = t;
        } else if (star != std::string_view::npos) {
            p = star + 1;
            t = ++match;
        } else {
            return false;
        }
        if (path_mode && p < pattern.size() && pattern[p] == '/' &&
            t < text.size() && text[t] == '/') {
            // regular loop handles it
        }
    }
    while (p < pattern.size() && pattern[p] == '*') {
        ++p;
    }
    return p == pattern.size();
}

}  // namespace

bool wildcard_match(std::string_view pattern, std::string_view text) {
    return wildcard_impl(pattern, text, false);
}

}  // namespace rag
