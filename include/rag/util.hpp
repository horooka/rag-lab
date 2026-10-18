#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace rag {

namespace fs = std::filesystem;

std::string read_file(const fs::path& path);
std::string lower_ascii(std::string_view text);
std::string trim(std::string_view text);
std::string collapse_ws(std::string_view text);
bool starts_with(std::string_view text, std::string_view prefix);
bool ends_with(std::string_view text, std::string_view suffix);
std::vector<std::string> split(std::string_view text, char sep);
std::string join(const std::vector<std::string>& parts, std::string_view sep);

uint64_t fnv1a64(std::string_view data, uint64_t seed = 1469598103934665603ULL);
std::string hex_u64(uint64_t value);
int64_t now_unix();

// Parses YYYY-MM-DD, YYYY-MM-DDTHH:MM or YYYY-MM-DDTHH:MM:SS (UTC).
// A trailing Z is accepted. Returns Unix nanoseconds.
int64_t parse_iso8601_utc(std::string_view text);
std::string format_bytes(uint64_t bytes);
std::string truncate_utf8(std::string_view text, size_t max_bytes);

// Simple glob matching. '*' and '?' are supported; matching is path-aware.
bool wildcard_match(std::string_view pattern, std::string_view text);

}  // namespace rag
