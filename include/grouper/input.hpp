#pragma once

#include <string>
#include <vector>

namespace grouper {

struct InputOptions {
    // Drop exact duplicates while preserving first-seen order.
    bool unique = true;

    // Split every input line/argument on this character. '\0' disables
    // splitting, so one line is one item.
    char delimiter = '\0';
};

// Collects items from positional arguments and newline-separated files.
// A file named "-" reads standard input. When both lists are empty, standard
// input is read. Blank items are ignored; surrounding whitespace is trimmed.
std::vector<std::string> read_inputs(
    const std::vector<std::string>& positional,
    const std::vector<std::string>& files,
    const InputOptions& options);

}  // namespace grouper
