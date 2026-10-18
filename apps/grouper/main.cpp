#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "grouper/clustering.hpp"
#include "grouper/input.hpp"
#include "rag/embedder.hpp"

#ifndef GROUPER_DEFAULT_MODEL_DIR
#define GROUPER_DEFAULT_MODEL_DIR ""
#endif

namespace {

constexpr const char* kVersion = "0.1.0";

std::filesystem::path resolve_model_dir(const std::string& explicit_dir) {
    if (!explicit_dir.empty()) {
        return std::filesystem::absolute(explicit_dir);
    }
    for (const char* variable : {"GROUPER_MODEL_DIR", "RAG_MODEL_DIR"}) {
        if (const char* value = std::getenv(variable); value && *value) {
            return std::filesystem::absolute(value);
        }
    }
    std::error_code error;
    const std::filesystem::path local = "models/bge-small-en-v1.5";
    if (std::filesystem::exists(local / "model.json", error) ||
        std::filesystem::exists(local / "vocab.txt", error)) {
        return std::filesystem::absolute(local);
    }
    if (std::string(GROUPER_DEFAULT_MODEL_DIR).size() > 0) {
        return std::filesystem::path(GROUPER_DEFAULT_MODEL_DIR);
    }
    return local;
}

grouper::Linkage parse_linkage(const std::string& value) {
    if (value == "complete") {
        return grouper::Linkage::Complete;
    }
    if (value == "single") {
        return grouper::Linkage::Single;
    }
    return grouper::Linkage::Average;
}

std::string plural(std::size_t count, const char* singular,
                   const char* plural_form) {
    return std::to_string(count) + " " +
           (count == 1 ? singular : plural_form);
}

void print_text(const grouper::ClusterResult& result,
                const std::vector<std::string>& items) {
    std::cout << plural(result.clusters.size(), "group", "groups") << " from "
              << plural(items.size(), "item", "items") << "\n\n";

    std::size_t id = 1;
    for (const auto& cluster : result.clusters) {
        std::cout << "Group " << id++ << "  ("
                  << plural(cluster.members.size(), "item", "items")
                  << ", cohesion " << std::fixed << std::setprecision(3)
                  << cluster.cohesion << ")\n";
        for (std::size_t index : cluster.members) {
            std::cout << "  " << items[index] << "\n";
        }
        std::cout << "\n";
    }
}

void print_json(const grouper::ClusterResult& result,
                const std::vector<std::string>& items, const std::string& model,
                int dim) {
    nlohmann::json output;
    output["model"] = model;
    output["dim"] = dim;
    output["items"] = items.size();
    output["groups"] = result.clusters.size();
    output["mode"] = result.mode;
    output["merges"] = result.merges;
    output["cut_distance"] = result.cut_distance;
    output["clusters"] = nlohmann::json::array();

    std::size_t id = 1;
    for (const auto& cluster : result.clusters) {
        nlohmann::json entry;
        entry["id"] = id++;
        entry["size"] = cluster.members.size();
        entry["cohesion"] = cluster.cohesion;
        entry["representative"] = items[cluster.representative];
        entry["items"] = nlohmann::json::array();
        for (std::size_t index : cluster.members) {
            entry["items"].push_back(items[index]);
        }
        output["clusters"].push_back(std::move(entry));
    }
    std::cout << output.dump(2) << "\n";
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{
        "Group texts or keywords into semantic clusters using local embeddings",
        "grouper"};
    app.set_version_flag("--version", kVersion);
    app.footer(
        "Examples:\n"
        "  printf 'apple\\npear\\ncar\\n' | grouper\n"
        "  grouper --groups 3 apple pear banana carrot beef chicken\n"
        "  grouper --threshold 0.30 --file keywords.txt");

    std::vector<std::string> items;
    std::vector<std::string> files;
    int groups = 0;
    double threshold = -1.0;
    int max_groups = 0;
    std::string linkage = "average";
    bool keep_duplicates = false;
    std::string delimiter;
    std::string model_dir;
    int threads = 0;
    int batch_size = 8;
    bool json_output = false;
    bool verbose = false;

    app.add_option("items", items,
                   "Items to group (default: read newline-separated stdin)")
        ->expected(-1);
    app.add_option("-f,--file", files,
                   "Read newline-separated items from a file ('-' = stdin; "
                   "repeatable)")
        ->expected(1, -1)
        ->type_size(1)
        ->allow_extra_args(false);
    app.add_option("-n,--groups", groups,
                   "Exact number of groups (0 = choose automatically)")
        ->check(CLI::Range(0, 1000000));
    app.add_option("-t,--threshold", threshold,
                   "Cosine-distance cutoff for automatic grouping")
        ->check(CLI::Range(0.0, 2.0));
    app.add_option("--max-groups", max_groups,
                   "Upper bound explored in auto mode (0 = min(items, 32))")
        ->check(CLI::Range(0, 1000000));
    app.add_option("--linkage", linkage, "average, complete or single")
        ->check(CLI::IsMember({"average", "complete", "single"}));
    app.add_flag("--keep-duplicates", keep_duplicates,
                 "Keep exact duplicate items instead of deduplicating");
    app.add_option("-d,--delimiter", delimiter,
                   "Split each input line on this single character");
    app.add_option("--model-dir", model_dir,
                   "Embedding model directory (default: rag-lab model)");
    app.add_option("--threads", threads,
                   "ONNX Runtime threads (0 = auto)")
        ->check(CLI::Range(0, 1024));
    app.add_option("--batch-size", batch_size, "Embedding batch size")
        ->check(CLI::Range(1, 1024));
    app.add_flag("--json", json_output, "Print machine-readable JSON");
    app.add_flag("-v,--verbose", verbose,
                 "Print model and timing details to stderr");

    CLI11_PARSE(app, argc, argv);

    try {
        if (groups > 0 && threshold >= 0.0) {
            throw std::runtime_error(
                "--groups and --threshold are mutually exclusive");
        }

        grouper::InputOptions input_options;
        input_options.unique = !keep_duplicates;
        if (!delimiter.empty()) {
            if (delimiter.size() != 1) {
                throw std::runtime_error(
                    "--delimiter must be a single character");
            }
            input_options.delimiter = delimiter.front();
        }

        if (items.empty() && files.empty() && isatty(STDIN_FILENO)) {
            throw std::runtime_error(
                "no input items; pass items, --file, or pipe text on stdin");
        }

        const auto inputs =
            grouper::read_inputs(items, files, input_options);
        if (inputs.empty()) {
            throw std::runtime_error("no non-empty input items");
        }
        if (groups > static_cast<int>(inputs.size())) {
            throw std::runtime_error(
                "requested " + std::to_string(groups) +
                " groups but only " + std::to_string(inputs.size()) +
                " items were provided");
        }

        const auto model_path = resolve_model_dir(model_dir);
        const auto embed_start = std::chrono::steady_clock::now();
        auto embedder =
            rag::make_onnx_embedder(model_path, batch_size, threads);
        auto embeddings = embedder->embed_documents(inputs);
        const double embed_ms = elapsed_ms(embed_start);

        grouper::ClusterOptions cluster_options;
        cluster_options.groups = groups;
        cluster_options.threshold = threshold;
        cluster_options.max_groups = max_groups;
        cluster_options.linkage = parse_linkage(linkage);

        const auto cluster_start = std::chrono::steady_clock::now();
        auto result = grouper::cluster_embeddings(embeddings, cluster_options);
        const double cluster_ms = elapsed_ms(cluster_start);

        if (verbose) {
            std::cerr << "model:   " << embedder->spec().name << " ("
                      << embedder->dim() << "d)\n"
                      << "items:   " << inputs.size() << "\n"
                      << "mode:    " << result.mode << "\n"
                      << "linkage: " << grouper::linkage_name(
                                            cluster_options.linkage)
                      << "\n"
                      << "embed:   " << std::fixed << std::setprecision(1)
                      << embed_ms << " ms\n"
                      << "cluster: " << cluster_ms << " ms\n";
        }

        if (json_output) {
            print_json(result, inputs, embedder->spec().name,
                       embedder->dim());
        } else {
            print_text(result, inputs);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
