#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "coverage/coverage.hpp"
#include "rag/config.hpp"
#include "rag/embedder.hpp"
#include "rag/store.hpp"
#include "rag/util.hpp"

namespace {

using namespace rag;

constexpr const char* kVersion = "0.1.0";

void load_groups_file(const std::string& path,
                      std::vector<coverage::GroupSpec>& groups) {
    std::filesystem::path file(path);
    if (!std::filesystem::exists(file)) {
        throw std::runtime_error("groups file not found: " + path);
    }
    nlohmann::json json = nlohmann::json::parse(rag::read_file(file));

    auto add_keywords = [&groups](const std::string& label,
                                  const std::vector<std::string>& keywords) {
        coverage::GroupSpec group;
        group.label = label;
        for (const auto& keyword : keywords) {
            const std::string value = trim(keyword);
            if (!value.empty()) {
                group.keywords.push_back(value);
            }
        }
        if (group.label.empty() || group.keywords.empty()) {
            throw std::runtime_error("groups file contains an empty group");
        }
        groups.push_back(std::move(group));
    };

    if (json.is_array()) {
        for (const auto& item : json) {
            if (item.is_string()) {
                add_keywords(item.get<std::string>(),
                             {item.get<std::string>()});
                continue;
            }
            if (!item.is_object()) {
                throw std::runtime_error(
                    "groups file array items must be strings or objects");
            }
            const std::string label = item.value("label", "");
            std::vector<std::string> keywords;
            if (item.contains("keywords")) {
                for (const auto& keyword : item["keywords"]) {
                    keywords.push_back(keyword.get<std::string>());
                }
            }
            if (keywords.empty() && item.contains("description")) {
                keywords.push_back(item["description"].get<std::string>());
            }
            add_keywords(label, keywords);
        }
        return;
    }

    if (json.is_object()) {
        for (auto it = json.begin(); it != json.end(); ++it) {
            std::vector<std::string> keywords;
            if (it.value().is_array()) {
                for (const auto& keyword : it.value()) {
                    keywords.push_back(keyword.get<std::string>());
                }
            } else if (it.value().is_string()) {
                keywords.push_back(it.value().get<std::string>());
            } else {
                throw std::runtime_error(
                    "groups file values must be strings or arrays");
            }
            add_keywords(it.key(), keywords);
        }
        return;
    }

    throw std::runtime_error(
        "groups file must contain a JSON array or object");
}

std::string percent(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << value * 100.0 << "%";
    return out.str();
}

void print_text(const std::string& db, const std::string& model, int dim,
                const coverage::CoverageOptions& options,
                const coverage::CoverageResult& result, int top) {
    std::cout << "coverage: " << db << "\n"
              << "model:    " << model << " (" << dim << "d)\n"
              << "chunks:   " << result.total_chunks << "   groups: "
              << result.groups.size() << "   unit: "
              << (options.line_unit ? "lines" : "chunks") << "\n"
              << "threshold " << std::fixed << std::setprecision(2)
              << options.threshold << "   strategy: "
              << (options.max_similarity ? "max" : "centroid") << "\n\n";

    std::cout << std::left << std::setw(26) << "group" << std::right
              << std::setw(9) << "chunks" << std::setw(10) << "lines"
              << std::setw(10) << "share" << std::setw(11) << "mean_sim"
              << "\n";
    auto print_row = [](const coverage::GroupStat& group) {
        std::cout << std::left << std::setw(26) << group.label << std::right
                  << std::setw(9) << group.chunks << std::setw(10)
                  << group.lines << std::setw(10) << percent(group.share)
                  << std::setw(11) << std::fixed << std::setprecision(3)
                  << group.mean_similarity << "\n";
    };
    for (const auto& group : result.groups) {
        print_row(group);
    }
    print_row(result.unassigned);

    std::cout << "\nassigned: " << result.assigned_chunks << "/"
              << result.total_chunks << " chunks";
    if (options.line_unit) {
        std::cout << ", " << result.assigned_lines << "/" << result.total_lines
                  << " lines";
    } else {
        std::cout << ", " << result.assigned_lines << " spanned lines";
    }
    std::cout << "\n";

    if (options.details && !result.matches.empty()) {
        std::cout << "\nbest matches:\n";
        for (const auto& group : result.groups) {
            std::cout << "[" << group.label << "]\n";
            int shown = 0;
            for (const auto& match : result.matches) {
                if (match.label != group.label) {
                    continue;
                }
                std::cout << "  " << std::fixed << std::setprecision(3)
                          << match.similarity << "  " << match.path << ":"
                          << match.start_line << "-" << match.end_line;
                if (!match.symbol.empty()) {
                    std::cout << "  " << match.symbol;
                }
                if (!match.keyword.empty()) {
                    std::cout << "  (kw: " << match.keyword << ")";
                }
                std::cout << "\n";
                if (++shown >= top) {
                    break;
                }
            }
            if (shown == 0) {
                std::cout << "  (none)\n";
            }
        }
    }
}

void print_json(const std::string& db, const std::string& model, int dim,
                const coverage::CoverageOptions& options,
                const coverage::CoverageResult& result) {
    nlohmann::json out;
    out["db"] = db;
    out["model"] = model;
    out["dim"] = dim;
    out["unit"] = options.line_unit ? "lines" : "chunks";
    out["threshold"] = options.threshold;
    out["strategy"] = options.max_similarity ? "max" : "centroid";

    out["totals"] = {
        {"chunks", result.total_chunks},
        {"lines", result.total_lines},
        {"assigned_chunks", result.assigned_chunks},
        {"assigned_lines", result.assigned_lines},
        {"chunk_share",
         result.total_chunks > 0
             ? static_cast<double>(result.assigned_chunks) /
                   static_cast<double>(result.total_chunks)
             : 0.0},
        {"line_share",
         result.total_lines > 0
             ? static_cast<double>(result.assigned_lines) /
                   static_cast<double>(result.total_lines)
             : 0.0},
    };

    auto group_json = [](const coverage::GroupStat& group) {
        nlohmann::json item;
        item["label"] = group.label;
        item["chunks"] = group.chunks;
        item["lines"] = group.lines;
        item["share"] = group.share;
        item["mean_similarity"] = group.mean_similarity;
        return item;
    };

    out["groups"] = nlohmann::json::array();
    for (const auto& group : result.groups) {
        out["groups"].push_back(group_json(group));
    }
    out["unassigned"] = group_json(result.unassigned);

    if (options.details) {
        out["matches"] = nlohmann::json::array();
        for (const auto& match : result.matches) {
            nlohmann::json item;
            item["path"] = match.path;
            item["symbol"] = match.symbol;
            item["label"] = match.label;
            item["keyword"] = match.keyword;
            item["start_line"] = match.start_line;
            item["end_line"] = match.end_line;
            item["similarity"] = match.similarity;
            out["matches"].push_back(std::move(item));
        }
    }
    std::cout << out.dump(2) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{
        "Measure how much of an indexed codebase belongs to supplied "
        "semantic groups",
        "coverage"};
    app.set_version_flag("--version", kVersion);
    app.footer(
        "Examples:\n"
        "  coverage --db indexes/Project.db \\\n"
        "    --group 'network=socket,packet,rtp,connection' \\\n"
        "    --group 'database=sql,query,transaction,table'\n"
        "  coverage --db indexes/Project.db --groups-file groups.json "
        "--unit lines --details");

    std::string db_path;
    std::string model_dir;
    std::string groups_file;
    std::vector<std::string> group_args;
    std::string unit = "chunks";
    std::string strategy = "max";
    double threshold = 0.0;
    int threads = 0;
    bool details = false;
    int top = 5;
    bool json_output = false;

    app.add_option("--db", db_path, "Index database (required)")->required();
    app.add_option("--model-dir", model_dir,
                   "Embedding model directory (default: index root model)");
    app.add_option("-g,--group", group_args,
                   "Group as 'label=keyword1,keyword2' (repeatable)")
        ->expected(1, -1);
    app.add_option("--groups-file", groups_file,
                   "JSON array/object of groups");
    app.add_option("--unit", unit, "chunks or lines")
        ->check(CLI::IsMember({"chunks", "lines"}));
    app.add_option("--strategy", strategy, "max or centroid")
        ->check(CLI::IsMember({"max", "centroid"}));
    app.add_option("--threshold", threshold,
                   "Minimum cosine similarity for assignment")
        ->check(CLI::Range(-1.0, 1.0));
    app.add_option("--threads", threads, "ONNX Runtime threads (0=auto)");
    app.add_flag("--details", details, "Include the best matching chunks");
    app.add_option("--top", top, "Matches to show per group with --details")
        ->check(CLI::Range(1, 1000));
    app.add_flag("--json", json_output, "Print machine-readable JSON");

    CLI11_PARSE(app, argc, argv);

    try {
        std::vector<coverage::GroupSpec> groups;
        for (const auto& argument : group_args) {
            groups.push_back(coverage::parse_group(argument));
        }
        if (!groups_file.empty()) {
            load_groups_file(groups_file, groups);
        }
        if (groups.empty()) {
            throw std::runtime_error(
                "at least one --group or --groups-file is required");
        }
        std::unordered_set<std::string> labels;
        for (const auto& group : groups) {
            if (!labels.insert(group.label).second) {
                throw std::runtime_error("duplicate group label: " +
                                         group.label);
            }
        }

        const std::filesystem::path db = std::filesystem::absolute(db_path);
        if (!std::filesystem::exists(db)) {
            throw std::runtime_error("index not found: " + db.string());
        }
        auto store = open_sqlite_store(db);

        const std::string root = store->meta("root");
        const std::filesystem::path root_path =
            root.empty() ? std::filesystem::current_path()
                         : std::filesystem::path(root);
        Config config = load_config(root_path);
        if (!model_dir.empty()) {
            config.model_dir = std::filesystem::absolute(model_dir);
        } else {
            config.model_dir = resolve_model_dir(root_path);
        }

        auto embedder =
            make_onnx_embedder(config.model_dir, 8, threads);
        store->initialize(embedder->dim());
        const std::string index_model = store->meta("model");
        if (!index_model.empty() && index_model != embedder->spec().name) {
            throw std::runtime_error(
                "index model mismatch: index uses '" + index_model +
                "', current model is '" + embedder->spec().name + "'");
        }

        auto chunks = store->all_chunks();
        if (chunks.empty()) {
            throw std::runtime_error("index contains no chunks");
        }

        coverage::CoverageOptions options;
        options.line_unit = unit == "lines";
        options.max_similarity = strategy == "max";
        options.threshold = threshold;
        options.details = details;
        options.details_limit = static_cast<std::size_t>(top);

        const auto result =
            coverage::analyze(chunks, groups, *embedder, options);
        if (json_output) {
            print_json(db.string(), embedder->spec().name, embedder->dim(),
                       options, result);
        } else {
            print_text(db.string(), embedder->spec().name, embedder->dim(),
                       options, result, top);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
