#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "rag/chunker.hpp"
#include "rag/config.hpp"
#include "rag/embedder.hpp"
#include "rag/eval.hpp"
#include "rag/indexer.hpp"
#include "rag/search.hpp"
#include "rag/source.hpp"
#include "rag/store.hpp"
#include "rag/util.hpp"

namespace {

using namespace rag;

constexpr const char* kVersion = "0.1.0";

std::string format_time(int64_t epoch) {
    if (epoch <= 0) {
        return "never";
    }
    std::time_t value = static_cast<std::time_t>(epoch);
    std::tm local{};
    localtime_r(&value, &local);
    char buffer[64];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &local);
    return buffer;
}

struct CommonOptions {
    std::string root = ".";
    std::string db;
    std::string model_dir;
};

Config make_config(const CommonOptions& common, const std::string& model_dir_opt,
                   int threads, int batch_size, int window_lines,
                   int overlap_lines, int max_tokens,
                   const std::vector<std::string>& excludes) {
    Config config = load_config(common.root);
    if (!common.db.empty()) {
        config.db_path = std::filesystem::absolute(common.db);
    }
    if (!model_dir_opt.empty()) {
        config.model_dir = std::filesystem::absolute(model_dir_opt);
    } else {
        config.model_dir = resolve_model_dir(config.root);
    }
    if (threads > 0) {
        config.threads = threads;
    }
    if (batch_size > 0) {
        config.batch_size = batch_size;
    }
    if (window_lines > 0) {
        config.chunk.window_lines = window_lines;
    }
    if (overlap_lines >= 0) {
        config.chunk.overlap_lines = overlap_lines;
    }
    if (max_tokens > 0) {
        config.chunk.max_tokens = max_tokens;
    }
    for (const auto& exclude : excludes) {
        config.exclude_globs.push_back(exclude);
    }
    return config;
}

void require_index(const std::filesystem::path& db_path) {
    std::error_code ec;
    if (!std::filesystem::exists(db_path, ec)) {
        throw std::runtime_error("index not found: " + db_path.string() +
                                 "\nRun: rag index <path>");
    }
}

std::unique_ptr<Store> open_index(const Config& config, Embedder& embedder) {
    require_index(config.db_path);
    auto store = open_sqlite_store(config.db_path);
    store->initialize(embedder.dim());
    const std::string model = store->meta("model");
    if (!model.empty() && model != embedder.spec().name) {
        throw std::runtime_error("index model mismatch: index uses '" + model +
                                 "', current model is '" +
                                 embedder.spec().name +
                                 "'; rebuild with --full");
    }
    return store;
}

std::string snippet(const std::string& text, int max_lines = 10) {
    std::string out;
    int lines = 0;
    for (char c : text) {
        out.push_back(c);
        if (c == '\n' && ++lines >= max_lines) {
            out += "...";
            break;
        }
    }
    return truncate_utf8(out, 1600);
}

int run_index(const CommonOptions& common, const std::string& model_dir_opt,
              int threads, int batch_size, int window_lines,
              int overlap_lines, int max_tokens,
              const std::vector<std::string>& excludes, bool full,
              bool quiet) {
    Config config = make_config(common, model_dir_opt, threads, batch_size,
                                window_lines, overlap_lines, max_tokens,
                                excludes);
    FilesystemSource source(config.root, config.exclude_globs,
                            config.include_docs, config.max_file_bytes);
    auto chunker =
        make_chunker_registry(config.chunk, config.chunker_profiles);
    auto embedder =
        make_onnx_embedder(config.model_dir, config.batch_size, config.threads);
    auto store = open_sqlite_store(config.db_path);

    Indexer indexer(config, source, *chunker, *embedder, *store);
    IndexResult result = indexer.run(full, quiet);

    IndexStats stats = store->stats();
    std::cout << "indexed " << config.root.string() << "\n"
              << "  model:     " << embedder->spec().name << " ("
              << embedder->dim() << "d)\n"
              << "  database:  " << config.db_path.string() << "\n"
              << "  scanned:   " << result.scanned << " files\n"
              << "  added:     " << result.added << "  updated: "
              << result.updated << "  unchanged: " << result.unchanged
              << "  removed: " << result.removed << "\n"
              << "  chunks:    " << result.chunks_written
              << " written, " << stats.chunks << " total\n"
              << "  failed:    " << result.failed << "\n"
              << "  elapsed:   " << std::fixed << std::setprecision(2)
              << result.seconds << " s\n";
    return result.failed > 0 ? 2 : 0;
}

int run_search(const CommonOptions& common, const std::string& model_dir_opt,
               const std::string& query, const std::string& mode, int k,
               int candidate_k, double dense_weight, double bm25_weight,
               bool no_auto_weights, const std::string& repo,
               const std::string& lang, const std::string& path_prefix,
               const std::string& after, const std::string& before,
               int threads, bool json_output) {
    Config config = make_config(common, model_dir_opt, threads, 0, 0, -1, 0, {});
    auto embedder =
        make_onnx_embedder(config.model_dir, config.batch_size, config.threads);
    auto store = open_index(config, *embedder);
    if ((!after.empty() || !before.empty()) &&
        store->meta("mtime_epoch") != "unix") {
        throw std::runtime_error(
            "this index predates Unix mtime tracking; rebuild it with --full "
            "before using --after/--before");
    }

    SearchOptions options;
    options.mode = mode;
    options.k = k;
    options.candidate_k = candidate_k;
    if (dense_weight >= 0.0) {
        options.dense_weight = dense_weight;
        options.auto_weights = false;
    }
    if (bm25_weight >= 0.0) {
        options.bm25_weight = bm25_weight;
        options.auto_weights = false;
    }
    if (no_auto_weights) {
        options.auto_weights = false;
    }
    options.filters.repo = repo;
    options.filters.lang = lang;
    options.filters.path_prefix = path_prefix;
    if (!after.empty()) {
        options.filters.after_ns = parse_iso8601_utc(after);
    }
    if (!before.empty()) {
        options.filters.before_ns = parse_iso8601_utc(before);
    }

    Searcher searcher(*store, *embedder, options);
    SearchResult result = searcher.search(query);

    if (json_output) {
        nlohmann::json out;
        out["query"] = query;
        out["mode"] = mode;
        out["k"] = k;
        out["embed_ms"] = result.embed_ms;
        out["dense_ms"] = result.dense_ms;
        out["bm25_ms"] = result.bm25_ms;
        out["total_ms"] = result.total_ms;
        out["results"] = nlohmann::json::array();
        for (const auto& hit : result.hits) {
            nlohmann::json item;
            item["path"] = hit.chunk.path;
            item["repo"] = hit.chunk.repo;
            item["lang"] = hit.chunk.lang;
            item["symbol"] = hit.chunk.symbol;
            item["kind"] = hit.chunk.kind;
            item["start_line"] = hit.chunk.start_line;
            item["end_line"] = hit.chunk.end_line;
            item["score"] = hit.score;
            item["dense_score"] = hit.dense_score;
            item["dense_rank"] = hit.dense_rank;
            item["bm25_score"] = hit.bm25_score;
            item["bm25_rank"] = hit.bm25_rank;
            item["text"] = hit.chunk.text;
            out["results"].push_back(std::move(item));
        }
        std::cout << out.dump(2) << "\n";
        return 0;
    }

    std::cout << "query: " << query << "\n"
              << "mode:  " << mode << "   hits: " << result.hits.size()
              << "   time: " << std::fixed << std::setprecision(1)
              << result.total_ms << " ms";
    if (result.embed_ms > 0) {
        std::cout << " (embed " << result.embed_ms << " ms)";
    }
    std::cout << "\n\n";
    int rank = 0;
    for (const auto& hit : result.hits) {
        ++rank;
        std::cout << rank << ". " << hit.chunk.path << ":" << hit.chunk.start_line
                  << "-" << hit.chunk.end_line;
        if (!hit.chunk.symbol.empty()) {
            std::cout << "  " << hit.chunk.symbol;
        }
        std::cout << "\n   score " << std::setprecision(5) << hit.score;
        if (hit.dense_rank > 0) {
            std::cout << "  dense #" << hit.dense_rank;
        }
        if (hit.bm25_rank > 0) {
            std::cout << "  bm25 #" << hit.bm25_rank;
        }
        std::cout << "\n";
        std::cout << "   " << snippet(hit.chunk.text) << "\n\n";
    }
    if (result.hits.empty()) {
        std::cout << "no results\n";
    }
    return 0;
}

int run_stats(const CommonOptions& common, const std::string& model_dir_opt,
              int threads, bool json_output) {
    Config config = make_config(common, model_dir_opt, threads, 0, 0, -1, 0, {});
    auto embedder =
        make_onnx_embedder(config.model_dir, config.batch_size, config.threads);
    auto store = open_index(config, *embedder);
    IndexStats stats = store->stats();

    if (json_output) {
        nlohmann::json out;
        out["root"] = stats.root;
        out["model"] = stats.model;
        out["indexed_at"] = stats.indexed_at;
        out["files"] = stats.files;
        out["chunks"] = stats.chunks;
        out["db_bytes"] = stats.db_bytes;
        for (const auto& [repo, count] : stats.repos) {
            out["repos"][repo] = count;
        }
        for (const auto& [lang, count] : stats.languages) {
            out["languages"][lang] = count;
        }
        std::cout << out.dump(2) << "\n";
        return 0;
    }

    std::cout << "index:     " << config.db_path.string() << " ("
              << format_bytes(stats.db_bytes) << ")\n"
              << "root:      " << stats.root << "\n"
              << "model:     " << stats.model << "\n"
              << "indexed:   " << format_time(stats.indexed_at) << "\n"
              << "files:     " << stats.files << "\n"
              << "chunks:    " << stats.chunks << "\n";
    std::cout << "repos:\n";
    for (const auto& [repo, count] : stats.repos) {
        std::cout << "  " << std::setw(28) << std::left << repo << count << "\n";
    }
    std::cout << "languages:\n";
    for (const auto& [lang, count] : stats.languages) {
        std::cout << "  " << std::setw(28) << std::left << lang << count << "\n";
    }
    return 0;
}

int run_eval(const CommonOptions& common, const std::string& model_dir_opt,
             const std::string& queries_path, const std::string& mode, int k,
             int threads, bool json_output, bool details) {
    Config config = make_config(common, model_dir_opt, threads, 0, 0, -1, 0, {});
    auto embedder =
        make_onnx_embedder(config.model_dir, config.batch_size, config.threads);
    auto store = open_index(config, *embedder);
    auto queries = load_eval_queries(queries_path);

    std::vector<std::string> modes;
    if (mode == "all") {
        modes = {"dense", "bm25", "hybrid"};
    } else {
        modes = {mode};
    }

    std::vector<EvalSummary> summaries;
    for (const auto& current : modes) {
        SearchOptions options;
        options.mode = current;
        options.k = k;
        options.candidate_k = std::max(50, k * 5);
        Searcher searcher(*store, *embedder, options);
        summaries.push_back(evaluate(searcher, queries, k));
    }

    if (json_output) {
        nlohmann::json out;
        out["queries_file"] = queries_path;
        out["k"] = k;
        out["results"] = nlohmann::json::array();
        for (const auto& summary : summaries) {
            nlohmann::json item;
            item["mode"] = summary.mode;
            item["queries"] = summary.queries;
            item["recall_at_k"] = summary.recall;
            item["mrr"] = summary.mrr;
            item["hit_rate"] = summary.hit_rate;
            item["details"] = nlohmann::json::array();
            for (const auto& detail : summary.details) {
                nlohmann::json row;
                row["query"] = detail.query;
                row["recall"] = detail.recall;
                row["reciprocal_rank"] = detail.reciprocal_rank;
                row["first_relevant_rank"] = detail.first_relevant_rank;
                row["retrieved"] = detail.retrieved;
                item["details"].push_back(std::move(row));
            }
            out["results"].push_back(std::move(item));
        }
        std::cout << out.dump(2) << "\n";
        return 0;
    }

    std::cout << "queries: " << queries.size() << "   k: " << k << "\n\n";
    std::cout << std::left << std::setw(10) << "mode" << std::right
              << std::setw(12) << "recall@k" << std::setw(10) << "mrr"
              << std::setw(10) << "hit@k" << "\n";
    for (const auto& summary : summaries) {
        std::cout << std::left << std::setw(10) << summary.mode << std::right
                  << std::fixed << std::setprecision(3) << std::setw(12)
                  << summary.recall << std::setw(10) << summary.mrr
                  << std::setw(10) << summary.hit_rate << "\n";
    }

    if (details) {
        for (const auto& summary : summaries) {
            std::cout << "\n[" << summary.mode << "]\n";
            for (const auto& detail : summary.details) {
                std::cout << "  "
                          << (detail.first_relevant_rank > 0
                                  ? "rank " +
                                        std::to_string(
                                            detail.first_relevant_rank)
                                  : "MISS")
                          << "  " << detail.query << "\n";
            }
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"rag-lab: semantic code search over a local tree"};
    app.set_version_flag("--version", kVersion);
    app.require_subcommand(1, 1);

    auto* index = app.add_subcommand("index", "Build or incrementally update the index");
    std::string index_path;
    bool full = false;
    bool quiet = false;
    std::string index_db;
    std::string index_model;
    int index_threads = 0;
    int index_batch = 0;
    int window_lines = 0;
    int overlap_lines = -1;
    int max_tokens = 0;
    std::vector<std::string> excludes;
    index->add_option("path", index_path,
                      "Project or tree to index (required)")
        ->required();
    index->add_option("--db", index_db, "Index database path");
    index->add_option("--model-dir", index_model, "Embedding model directory");
    index->add_option("--threads", index_threads, "ONNX Runtime threads (0=auto)");
    index->add_option("--batch-size", index_batch, "Embedding batch size");
    index->add_option("--window-lines", window_lines, "Fallback window size");
    index->add_option("--overlap-lines", overlap_lines, "Fallback window overlap");
    index->add_option("--max-tokens", max_tokens, "Max tokens per chunk");
    index->add_option("--exclude", excludes, "Additional exclude glob (repeatable)");
    index->add_flag("--full", full, "Drop and rebuild the index");
    index->add_flag("--quiet", quiet, "Suppress per-file progress");

    auto* search = app.add_subcommand("search", "Search the index");
    std::string query;
    std::string search_root = ".";
    std::string search_db;
    std::string search_model;
    std::string mode = "hybrid";
    int k = 10;
    int candidate_k = 50;
    double dense_weight = -1.0;
    double bm25_weight = -1.0;
    bool no_auto_weights = false;
    std::string repo;
    std::string lang;
    std::string path_prefix;
    std::string after;
    std::string before;
    int search_threads = 0;
    bool search_json = false;
    search->add_option("query", query, "Natural-language or identifier query")->required();
    search->add_option("--root", search_root, "Indexed root (default: cwd)");
    search->add_option("--db", search_db, "Index database path");
    search->add_option("--model-dir", search_model, "Embedding model directory");
    search->add_option("--mode", mode, "hybrid, dense or bm25")->check(CLI::IsMember({"hybrid", "dense", "bm25"}));
    search->add_option("-k,--top-k", k, "Number of results");
    search->add_option("--candidate-k", candidate_k, "Candidates per retriever");
    search->add_option("--dense-weight", dense_weight, "RRF weight for dense retrieval");
    search->add_option("--bm25-weight", bm25_weight, "RRF weight for BM25 retrieval");
    search->add_flag("--no-auto-weights", no_auto_weights, "Disable query-adaptive weights");
    search->add_option("--repo", repo, "Filter by repository name");
    search->add_option("--lang", lang, "Filter by language");
    search->add_option("--path", path_prefix, "Filter by path prefix");
    search->add_option("--after", after,
                       "Only files modified at/after this ISO-8601 UTC time");
    search->add_option("--before", before,
                       "Only files modified at/before this ISO-8601 UTC time");
    search->add_option("--threads", search_threads, "ONNX Runtime threads (0=auto)");
    search->add_flag("--json", search_json, "Print JSON");

    auto* stats = app.add_subcommand("stats", "Show index statistics");
    std::string stats_root = ".";
    std::string stats_db;
    std::string stats_model;
    int stats_threads = 0;
    bool stats_json = false;
    stats->add_option("--root", stats_root, "Indexed root (default: cwd)");
    stats->add_option("--db", stats_db, "Index database path");
    stats->add_option("--model-dir", stats_model, "Embedding model directory");
    stats->add_option("--threads", stats_threads, "ONNX Runtime threads (0=auto)");
    stats->add_flag("--json", stats_json, "Print JSON");

    auto* eval = app.add_subcommand("eval", "Evaluate retrieval quality");
    std::string eval_queries;
    std::string eval_root = ".";
    std::string eval_db;
    std::string eval_model;
    std::string eval_mode = "all";
    int eval_k = 10;
    int eval_threads = 0;
    bool eval_json = false;
    bool eval_details = false;
    eval->add_option("queries", eval_queries, "JSONL file with query/relevant pairs")->required();
    eval->add_option("--root", eval_root, "Indexed root (default: cwd)");
    eval->add_option("--db", eval_db, "Index database path");
    eval->add_option("--model-dir", eval_model, "Embedding model directory");
    eval->add_option("--mode", eval_mode, "dense, bm25, hybrid or all")->check(CLI::IsMember({"hybrid", "dense", "bm25", "all"}));
    eval->add_option("-k,--top-k", eval_k, "Cutoff for recall@k");
    eval->add_option("--threads", eval_threads, "ONNX Runtime threads (0=auto)");
    eval->add_flag("--json", eval_json, "Print JSON");
    eval->add_flag("--details", eval_details, "Show per-query ranks");

    CLI11_PARSE(app, argc, argv);

    try {
        if (*index) {
            CommonOptions options;
            options.root = index_path;
            options.db = index_db;
            return run_index(options, index_model, index_threads, index_batch,
                             window_lines, overlap_lines, max_tokens, excludes,
                             full, quiet);
        }
        if (*search) {
            CommonOptions options;
            options.root = search_root;
            options.db = search_db;
            return run_search(options, search_model, query, mode, k, candidate_k,
                              dense_weight, bm25_weight, no_auto_weights, repo,
                              lang, path_prefix, after, before, search_threads,
                              search_json);
        }
        if (*stats) {
            CommonOptions options;
            options.root = stats_root;
            options.db = stats_db;
            return run_stats(options, stats_model, stats_threads, stats_json);
        }
        if (*eval) {
            CommonOptions options;
            options.root = eval_root;
            options.db = eval_db;
            return run_eval(options, eval_model, eval_queries, eval_mode, eval_k,
                            eval_threads, eval_json, eval_details);
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
    app.exit(CLI::CallForHelp());
    return 0;
}
