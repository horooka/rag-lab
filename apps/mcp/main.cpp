#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "rag/config.hpp"
#include "rag/embedder.hpp"
#include "rag/resolve.hpp"
#include "rag/search.hpp"
#include "rag/store.hpp"
#include "rag/util.hpp"
#include "rag/workspace.hpp"

namespace {

using namespace rag;

constexpr const char* kVersion = "0.1.0";
constexpr const char* kProtocolVersion = "2026-07-28";

struct ReadChunk {
    std::string source;
    Chunk chunk;
};

class Backend {
public:
    Backend(const std::string& workspace_path, const std::string& db_path,
            const std::string& model_dir, const std::string& root,
            int threads) {
        if (!workspace_path.empty() && !db_path.empty()) {
            throw std::runtime_error(
                "use either --workspace or --db, not both");
        }
        if (workspace_path.empty() && db_path.empty()) {
            throw std::runtime_error("--workspace or --db is required");
        }

        std::filesystem::path config_root =
            root.empty() ? std::filesystem::current_path()
                         : std::filesystem::absolute(root);
        config_ = load_config(config_root);
        if (!db_path.empty()) {
            config_.db_path = std::filesystem::absolute(db_path);
        }
        if (!model_dir.empty()) {
            config_.model_dir = std::filesystem::absolute(model_dir);
        }

        if (!workspace_path.empty()) {
            workspace_ = Workspace::load(workspace_path);
            const std::string index_root =
                workspace_->entries().front().stats.root;
            if (!index_root.empty()) {
                config_.root = index_root;
            }
            if (model_dir.empty() && !workspace_->model_dir().empty()) {
                config_.model_dir = workspace_->model_dir();
            } else if (model_dir.empty()) {
                config_.model_dir = resolve_model_dir(config_.root);
            }
            embedder_ = make_onnx_embedder(config_.model_dir,
                                           config_.batch_size, threads);
            workspace_->validate_embedder(*embedder_);
            return;
        }

        if (!std::filesystem::exists(config_.db_path)) {
            throw std::runtime_error("index not found: " +
                                     config_.db_path.string());
        }
        embedder_ = make_onnx_embedder(config_.model_dir, config_.batch_size,
                                       threads);
        store_ = open_sqlite_store(config_.db_path);
        store_->initialize(embedder_->dim());
        const std::string model = store_->meta("model");
        if (!model.empty() && model != embedder_->spec().name) {
            throw std::runtime_error(
                "index model mismatch: index uses '" + model +
                "', current model is '" + embedder_->spec().name + "'");
        }
        single_name_ = config_.db_path.stem().string();
        if (single_name_.empty()) {
            single_name_ = "index";
        }
    }

    bool is_workspace() const { return workspace_.has_value(); }

    SearchResult search(const std::string& query,
                        const SearchOptions& options) {
        if (workspace_) {
            return search_workspace(*workspace_, *embedder_, options, query);
        }
        Searcher searcher(*store_, *embedder_, options);
        return searcher.search(query);
    }

    ResolveResult resolve(const std::string& symbol,
                          const SearchOptions& options) {
        return classify_symbol_hits(search(symbol, options).hits, symbol);
    }

    std::vector<ReadChunk> read_chunks(const std::string& source,
                                       const std::string& path, int start_line,
                                       int end_line, int limit) {
        std::vector<ReadChunk> result;
        if (workspace_) {
            bool found_source = source.empty();
            for (auto& entry : workspace_->entries()) {
                if (!source.empty() && entry.name != source) {
                    continue;
                }
                found_source = true;
                auto chunks = entry.store->chunks_in_range(
                    path, start_line, end_line, limit);
                for (auto& chunk : chunks) {
                    result.push_back({entry.name, std::move(chunk)});
                }
            }
            if (!found_source) {
                throw std::runtime_error("workspace has no index named: " +
                                         source);
            }
        } else {
            if (!source.empty() && source != single_name_) {
                throw std::runtime_error("unknown index: " + source);
            }
            auto chunks =
                store_->chunks_in_range(path, start_line, end_line, limit);
            for (auto& chunk : chunks) {
                result.push_back({single_name_, std::move(chunk)});
            }
        }
        std::sort(result.begin(), result.end(),
                  [](const ReadChunk& a, const ReadChunk& b) {
                      if (a.source != b.source) return a.source < b.source;
                      if (a.chunk.start_line != b.chunk.start_line) {
                          return a.chunk.start_line < b.chunk.start_line;
                      }
                      return a.chunk.id < b.chunk.id;
                  });
        if (result.size() > static_cast<size_t>(limit)) {
            result.resize(static_cast<size_t>(limit));
        }
        return result;
    }

    nlohmann::json stats_json() const {
        if (workspace_) {
            nlohmann::json out;
            out["workspace"] = true;
            out["indexes"] = nlohmann::json::array();
            for (const auto& entry : workspace_->entries()) {
                out["indexes"].push_back(
                    index_stats_json(entry.name, entry.db_path.string(),
                                     entry.stats));
            }
            const IndexStats totals = workspace_->aggregate_stats();
            out["totals"] = {
                {"files", totals.files},
                {"chunks", totals.chunks},
                {"db_bytes", totals.db_bytes},
            };
            return out;
        }
        nlohmann::json out = index_stats_json(single_name_,
                                              config_.db_path.string(),
                                              store_->stats());
        return out;
    }

    nlohmann::json sources_json() const {
        nlohmann::json sources = nlohmann::json::array();
        if (workspace_) {
            for (const auto& entry : workspace_->entries()) {
                nlohmann::json item;
                item["name"] = entry.name;
                item["db"] = entry.db_path.string();
                item["root"] = entry.stats.root;
                item["model"] = entry.stats.model;
                item["files"] = entry.stats.files;
                item["chunks"] = entry.stats.chunks;
                sources.push_back(std::move(item));
            }
        } else {
            const IndexStats stats = store_->stats();
            nlohmann::json item;
            item["name"] = single_name_;
            item["db"] = config_.db_path.string();
            item["root"] = stats.root;
            item["model"] = stats.model;
            item["files"] = stats.files;
            item["chunks"] = stats.chunks;
            sources.push_back(std::move(item));
        }
        return sources;
    }

private:
    static nlohmann::json index_stats_json(const std::string& name,
                                           const std::string& db,
                                           const IndexStats& stats) {
        nlohmann::json item;
        item["name"] = name;
        item["db"] = db;
        item["root"] = stats.root;
        item["model"] = stats.model;
        item["indexed_at"] = stats.indexed_at;
        item["files"] = stats.files;
        item["chunks"] = stats.chunks;
        item["db_bytes"] = stats.db_bytes;
        return item;
    }

    Config config_;
    std::unique_ptr<Embedder> embedder_;
    std::optional<Workspace> workspace_;
    std::unique_ptr<Store> store_;
    std::string single_name_;
};

std::string string_arg(const nlohmann::json& args, const char* name,
                       bool required = false) {
    if (!args.contains(name) || args[name].is_null()) {
        if (required) {
            throw std::runtime_error(std::string("missing required argument: ") +
                                     name);
        }
        return {};
    }
    if (!args[name].is_string()) {
        throw std::runtime_error(std::string("argument '") + name +
                                 "' must be a string");
    }
    return args[name].get<std::string>();
}

int int_arg(const nlohmann::json& args, const char* name, int fallback,
            int minimum, int maximum) {
    if (!args.contains(name) || args[name].is_null()) {
        return fallback;
    }
    if (!args[name].is_number_integer()) {
        throw std::runtime_error(std::string("argument '") + name +
                                 "' must be an integer");
    }
    const int value = args[name].get<int>();
    if (value < minimum || value > maximum) {
        throw std::runtime_error(std::string("argument '") + name +
                                 "' is out of range");
    }
    return value;
}

SearchOptions search_options_from_args(const nlohmann::json& args,
                                       bool workspace) {
    SearchOptions options;
    options.mode = string_arg(args, "mode");
    if (options.mode.empty()) {
        options.mode = "hybrid";
    }
    if (options.mode != "hybrid" && options.mode != "dense" &&
        options.mode != "bm25") {
        throw std::runtime_error("mode must be hybrid, dense or bm25");
    }
    options.k = int_arg(args, "top_k", 10, 1, 100);
    options.candidate_k = std::max(50, options.k * 5);
    options.filters.repo = string_arg(args, "repo");
    options.filters.lang = string_arg(args, "lang");
    options.filters.path_prefix = string_arg(args, "path");
    options.filters.source = string_arg(args, "source");
    if (!workspace && !options.filters.source.empty()) {
        throw std::runtime_error("source is only valid for workspaces");
    }
    const std::string after = string_arg(args, "after");
    const std::string before = string_arg(args, "before");
    if (!after.empty()) {
        options.filters.after_ns = parse_iso8601_utc(after);
    }
    if (!before.empty()) {
        options.filters.before_ns = parse_iso8601_utc(before);
    }
    return options;
}

nlohmann::json hit_json(const SearchHit& hit) {
    nlohmann::json item;
    if (!hit.source.empty()) {
        item["source"] = hit.source;
    }
    item["chunk_id"] = hit.chunk.id;
    item["path"] = hit.chunk.path;
    item["repo"] = hit.chunk.repo;
    item["lang"] = hit.chunk.lang;
    item["symbol"] = hit.chunk.symbol;
    item["kind"] = hit.chunk.kind;
    item["start_line"] = hit.chunk.start_line;
    item["end_line"] = hit.chunk.end_line;
    item["score"] = hit.score;
    item["dense_rank"] = hit.dense_rank;
    item["bm25_rank"] = hit.bm25_rank;
    item["text"] = hit.chunk.text;
    return item;
}

nlohmann::json chunk_json(const ReadChunk& item) {
    nlohmann::json chunk;
    chunk["source"] = item.source;
    chunk["chunk_id"] = item.chunk.id;
    chunk["path"] = item.chunk.path;
    chunk["repo"] = item.chunk.repo;
    chunk["lang"] = item.chunk.lang;
    chunk["symbol"] = item.chunk.symbol;
    chunk["kind"] = item.chunk.kind;
    chunk["start_line"] = item.chunk.start_line;
    chunk["end_line"] = item.chunk.end_line;
    chunk["text"] = item.chunk.text;
    return chunk;
}

nlohmann::json tool_success(const nlohmann::json& payload) {
    nlohmann::json result;
    result["content"] = nlohmann::json::array(
        {{{"type", "text"}, {"text", payload.dump(2)}}});
    result["isError"] = false;
    return result;
}

nlohmann::json tool_error(const std::string& message) {
    nlohmann::json result;
    result["content"] = nlohmann::json::array(
        {{{"type", "text"}, {"text", "error: " + message}}});
    result["isError"] = true;
    return result;
}

nlohmann::json schema(nlohmann::json properties,
                      std::vector<std::string> required = {}) {
    nlohmann::json result;
    result["type"] = "object";
    result["properties"] = std::move(properties);
    result["additionalProperties"] = false;
    if (!required.empty()) {
        result["required"] = std::move(required);
    }
    return result;
}

nlohmann::json tool_definitions() {
    nlohmann::json search_properties;
    search_properties["query"] = {
        {"type", "string"},
        {"description", "Natural-language or identifier query"}};
    search_properties["top_k"] = {
        {"type", "integer"},
        {"minimum", 1},
        {"maximum", 100},
        {"default", 10}};
    search_properties["mode"] = {
        {"type", "string"},
        {"enum", {"hybrid", "dense", "bm25"}},
        {"default", "hybrid"}};
    search_properties["source"] = {
        {"type", "string"},
        {"description", "Workspace index name"}};
    search_properties["lang"] = {{"type", "string"}};
    search_properties["repo"] = {{"type", "string"}};
    search_properties["path"] = {{"type", "string"}};
    search_properties["after"] = {
        {"type", "string"},
        {"description", "ISO-8601 UTC lower bound on file mtime"}};
    search_properties["before"] = {
        {"type", "string"},
        {"description", "ISO-8601 UTC upper bound on file mtime"}};

    nlohmann::json read_properties;
    read_properties["path"] = {
        {"type", "string"},
        {"description", "Relative source path from search results"}};
    read_properties["source"] = {{"type", "string"}};
    read_properties["start_line"] = {
        {"type", "integer"}, {"minimum", 0}};
    read_properties["end_line"] = {
        {"type", "integer"}, {"minimum", 0}};
    read_properties["limit"] = {
        {"type", "integer"},
        {"minimum", 1},
        {"maximum", 100},
        {"default", 20}};

    nlohmann::json resolve_properties = search_properties;
    resolve_properties.erase("query");
    resolve_properties["symbol"] = {
        {"type", "string"},
        {"description", "Symbol name, e.g. FillStationsList"}};

    nlohmann::json tools = nlohmann::json::array();
    tools.push_back(
        {{"name", "search_codebase"},
         {"description",
          "Search indexed source code across one index or a workspace. "
          "Returns path, line range, symbol and text."},
         {"inputSchema", schema(search_properties, {"query"})}});
    tools.push_back(
        {{"name", "read_chunk"},
         {"description",
          "Read chunks overlapping a source path and optional line range."},
         {"inputSchema", schema(read_properties, {"path"})}});
    tools.push_back(
        {{"name", "resolve_symbol"},
         {"description",
          "Resolve a symbol into exact definitions, text references and "
          "cross-language candidates."},
         {"inputSchema", schema(resolve_properties, {"symbol"})}});
    tools.push_back(
        {{"name", "index_stats"},
         {"description",
          "Return file/chunk/model statistics for the index or workspace."},
         {"inputSchema", schema(nlohmann::json::object())}});
    tools.push_back(
        {{"name", "list_sources"},
         {"description",
          "List the indexes available to this server and their metadata."},
         {"inputSchema", schema(nlohmann::json::object())}});
    return tools;
}

nlohmann::json error_response(const nlohmann::json& id, int code,
                              const std::string& message) {
    nlohmann::json response;
    response["jsonrpc"] = "2.0";
    response["id"] = id;
    response["error"] = {{"code", code}, {"message", message}};
    return response;
}

nlohmann::json result_response(const nlohmann::json& id,
                               const nlohmann::json& result) {
    nlohmann::json response;
    response["jsonrpc"] = "2.0";
    response["id"] = id;
    response["result"] = result;
    return response;
}

nlohmann::json handle_tool_call(Backend& backend,
                                const nlohmann::json& params) {
    const std::string name = params.value("name", "");
    const nlohmann::json args =
        params.contains("arguments") && params["arguments"].is_object()
            ? params["arguments"]
            : nlohmann::json::object();

    try {
        if (name == "search_codebase") {
            const std::string query = string_arg(args, "query", true);
            const SearchOptions options =
                search_options_from_args(args, backend.is_workspace());
            const SearchResult result = backend.search(query, options);
            nlohmann::json payload;
            payload["query"] = query;
            payload["mode"] = options.mode;
            payload["embed_ms"] = result.embed_ms;
            payload["total_ms"] = result.total_ms;
            payload["results"] = nlohmann::json::array();
            for (const auto& hit : result.hits) {
                payload["results"].push_back(hit_json(hit));
            }
            return tool_success(payload);
        }

        if (name == "read_chunk") {
            const std::string path = string_arg(args, "path", true);
            const std::string source = string_arg(args, "source");
            const int start = int_arg(args, "start_line", 0, 0, 100000000);
            const int end = int_arg(args, "end_line", 0, 0, 100000000);
            const int limit = int_arg(args, "limit", 20, 1, 100);
            const auto chunks =
                backend.read_chunks(source, path, start, end, limit);
            nlohmann::json payload;
            payload["chunks"] = nlohmann::json::array();
            for (const auto& chunk : chunks) {
                payload["chunks"].push_back(chunk_json(chunk));
            }
            return tool_success(payload);
        }

        if (name == "resolve_symbol") {
            const std::string symbol = string_arg(args, "symbol", true);
            const SearchOptions options =
                search_options_from_args(args, backend.is_workspace());
            const ResolveResult result = backend.resolve(symbol, options);
            nlohmann::json payload;
            payload["query"] = symbol;
            payload["definitions"] = nlohmann::json::array();
            payload["references"] = nlohmann::json::array();
            payload["candidates"] = nlohmann::json::array();
            for (const auto& hit : result.definitions) {
                payload["definitions"].push_back(hit_json(hit));
            }
            for (const auto& hit : result.references) {
                payload["references"].push_back(hit_json(hit));
            }
            for (const auto& hit : result.candidates) {
                payload["candidates"].push_back(hit_json(hit));
            }
            return tool_success(payload);
        }

        if (name == "index_stats") {
            return tool_success(backend.stats_json());
        }

        if (name == "list_sources") {
            return tool_success(backend.sources_json());
        }

        throw std::runtime_error("unknown tool: " + name);
    } catch (const std::exception& error) {
        return tool_error(error.what());
    }
}

nlohmann::json handle_request(Backend& backend, const nlohmann::json& request) {
    if (!request.is_object()) {
        return error_response(nullptr, -32600, "invalid request");
    }
    const bool has_id = request.contains("id");
    const nlohmann::json id =
        has_id ? request["id"] : nlohmann::json(nullptr);
    const std::string method = request.value("method", "");
    const nlohmann::json params =
        request.contains("params") && request["params"].is_object()
            ? request["params"]
            : nlohmann::json::object();

    if (method == "initialize") {
        const std::string requested = params.value("protocolVersion", "");
        std::string version = kProtocolVersion;
        if (requested == "2024-11-05" || requested == "2025-03-26" ||
            requested == "2025-06-18" || requested == "2025-11-25" ||
            requested == "2026-07-28") {
            version = requested;
        }
        nlohmann::json result;
        result["protocolVersion"] = version;
        result["capabilities"] = {{"tools", nlohmann::json::object()}};
        result["serverInfo"] = {
            {"name", "rag-lab"},
            {"version", kVersion},
        };
        result["instructions"] =
            "Search and resolve symbols across indexed codebases. "
            "Use search_codebase for discovery, resolve_symbol for "
            "cross-index definitions/candidates, and read_chunk for context.";
        return has_id ? result_response(id, result) : nlohmann::json();
    }

    if (method == "notifications/initialized" ||
        method == "notifications/cancelled" || method == "initialized") {
        return nlohmann::json();
    }

    if (method == "tools/list") {
        return has_id ? result_response(id, {{"tools", tool_definitions()}})
                      : nlohmann::json();
    }

    if (method == "tools/call") {
        if (!has_id) {
            return nlohmann::json();
        }
        return result_response(id, handle_tool_call(backend, params));
    }

    if (method == "ping") {
        return has_id ? result_response(id, nlohmann::json::object())
                      : nlohmann::json();
    }

    if (method == "resources/list") {
        return has_id ? result_response(id, {{"resources", nlohmann::json::array()}})
                      : nlohmann::json();
    }

    if (method == "prompts/list") {
        return has_id ? result_response(id, {{"prompts", nlohmann::json::array()}})
                      : nlohmann::json();
    }

    if (method == "logging/setLevel") {
        return has_id ? result_response(id, nlohmann::json::object())
                      : nlohmann::json();
    }

    if (method == "shutdown") {
        return has_id ? result_response(id, nlohmann::json::object())
                      : nlohmann::json();
    }

    if (!has_id) {
        return nlohmann::json();
    }
    return error_response(id, -32601, "method not found: " + method);
}

void send_message(const nlohmann::json& message) {
    if (message.is_null()) {
        return;
    }
    std::cout << message.dump() << '\n' << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"rag-lab MCP stdio server", "rag-mcp"};
    app.set_version_flag("--version", kVersion);

    std::string workspace_path;
    std::string db_path;
    std::string model_dir;
    std::string root = ".";
    int threads = 0;
    bool verbose = false;
    app.add_option("--workspace", workspace_path,
                   "Workspace JSON with multiple named indexes");
    app.add_option("--db", db_path, "Single index database path");
    app.add_option("--model-dir", model_dir, "Embedding model directory");
    app.add_option("--root", root, "Root used to find .rag/config.json");
    app.add_option("--threads", threads, "ONNX Runtime threads (0=auto)");
    app.add_flag("-v,--verbose", verbose, "Log startup information to stderr");

    CLI11_PARSE(app, argc, argv);

    try {
        Backend backend(workspace_path, db_path, model_dir, root, threads);
        if (verbose) {
            std::cerr << "rag-mcp ready (" << (backend.is_workspace()
                                                   ? "workspace"
                                                   : "single index")
                      << ")\n";
        }
        std::string line;
        while (std::getline(std::cin, line)) {
            if (trim(line).empty()) {
                continue;
            }
            nlohmann::json request =
                nlohmann::json::parse(line, nullptr, false);
            if (request.is_discarded()) {
                send_message(error_response(nullptr, -32700,
                                            "parse error"));
                continue;
            }
            if (request.is_array()) {
                for (const auto& item : request) {
                    send_message(handle_request(backend, item));
                }
            } else {
                send_message(handle_request(backend, request));
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rag-mcp: " << error.what() << "\n";
        return 1;
    }
}
