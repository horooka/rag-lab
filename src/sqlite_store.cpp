#include "rag/store.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include <sqlite3.h>

#include "sqlite-vec.h"
#include "rag/util.hpp"

namespace rag {

namespace {

void check(int rc, sqlite3 *db, const std::string &action) {
    if (rc != SQLITE_OK && rc != SQLITE_DONE && rc != SQLITE_ROW) {
        throw std::runtime_error(action + ": " +
                                 (db ? sqlite3_errmsg(db) : "sqlite error"));
    }
}

class Statement {
    public:
        Statement(sqlite3 *db, const std::string &sql) : db_(db) {
            int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt_, nullptr);
            if (rc != SQLITE_OK) {
                throw std::runtime_error(
                    "prepare failed: " + std::string(sqlite3_errmsg(db_)) +
                    "\nSQL: " + sql);
            }
        }

        ~Statement() {
            if (stmt_) {
                sqlite3_finalize(stmt_);
            }
        }

        Statement(const Statement &) = delete;
        Statement &operator=(const Statement &) = delete;

        void bind(int index, int64_t value) {
            check(sqlite3_bind_int64(stmt_, index, value), db_, "bind int");
        }
        void bind(int index, double value) {
            check(sqlite3_bind_double(stmt_, index, value), db_, "bind double");
        }
        void bind(int index, const std::string &value) {
            check(sqlite3_bind_text(stmt_, index, value.c_str(),
                                    static_cast<int>(value.size()),
                                    SQLITE_TRANSIENT),
                  db_, "bind text");
        }
        void bind_null(int index) {
            check(sqlite3_bind_null(stmt_, index), db_, "bind null");
        }
        void bind_blob(int index, const void *data, size_t size) {
            check(sqlite3_bind_blob(stmt_, index, data, static_cast<int>(size),
                                    SQLITE_TRANSIENT),
                  db_, "bind blob");
        }

        bool step() {
            int rc = sqlite3_step(stmt_);
            if (rc == SQLITE_ROW) {
                return true;
            }
            if (rc == SQLITE_DONE) {
                return false;
            }
            throw std::runtime_error("step failed: " +
                                     std::string(sqlite3_errmsg(db_)));
        }

        void reset() {
            check(sqlite3_reset(stmt_), db_, "reset");
            check(sqlite3_clear_bindings(stmt_), db_, "clear bindings");
        }

        int64_t int64(int column) const {
            return sqlite3_column_int64(stmt_, column);
        }
        double as_double(int column) const {
            return sqlite3_column_double(stmt_, column);
        }
        bool is_null(int column) const {
            return sqlite3_column_type(stmt_, column) == SQLITE_NULL;
        }
        const void *blob(int column) const {
            return sqlite3_column_blob(stmt_, column);
        }
        int blob_bytes(int column) const {
            return sqlite3_column_bytes(stmt_, column);
        }
        std::string text(int column) const {
            const unsigned char *value = sqlite3_column_text(stmt_, column);
            if (!value) {
                return {};
            }
            return std::string(
                reinterpret_cast<const char *>(value),
                static_cast<size_t>(sqlite3_column_bytes(stmt_, column)));
        }

    private:
        sqlite3 *db_;
        sqlite3_stmt *stmt_ = nullptr;
};

std::string fts_terms(const std::string &query) {
    std::vector<std::string> terms;
    std::string current;
    auto flush = [&]() {
        if (current.size() >= 2) {
            terms.push_back(current);
            // Also split snake_case and camel-ish compounds; FTS unicode61
            // already treats underscores as separators, but keep both forms.
            auto parts = split(current, '_');
            for (const auto &part : parts) {
                if (part.size() >= 2 && part != current) {
                    terms.push_back(part);
                }
            }
        }
        current.clear();
    };
    for (unsigned char c : query) {
        if (std::isalnum(c) || c == '_' || c >= 0x80) {
            current.push_back(static_cast<char>(c));
        } else {
            flush();
        }
    }
    flush();

    std::sort(terms.begin(), terms.end());
    terms.erase(std::unique(terms.begin(), terms.end()), terms.end());
    if (terms.empty()) {
        return {};
    }
    std::string out;
    for (const auto &term : terms) {
        if (!out.empty()) {
            out += " OR ";
        }
        out += "\"";
        for (char c : term) {
            if (c == '"') {
                out += "\"\"";
            } else {
                out.push_back(c);
            }
        }
        out += "\"";
    }
    return out;
}

struct FilterBindings {
        std::string where = "1=1";
        std::vector<std::string> texts;
        std::vector<int64_t> numbers;
};

std::string escape_like(const std::string &value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char c : value) {
        if (c == '\\' || c == '%' || c == '_') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

FilterBindings build_filters(const SearchFilters &filters) {
    FilterBindings bindings;
    std::vector<std::string> clauses;
    if (!filters.repo.empty()) {
        clauses.emplace_back("f.repo = ?");
        bindings.texts.push_back(filters.repo);
    }
    if (!filters.lang.empty()) {
        clauses.emplace_back("c.lang = ?");
        bindings.texts.push_back(filters.lang);
    }
    if (!filters.path_prefix.empty()) {
        clauses.emplace_back("c.path LIKE ? ESCAPE '\\'");
        bindings.texts.push_back(escape_like(filters.path_prefix) + "%");
    }
    if (filters.after_ns.has_value()) {
        clauses.emplace_back("f.mtime_ns >= ?");
        bindings.numbers.push_back(*filters.after_ns);
    }
    if (filters.before_ns.has_value()) {
        clauses.emplace_back("f.mtime_ns <= ?");
        bindings.numbers.push_back(*filters.before_ns);
    }
    if (!clauses.empty()) {
        bindings.where.clear();
        for (size_t i = 0; i < clauses.size(); ++i) {
            if (i)
                bindings.where += " AND ";
            bindings.where += clauses[i];
        }
    }
    return bindings;
}

void bind_filters(Statement &stmt, int &index, const FilterBindings &bindings) {
    for (const auto &text : bindings.texts) {
        stmt.bind(index++, text);
    }
    for (int64_t number : bindings.numbers) {
        stmt.bind(index++, number);
    }
}

bool has_filters(const SearchFilters &filters) {
    return !filters.repo.empty() || !filters.lang.empty() ||
           !filters.path_prefix.empty() || filters.after_ns.has_value() ||
           filters.before_ns.has_value();
}

double cosine_similarity(const std::vector<float> &a,
                         const std::vector<float> &b) {
    double value = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        value += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return std::max(-1.0, std::min(1.0, value));
}

class SqliteStore final : public Store {
    public:
        explicit SqliteStore(const std::filesystem::path &path) {
            std::error_code ec;
            if (path.has_parent_path()) {
                std::filesystem::create_directories(path.parent_path(), ec);
            }
            register_vec_extension();
            int rc =
                sqlite3_open_v2(path.string().c_str(), &db_,
                                SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                                    SQLITE_OPEN_FULLMUTEX,
                                nullptr);
            if (rc != SQLITE_OK) {
                std::string message = db_ ? sqlite3_errmsg(db_) : "open failed";
                if (db_) {
                    sqlite3_close(db_);
                    db_ = nullptr;
                }
                throw std::runtime_error("cannot open index " + path.string() +
                                         ": " + message);
            }
            path_ = path;
            sqlite3_busy_timeout(db_, 5000);
            exec("PRAGMA journal_mode=WAL;");
            exec("PRAGMA synchronous=NORMAL;");
            exec("PRAGMA foreign_keys=ON;");
        }

        ~SqliteStore() override {
            if (db_) {
                sqlite3_close(db_);
            }
        }

        void initialize(int dim) override {
            dim_ = dim;
            create_schema();
            std::string stored = meta("dim");
            if (stored.empty()) {
                set_meta("dim", std::to_string(dim));
            } else if (std::stoi(stored) != dim) {
                throw std::runtime_error("index was built with dim=" + stored +
                                         ", current model has "
                                         "dim=" +
                                         std::to_string(dim) +
                                         "; rerun indexing with --full");
            }
        }

        void reset() override {
            exec("DROP TABLE IF EXISTS vec_chunks;");
            exec("DROP TABLE IF EXISTS fts_chunks;");
            exec("DROP TABLE IF EXISTS chunks;");
            exec("DROP TABLE IF EXISTS files;");
            exec("DROP TABLE IF EXISTS meta;");
            create_schema();
            if (dim_ > 0) {
                set_meta("dim", std::to_string(dim_));
            }
        }

        void begin() override {
            if (in_transaction_) {
                return;
            }
            exec("BEGIN IMMEDIATE;");
            in_transaction_ = true;
        }

        void commit() override {
            if (!in_transaction_) {
                return;
            }
            exec("COMMIT;");
            in_transaction_ = false;
        }

        void rollback() override {
            if (!in_transaction_) {
                return;
            }
            exec("ROLLBACK;");
            in_transaction_ = false;
        }

        std::optional<FileState> file_state(const std::string &path) override {
            Statement stmt(
                db_,
                "SELECT id, mtime_ns, size, hash FROM files WHERE path=?;");
            stmt.bind(1, path);
            if (!stmt.step()) {
                return std::nullopt;
            }
            FileState state;
            state.id = stmt.int64(0);
            state.mtime_ns = stmt.int64(1);
            state.size = stmt.int64(2);
            state.hash = stmt.text(3);
            return state;
        }

        void touch_file(int64_t file_id, int64_t mtime_ns,
                        int64_t size) override {
            Statement stmt(db_,
                           "UPDATE files SET mtime_ns=?, size=? WHERE id=?;");
            stmt.bind(1, mtime_ns);
            stmt.bind(2, size);
            stmt.bind(3, file_id);
            stmt.step();
        }

        int64_t upsert_file(const FileRecord &file) override {
            {
                Statement stmt(
                    db_,
                    "INSERT INTO files(path, repo, lang, mtime_ns, size, hash, "
                    "chunk_count, indexed_at) VALUES(?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(path) DO UPDATE SET repo=excluded.repo, "
                    "lang=excluded.lang, mtime_ns=excluded.mtime_ns, "
                    "size=excluded.size, hash=excluded.hash, "
                    "chunk_count=excluded.chunk_count, "
                    "indexed_at=excluded.indexed_at;");
                stmt.bind(1, file.path);
                stmt.bind(2, file.repo);
                stmt.bind(3, file.lang);
                stmt.bind(4, file.mtime_ns);
                stmt.bind(5, file.size);
                stmt.bind(6, file.hash);
                stmt.bind(7, static_cast<int64_t>(file.chunk_count));
                stmt.bind(8, now_unix());
                stmt.step();
            }
            Statement stmt(db_, "SELECT id FROM files WHERE path=?;");
            stmt.bind(1, file.path);
            if (!stmt.step()) {
                throw std::runtime_error("failed to read back file id");
            }
            return stmt.int64(0);
        }

        void delete_file_chunks(int64_t file_id) override {
            std::vector<int64_t> ids;
            {
                Statement stmt(db_, "SELECT id FROM chunks WHERE file_id=?;");
                stmt.bind(1, file_id);
                while (stmt.step()) {
                    ids.push_back(stmt.int64(0));
                }
            }
            if (!ids.empty()) {
                Statement del_vec(db_, "DELETE FROM vec_chunks WHERE rowid=?;");
                Statement del_fts(db_, "DELETE FROM fts_chunks WHERE rowid=?;");
                for (int64_t id : ids) {
                    del_vec.bind(1, id);
                    del_vec.step();
                    del_vec.reset();
                    del_fts.bind(1, id);
                    del_fts.step();
                    del_fts.reset();
                }
            }
            Statement stmt(db_, "DELETE FROM chunks WHERE file_id=?;");
            stmt.bind(1, file_id);
            stmt.step();
        }

        void insert_chunk(int64_t file_id, int seq, const Chunk &chunk,
                          const std::vector<float> &embedding) override {
            int64_t chunk_id = 0;
            {
                Statement stmt(db_, "INSERT INTO chunks(file_id, seq, path, "
                                    "repo, lang, symbol, "
                                    "kind, start_line, end_line, text) "
                                    "VALUES(?,?,?,?,?,?,?,?,?,?);");
                stmt.bind(1, file_id);
                stmt.bind(2, static_cast<int64_t>(seq));
                stmt.bind(3, chunk.path);
                stmt.bind(4, chunk.repo);
                stmt.bind(5, chunk.lang);
                stmt.bind(6, chunk.symbol);
                stmt.bind(7, chunk.kind);
                stmt.bind(8, static_cast<int64_t>(chunk.start_line));
                stmt.bind(9, static_cast<int64_t>(chunk.end_line));
                stmt.bind(10, chunk.text);
                stmt.step();
                chunk_id = sqlite3_last_insert_rowid(db_);
            }
            {
                Statement stmt(
                    db_,
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES(?,?);");
                stmt.bind(1, chunk_id);
                stmt.bind_blob(2, embedding.data(),
                               embedding.size() * sizeof(float));
                stmt.step();
            }
            {
                Statement stmt(
                    db_, "INSERT INTO fts_chunks(rowid, text, symbol, path) "
                         "VALUES(?,?,?,?);");
                stmt.bind(1, chunk_id);
                stmt.bind(2, chunk.text);
                stmt.bind(3, chunk.symbol);
                stmt.bind(4, chunk.path);
                stmt.step();
            }
        }

        int64_t
        delete_missing(const std::unordered_set<std::string> &paths) override {
            std::vector<std::pair<int64_t, std::string>> existing;
            {
                Statement stmt(db_, "SELECT id, path FROM files;");
                while (stmt.step()) {
                    existing.emplace_back(stmt.int64(0), stmt.text(1));
                }
            }
            int64_t removed = 0;
            for (const auto &[id, path] : existing) {
                if (paths.count(path)) {
                    continue;
                }
                delete_file_chunks(id);
                Statement stmt(db_, "DELETE FROM files WHERE id=?;");
                stmt.bind(1, id);
                stmt.step();
                ++removed;
            }
            return removed;
        }

        std::vector<RankedId>
        dense_search(const std::vector<float> &query, int k,
                     const SearchFilters &filters = {}) override {
            std::vector<RankedId> result;
            if (k <= 0 || query.empty()) {
                return result;
            }
            if (!has_filters(filters)) {
                Statement stmt(
                    db_,
                    "SELECT rowid, distance FROM vec_chunks "
                    "WHERE embedding MATCH ? AND k = ? ORDER BY distance;");
                stmt.bind_blob(1, query.data(), query.size() * sizeof(float));
                stmt.bind(2, static_cast<int64_t>(k));
                while (stmt.step()) {
                    result.push_back({stmt.int64(0), stmt.as_double(1)});
                }
                return result;
            }

            // Metadata filters (especially recency) can eliminate most of the
            // KNN candidate set before it is returned, so scan the filtered
            // subset and rank it directly. Personal-scale indexes make this
            // cheap.
            const FilterBindings bindings = build_filters(filters);
            Statement stmt(db_, "SELECT c.id, v.embedding FROM chunks c "
                                "JOIN files f ON f.id = c.file_id "
                                "JOIN vec_chunks v ON v.rowid = c.id WHERE " +
                                    bindings.where + ";");
            int index = 1;
            bind_filters(stmt, index, bindings);
            while (stmt.step()) {
                const void *blob = stmt.blob(1);
                const int bytes = stmt.blob_bytes(1);
                if (!blob || bytes <= 0 ||
                    bytes % static_cast<int>(sizeof(float)) != 0 ||
                    static_cast<size_t>(bytes) !=
                        query.size() * sizeof(float)) {
                    continue;
                }
                std::vector<float> embedding(static_cast<size_t>(bytes) /
                                             sizeof(float));
                std::memcpy(embedding.data(), blob, static_cast<size_t>(bytes));
                const double similarity = cosine_similarity(query, embedding);
                result.push_back({stmt.int64(0), 1.0 - similarity});
            }
            std::sort(result.begin(), result.end(),
                      [](const RankedId &a, const RankedId &b) {
                          if (a.score != b.score)
                              return a.score < b.score;
                          return a.id < b.id;
                      });
            if (result.size() > static_cast<size_t>(k)) {
                result.resize(static_cast<size_t>(k));
            }
            return result;
        }

        std::vector<RankedId>
        bm25_search(const std::string &query, int k,
                    const SearchFilters &filters = {}) override {
            std::vector<RankedId> result;
            if (k <= 0) {
                return result;
            }
            const std::string terms = fts_terms(query);
            if (terms.empty()) {
                return result;
            }
            if (!has_filters(filters)) {
                Statement stmt(
                    db_,
                    "SELECT rowid, bm25(fts_chunks, 1.0, 4.0, 2.0) AS score "
                    "FROM fts_chunks WHERE fts_chunks MATCH ? "
                    "ORDER BY score LIMIT ?;");
                stmt.bind(1, terms);
                stmt.bind(2, static_cast<int64_t>(k));
                while (stmt.step()) {
                    result.push_back({stmt.int64(0), -stmt.as_double(1)});
                }
                return result;
            }

            const FilterBindings bindings = build_filters(filters);
            Statement stmt(db_, "SELECT fts_chunks.rowid, "
                                "bm25(fts_chunks, 1.0, 4.0, 2.0) AS score "
                                "FROM fts_chunks "
                                "JOIN chunks c ON c.id = fts_chunks.rowid "
                                "JOIN files f ON f.id = c.file_id "
                                "WHERE fts_chunks MATCH ? AND " +
                                    bindings.where +
                                    " ORDER BY score LIMIT ?;");
            int index = 1;
            stmt.bind(index++, terms);
            bind_filters(stmt, index, bindings);
            stmt.bind(index++, static_cast<int64_t>(k));
            while (stmt.step()) {
                result.push_back({stmt.int64(0), -stmt.as_double(1)});
            }
            return result;
        }

        std::vector<Chunk>
        chunks_by_ids(const std::vector<int64_t> &ids) override {
            std::vector<Chunk> chunks;
            if (ids.empty()) {
                return chunks;
            }
            std::string sql =
                "SELECT c.id, c.file_id, c.path, c.repo, c.lang, c.symbol, "
                "c.kind, c.start_line, c.end_line, f.mtime_ns, c.text "
                "FROM chunks c JOIN files f ON f.id = c.file_id WHERE c.id IN "
                "(";
            for (size_t i = 0; i < ids.size(); ++i) {
                if (i) {
                    sql += ",";
                }
                sql += "?";
            }
            sql += ");";
            Statement stmt(db_, sql);
            for (size_t i = 0; i < ids.size(); ++i) {
                stmt.bind(static_cast<int>(i + 1), ids[i]);
            }
            while (stmt.step()) {
                Chunk chunk;
                chunk.id = stmt.int64(0);
                chunk.file_id = stmt.int64(1);
                chunk.path = stmt.text(2);
                chunk.repo = stmt.text(3);
                chunk.lang = stmt.text(4);
                chunk.symbol = stmt.text(5);
                chunk.kind = stmt.text(6);
                chunk.start_line = static_cast<int>(stmt.int64(7));
                chunk.end_line = static_cast<int>(stmt.int64(8));
                chunk.mtime_ns = stmt.int64(9);
                chunk.text = stmt.text(10);
                chunks.push_back(std::move(chunk));
            }
            return chunks;
        }

        std::vector<Chunk> chunks_in_range(
        const std::string& path, int start_line, int end_line,
        int limit) override {
        std::vector<Chunk> chunks;
        if (limit <= 0) {
            return chunks;
        }
        std::string sql =
            "SELECT c.id, c.file_id, c.path, c.repo, c.lang, c.symbol, "
            "c.kind, c.start_line, c.end_line, f.mtime_ns, c.text "
            "FROM chunks c JOIN files f ON f.id = c.file_id "
            "WHERE c.path = ?";
        if (start_line > 0) {
            sql += " AND c.end_line >= ?";
        }
        if (end_line > 0) {
            sql += " AND c.start_line <= ?";
        }
        sql += " ORDER BY c.start_line LIMIT ?;";
        Statement stmt(db_, sql);
        int index = 1;
        stmt.bind(index++, path);
        if (start_line > 0) {
            stmt.bind(index++, static_cast<int64_t>(start_line));
        }
        if (end_line > 0) {
            stmt.bind(index++, static_cast<int64_t>(end_line));
        }
        stmt.bind(index++, static_cast<int64_t>(limit));
        while (stmt.step()) {
            Chunk chunk;
            chunk.id = stmt.int64(0);
            chunk.file_id = stmt.int64(1);
            chunk.path = stmt.text(2);
            chunk.repo = stmt.text(3);
            chunk.lang = stmt.text(4);
            chunk.symbol = stmt.text(5);
            chunk.kind = stmt.text(6);
            chunk.start_line = static_cast<int>(stmt.int64(7));
            chunk.end_line = static_cast<int>(stmt.int64(8));
            chunk.mtime_ns = stmt.int64(9);
            chunk.text = stmt.text(10);
            chunks.push_back(std::move(chunk));
        }
        return chunks;
    }

    std::vector<ChunkEmbedding> all_chunks() override {
            std::vector<ChunkEmbedding> result;
            Statement stmt(
                db_,
                "SELECT c.id, c.file_id, c.path, c.repo, c.lang, c.symbol, "
                "c.kind, c.start_line, c.end_line, c.text, f.mtime_ns, "
                "v.embedding FROM chunks c "
                "JOIN files f ON f.id = c.file_id "
                "JOIN vec_chunks v ON v.rowid = c.id "
                "ORDER BY c.path, c.start_line, c.id;");
            while (stmt.step()) {
                ChunkEmbedding item;
                item.chunk.id = stmt.int64(0);
                item.chunk.file_id = stmt.int64(1);
                item.chunk.path = stmt.text(2);
                item.chunk.repo = stmt.text(3);
                item.chunk.lang = stmt.text(4);
                item.chunk.symbol = stmt.text(5);
                item.chunk.kind = stmt.text(6);
                item.chunk.start_line = static_cast<int>(stmt.int64(7));
                item.chunk.end_line = static_cast<int>(stmt.int64(8));
                item.chunk.text = stmt.text(9);
                item.chunk.mtime_ns = stmt.int64(10);
                const void *blob = stmt.blob(11);
                const int bytes = stmt.blob_bytes(11);
                if (!blob || bytes <= 0 ||
                    bytes % static_cast<int>(sizeof(float)) != 0) {
                    throw std::runtime_error("invalid embedding blob in index");
                }
                const size_t count = static_cast<size_t>(bytes) / sizeof(float);
                item.embedding.resize(count);
                std::memcpy(item.embedding.data(), blob,
                            static_cast<size_t>(bytes));
                result.push_back(std::move(item));
            }
            return result;
        }

        IndexStats stats() override {
            IndexStats stats;
            {
                Statement stmt(db_, "SELECT COUNT(*) FROM files;");
                if (stmt.step())
                    stats.files = stmt.int64(0);
            }
            {
                Statement stmt(db_, "SELECT COUNT(*) FROM chunks;");
                if (stmt.step())
                    stats.chunks = stmt.int64(0);
            }
            {
                Statement stmt(
                    db_, "SELECT repo, COUNT(*) FROM chunks GROUP BY repo "
                         "ORDER BY COUNT(*) DESC, repo;");
                while (stmt.step()) {
                    stats.repos.emplace_back(stmt.text(0), stmt.int64(1));
                }
            }
            {
                Statement stmt(
                    db_, "SELECT lang, COUNT(*) FROM chunks GROUP BY lang "
                         "ORDER BY COUNT(*) DESC, lang;");
                while (stmt.step()) {
                    stats.languages.emplace_back(stmt.text(0), stmt.int64(1));
                }
            }
            stats.model = meta("model");
            stats.root = meta("root");
            const std::string indexed = meta("indexed_at");
            stats.indexed_at = indexed.empty() ? 0 : std::stoll(indexed);
            std::error_code ec;
            if (std::filesystem::exists(path_, ec)) {
                stats.db_bytes = std::filesystem::file_size(path_, ec);
            }
            return stats;
        }

        std::string meta(const std::string &key,
                         const std::string &fallback = {}) override {
            Statement stmt(db_, "SELECT value FROM meta WHERE key=?;");
            stmt.bind(1, key);
            if (!stmt.step()) {
                return fallback;
            }
            return stmt.text(0);
        }

        void set_meta(const std::string &key,
                      const std::string &value) override {
            Statement stmt(
                db_, "INSERT INTO meta(key, value) VALUES(?,?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value;");
            stmt.bind(1, key);
            stmt.bind(2, value);
            stmt.step();
        }

        void optimize() override {
            try {
                exec("INSERT INTO fts_chunks(fts_chunks) VALUES('optimize');");
            } catch (...) {
                // Non-fatal: an empty FTS table may reject the optimize
                // command.
            }
            exec("PRAGMA optimize;");
        }

    private:
        static void register_vec_extension() {
            static std::once_flag once;
            std::call_once(once, [] {
                sqlite3_auto_extension(
                    reinterpret_cast<void (*)(void)>(sqlite3_vec_init));
            });
        }

        void exec(const std::string &sql) {
            char *error = nullptr;
            int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &error);
            if (rc != SQLITE_OK) {
                std::string message = error ? error : sqlite3_errmsg(db_);
                sqlite3_free(error);
                throw std::runtime_error("sqlite: " + message +
                                         "\nSQL: " + sql);
            }
        }

        void create_schema() {
            exec("CREATE TABLE IF NOT EXISTS meta("
                 "key TEXT PRIMARY KEY, value TEXT NOT NULL);");
            exec("CREATE TABLE IF NOT EXISTS files("
                 "id INTEGER PRIMARY KEY,"
                 "path TEXT NOT NULL UNIQUE,"
                 "repo TEXT NOT NULL,"
                 "lang TEXT NOT NULL,"
                 "mtime_ns INTEGER NOT NULL,"
                 "size INTEGER NOT NULL,"
                 "hash TEXT NOT NULL,"
                 "chunk_count INTEGER NOT NULL DEFAULT 0,"
                 "indexed_at INTEGER NOT NULL);");
            exec("CREATE TABLE IF NOT EXISTS chunks("
                 "id INTEGER PRIMARY KEY,"
                 "file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE "
                 "CASCADE,"
                 "seq INTEGER NOT NULL,"
                 "path TEXT NOT NULL,"
                 "repo TEXT NOT NULL,"
                 "lang TEXT NOT NULL,"
                 "symbol TEXT NOT NULL DEFAULT '',"
                 "kind TEXT NOT NULL DEFAULT '',"
                 "start_line INTEGER NOT NULL,"
                 "end_line INTEGER NOT NULL,"
                 "text TEXT NOT NULL,"
                 "UNIQUE(file_id, seq));");
            exec("CREATE INDEX IF NOT EXISTS idx_chunks_file ON "
                 "chunks(file_id);");
            exec("CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo);");
            exec("CREATE INDEX IF NOT EXISTS idx_chunks_lang ON chunks(lang);");
            if (dim_ <= 0) {
                return;
            }
            exec("CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
                 "embedding float[" +
                 std::to_string(dim_) + "]);");
            exec("CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5("
                 "text, symbol, path, tokenize='unicode61 remove_diacritics "
                 "2');");
        }

        sqlite3 *db_ = nullptr;
        std::filesystem::path path_;
        int dim_ = 0;
        bool in_transaction_ = false;
};

} // namespace

std::unique_ptr<Store> open_sqlite_store(const std::filesystem::path &db_path) {
    return std::make_unique<SqliteStore>(db_path);
}

} // namespace rag
