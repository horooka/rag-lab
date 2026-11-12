#include "rag/indexer.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include "rag/util.hpp"

namespace rag {

std::string make_embed_text(const std::string& rel_path,
                            const RawChunk& chunk) {
    std::string text;
    text.reserve(rel_path.size() + chunk.symbol.size() + chunk.text.size() + 32);
    text += rel_path;
    text += '\n';
    if (!chunk.symbol.empty()) {
        text += chunk.symbol;
        text += " (";
        text += chunk.kind;
        text += ")\n";
    }
    text += chunk.text;
    return text;
}

Indexer::Indexer(const Config& config, Source& source, Chunker& chunker,
                 Embedder& embedder, Store& store)
    : config_(config),
      source_(source),
      chunker_(chunker),
      embedder_(embedder),
      store_(store) {}

IndexResult Indexer::run(bool full, bool quiet) {
    const auto started = std::chrono::steady_clock::now();
    IndexResult result;

    // Scan before resetting: a bad target path must not destroy a good index.
    std::vector<SourceFile> files = source_.scan();

    if (full) {
        store_.reset();
    }
    store_.initialize(embedder_.dim());
    result.scanned = static_cast<int64_t>(files.size());
    std::unordered_set<std::string> seen;
    seen.reserve(files.size() * 2);

    int64_t position = 0;
    for (const auto& file : files) {
        ++position;
        seen.insert(file.rel_path);

        try {
            auto state = store_.file_state(file.rel_path);
            if (state && state->mtime_ns == file.mtime_ns &&
                state->size == file.size) {
                ++result.unchanged;
                continue;
            }

            const std::string content = source_.read(file);
            const std::string hash = hex_u64(fnv1a64(content));
            if (state && state->hash == hash) {
                store_.touch_file(state->id, file.mtime_ns, file.size);
                ++result.unchanged;
                continue;
            }

            auto raw_chunks = chunker_.chunk(file, content);
            std::vector<std::string> embed_inputs;
            embed_inputs.reserve(raw_chunks.size());
            for (const auto& chunk : raw_chunks) {
                embed_inputs.push_back(make_embed_text(file.rel_path, chunk));
            }

            std::vector<std::vector<float>> embeddings;
            if (!embed_inputs.empty()) {
                embeddings = embedder_.embed_documents(embed_inputs);
            }
            if (embeddings.size() != raw_chunks.size()) {
                throw std::runtime_error("embedding count mismatch");
            }

            store_.begin();
            try {
                if (state) {
                    store_.delete_file_chunks(state->id);
                }
                Lang effective_lang = file.lang;
                for (const auto& raw : raw_chunks) {
                    if (raw.lang != Lang::Unknown) {
                        effective_lang = raw.lang;
                        break;
                    }
                }
                FileRecord record;
                record.path = file.rel_path;
                record.repo = file.repo;
                record.lang = lang_name(effective_lang);
                record.mtime_ns = file.mtime_ns;
                record.size = file.size;
                record.hash = hash;
                record.chunk_count = static_cast<int>(raw_chunks.size());
                const int64_t file_id = store_.upsert_file(record);

                for (size_t i = 0; i < raw_chunks.size(); ++i) {
                    const auto& raw = raw_chunks[i];
                    Chunk chunk;
                    chunk.file_id = file_id;
                    chunk.path = file.rel_path;
                    chunk.repo = file.repo;
                    chunk.lang = lang_name(raw.lang == Lang::Unknown
                                               ? effective_lang
                                               : raw.lang);
                    chunk.symbol = raw.symbol;
                    chunk.kind = raw.kind;
                    chunk.start_line = raw.start_line;
                    chunk.end_line = raw.end_line;
                    chunk.text = raw.text;
                    store_.insert_chunk(file_id, static_cast<int>(i), chunk,
                                        embeddings[i]);
                }
                store_.commit();
            } catch (...) {
                store_.rollback();
                throw;
            }

            if (state) {
                ++result.updated;
            } else {
                ++result.added;
            }
            result.chunks_written += static_cast<int64_t>(raw_chunks.size());

            if (!quiet) {
                std::cerr << "[" << position << "/" << files.size() << "] "
                          << file.rel_path << " (" << raw_chunks.size()
                          << " chunks)\n";
            }
        } catch (const std::exception& error) {
            ++result.failed;
            std::cerr << "warning: " << file.rel_path << ": " << error.what()
                      << "\n";
        }
    }

    store_.begin();
    try {
        result.removed = store_.delete_missing(seen);
        store_.commit();
    } catch (...) {
        store_.rollback();
        throw;
    }

    store_.set_meta("model", embedder_.spec().name);
    store_.set_meta("dim", std::to_string(embedder_.dim()));
    store_.set_meta("root", config_.root.string());
    store_.set_meta("indexed_at", std::to_string(now_unix()));
    store_.set_meta("mtime_epoch", "unix");
    store_.set_meta("chunk_max_tokens",
                    std::to_string(config_.chunk.max_tokens));
    store_.set_meta("chunk_window_lines",
                    std::to_string(config_.chunk.window_lines));
    store_.set_meta("chunk_overlap_lines",
                    std::to_string(config_.chunk.overlap_lines));
    store_.optimize();

    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    return result;
}

std::string Indexer::embed_text(const SourceFile& file,
                                const RawChunk& chunk) const {
    return make_embed_text(file.rel_path, chunk);
}

}  // namespace rag
