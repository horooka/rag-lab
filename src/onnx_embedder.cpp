#include "rag/embedder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>

#include "rag/config.hpp"
#include "rag/tokenizer.hpp"
#include "rag/util.hpp"

namespace rag {

EmbeddingModelSpec load_model_spec(const std::filesystem::path& model_dir) {
    EmbeddingModelSpec spec;
    std::filesystem::path manifest = model_dir / "model.json";
    std::error_code ec;
    if (std::filesystem::exists(manifest, ec)) {
        nlohmann::json json = nlohmann::json::parse(read_file(manifest));
        if (json.contains("name")) spec.name = json["name"].get<std::string>();
        if (json.contains("onnx_file"))
            spec.onnx_file = json["onnx_file"].get<std::string>();
        if (json.contains("vocab_file"))
            spec.vocab_file = json["vocab_file"].get<std::string>();
        if (json.contains("dim")) spec.dim = json["dim"].get<int>();
        if (json.contains("max_length"))
            spec.max_length = json["max_length"].get<int>();
        if (json.contains("pooling"))
            spec.pooling = json["pooling"].get<std::string>();
        if (json.contains("normalize"))
            spec.normalize = json["normalize"].get<bool>();
        if (json.contains("query_prefix"))
            spec.query_prefix = json["query_prefix"].get<std::string>();
        if (json.contains("doc_prefix"))
            spec.doc_prefix = json["doc_prefix"].get<std::string>();
    }
    if (!std::filesystem::exists(model_dir / spec.onnx_file, ec)) {
        throw std::runtime_error("embedding model not found: " +
                                 (model_dir / spec.onnx_file).string() +
                                 "\nRun scripts/fetch-deps.sh or pass --model-dir.");
    }
    if (!std::filesystem::exists(model_dir / spec.vocab_file, ec)) {
        throw std::runtime_error("tokenizer vocab not found: " +
                                 (model_dir / spec.vocab_file).string());
    }
    return spec;
}

namespace {

class OnnxEmbedder final : public Embedder {
public:
    OnnxEmbedder(const std::filesystem::path& model_dir, int batch_size,
                 int threads)
        : spec_(load_model_spec(model_dir)),
          tokenizer_(model_dir / spec_.vocab_file),
          env_(ORT_LOGGING_LEVEL_WARNING, "rag-lab"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                  OrtMemTypeDefault)) {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetInterOpNumThreads(1);
        if (threads > 0) {
            options.SetIntraOpNumThreads(threads);
        }
        const std::string model_path = (model_dir / spec_.onnx_file).string();
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(),
                                                  options);
        batch_size_ = std::max(1, batch_size);

        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.emplace_back(name.get());
        }
        for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.emplace_back(name.get());
        }
        for (const auto& name : input_names_) {
            input_cstr_.push_back(name.c_str());
        }
        for (const auto& name : output_names_) {
            output_cstr_.push_back(name.c_str());
        }
        if (output_names_.empty()) {
            throw std::runtime_error("embedding model has no outputs");
        }
        if (spec_.pooling != "cls" && spec_.pooling != "mean") {
            throw std::runtime_error("unsupported pooling: " + spec_.pooling);
        }
    }

    int dim() const override {
        return dim_ > 0 ? dim_ : spec_.dim;
    }

    const EmbeddingModelSpec& spec() const override { return spec_; }

    std::vector<float> embed_query(std::string_view text) override {
        std::string input = spec_.query_prefix + std::string(text);
        auto result = embed_batch({input});
        return std::move(result.at(0));
    }

    std::vector<std::vector<float>> embed_documents(
        const std::vector<std::string>& texts) override {
        std::vector<std::vector<float>> out;
        out.reserve(texts.size());
        for (size_t start = 0; start < texts.size();
             start += static_cast<size_t>(batch_size_)) {
            const size_t end =
                std::min(texts.size(), start + static_cast<size_t>(batch_size_));
            std::vector<std::string> batch;
            batch.reserve(end - start);
            for (size_t i = start; i < end; ++i) {
                batch.push_back(spec_.doc_prefix + texts[i]);
            }
            auto embedded = embed_batch(batch);
            for (auto& item : embedded) {
                out.push_back(std::move(item));
            }
        }
        return out;
    }

private:
    std::vector<std::vector<float>> embed_batch(
        const std::vector<std::string>& texts) {
        if (texts.empty()) {
            return {};
        }
        std::vector<std::vector<int64_t>> encoded;
        encoded.reserve(texts.size());
        size_t max_len = 0;
        for (const auto& text : texts) {
            auto ids = tokenizer_.encode(
                text, static_cast<size_t>(std::max(2, spec_.max_length)));
            encoded.emplace_back(ids.begin(), ids.end());
            max_len = std::max(max_len, encoded.back().size());
        }

        const size_t batch = texts.size();
        std::vector<int64_t> ids(batch * max_len, 0);
        std::vector<int64_t> mask(batch * max_len, 0);
        std::vector<int64_t> types(batch * max_len, 0);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t t = 0; t < encoded[b].size(); ++t) {
                ids[b * max_len + t] = encoded[b][t];
                mask[b * max_len + t] = 1;
            }
        }

        const std::array<int64_t, 2> shape = {
            static_cast<int64_t>(batch), static_cast<int64_t>(max_len)};
        std::vector<Ort::Value> inputs;
        for (const auto& name : input_names_) {
            if (name == "input_ids") {
                inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                    memory_info_, ids.data(), ids.size(), shape.data(),
                    shape.size()));
            } else if (name == "attention_mask") {
                inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                    memory_info_, mask.data(), mask.size(), shape.data(),
                    shape.size()));
            } else if (name == "token_type_ids") {
                inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                    memory_info_, types.data(), types.size(), shape.data(),
                    shape.size()));
            } else {
                throw std::runtime_error("unsupported model input: " + name);
            }
        }

        auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_cstr_.data(),
                                     inputs.data(), inputs.size(),
                                     output_cstr_.data(), output_cstr_.size());
        if (outputs.empty()) {
            throw std::runtime_error("embedding model returned no output");
        }
        auto info = outputs[0].GetTensorTypeAndShapeInfo();
        auto out_shape = info.GetShape();
        if (out_shape.size() != 3) {
            throw std::runtime_error("unexpected embedding output rank");
        }
        const size_t out_batch = static_cast<size_t>(out_shape[0]);
        const size_t seq = static_cast<size_t>(out_shape[1]);
        const size_t hidden = static_cast<size_t>(out_shape[2]);
        dim_ = static_cast<int>(hidden);
        const float* data = outputs[0].GetTensorMutableData<float>();

        std::vector<std::vector<float>> result;
        result.reserve(out_batch);
        for (size_t b = 0; b < out_batch; ++b) {
            std::vector<float> vec(hidden, 0.0f);
            if (spec_.pooling == "cls") {
                std::copy(data + b * seq * hidden, data + b * seq * hidden + hidden,
                          vec.begin());
            } else {
                size_t count = 0;
                for (size_t t = 0; t < seq; ++t) {
                    if (mask[b * max_len + t] == 0) {
                        continue;
                    }
                    const float* row = data + (b * seq + t) * hidden;
                    for (size_t h = 0; h < hidden; ++h) {
                        vec[h] += row[h];
                    }
                    ++count;
                }
                if (count > 0) {
                    for (float& value : vec) {
                        value /= static_cast<float>(count);
                    }
                }
            }
            if (spec_.normalize) {
                double norm = 0.0;
                for (float value : vec) {
                    norm += static_cast<double>(value) * value;
                }
                norm = std::sqrt(norm);
                if (norm > 0.0) {
                    for (float& value : vec) {
                        value = static_cast<float>(value / norm);
                    }
                }
            }
            result.push_back(std::move(vec));
        }
        return result;
    }

    EmbeddingModelSpec spec_;
    BertTokenizer tokenizer_;
    Ort::Env env_;
    Ort::MemoryInfo memory_info_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_cstr_;
    std::vector<const char*> output_cstr_;
    int batch_size_ = 8;
    int dim_ = 0;
};

}  // namespace

std::unique_ptr<Embedder> make_onnx_embedder(
    const std::filesystem::path& model_dir, int batch_size, int threads) {
    return std::make_unique<OnnxEmbedder>(model_dir, batch_size, threads);
}

}  // namespace rag
