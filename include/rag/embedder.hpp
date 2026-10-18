#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rag {

struct EmbeddingModelSpec {
    std::string name = "bge-small-en-v1.5";
    std::string onnx_file = "model_quantized.onnx";
    std::string vocab_file = "vocab.txt";
    int dim = 384;
    int max_length = 512;
    std::string pooling = "cls";  // cls or mean
    bool normalize = true;
    std::string query_prefix;
    std::string doc_prefix;
};

EmbeddingModelSpec load_model_spec(const std::filesystem::path& model_dir);

class Embedder {
public:
    virtual ~Embedder() = default;

    virtual int dim() const = 0;
    virtual const EmbeddingModelSpec& spec() const = 0;
    virtual std::vector<float> embed_query(std::string_view text) = 0;
    virtual std::vector<std::vector<float>> embed_documents(
        const std::vector<std::string>& texts) = 0;
};

std::unique_ptr<Embedder> make_onnx_embedder(
    const std::filesystem::path& model_dir,
    int batch_size = 8,
    int threads = 0);

}  // namespace rag
