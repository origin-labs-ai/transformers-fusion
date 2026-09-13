// ============================================================================
// embeddings.h — Embedding model types for Transcender
// ============================================================================
#pragma once
#include "quant/tensor.h"

namespace quant {

// ========================================================================
// Sentence Embedding — project token embeddings to fixed-size vector
// ========================================================================
class SentenceEmbedder {
public:
    SentenceEmbedder(int64_t hidden_size, int64_t output_size);
    ~SentenceEmbedder() = default;

    Tensor forward(const Tensor& token_embeddings);

    Tensor projection;
    int64_t hidden_size() const { return hidden_size_; }
    int64_t output_size() const { return output_size_; }

private:
    int64_t hidden_size_;
    int64_t output_size_;
};

} // namespace quant
