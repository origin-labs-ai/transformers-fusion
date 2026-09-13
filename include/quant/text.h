// ============================================================================
// text.h — Text processing types for Transcender
// ============================================================================
#pragma once
#include "quant/tensor.h"
#include <string>
#include <vector>

namespace quant {

// ========================================================================
// TextEncoder — encode text to latent representations
// ========================================================================
class TextEncoder {
public:
    TextEncoder() = default;
    ~TextEncoder() = default;

    struct Config {
        int64_t vocab_size = 50257;
        int64_t max_seq_len = 2048;
        int64_t hidden_size = 768;
        int64_t num_layers = 12;
        int64_t num_heads = 12;
    };

    void init(const Config& cfg);
    Tensor encode(const std::string& text);
    Tensor encode(const std::vector<int32_t>& token_ids);

    int64_t hidden_size() const { return hidden_size_; }
    bool is_initialized() const { return initialized_; }

private:
    int64_t hidden_size_ = 0;
    bool initialized_ = false;
};

// ========================================================================
// TextDecoder — decode latent representations to text
// ========================================================================
class TextDecoder {
public:
    TextDecoder() = default;
    ~TextDecoder() = default;

    std::string decode(const Tensor& hidden_states);
    std::string decode(const std::vector<int32_t>& token_ids);

private:
};

} // namespace quant
