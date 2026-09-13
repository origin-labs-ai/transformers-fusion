// ============================================================================
// PILLAR 3: VectorQuantizer — EMA-based VQ + QUANT8/QUANT4 Codebooks
// ============================================================================

#include "quant/vector_quantizer.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace quant {

// ============================================================================
// VectorQuantizer — Core VQ with EMA
// ============================================================================

VectorQuantizer::VectorQuantizer(const VQConfig& cfg) : cfg_(cfg) {
    initialize_codebook();
}

void VectorQuantizer::initialize_codebook() {
    codebook_ = Tensor(Shape{cfg_.codebook_size, cfg_.embedding_dim}, DType::F32);
    codebook_counts_ = Tensor(Shape{cfg_.codebook_size}, DType::F32);
    codebook_sum_ = Tensor(Shape{cfg_.codebook_size, cfg_.embedding_dim}, DType::F32);

    // Xavier-like initialization
    float scale = std::sqrt(2.0f / cfg_.embedding_dim) * cfg_.codebook_init_std;
    float* cb = codebook_.data<float>();
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, scale);
    for (int64_t i = 0; i < cfg_.codebook_size * cfg_.embedding_dim; ++i) {
        cb[i] = dist(rng);
    }
    codebook_counts_.zero_();
    codebook_sum_.zero_();
}

Tensor VectorQuantizer::find_nearest(const Tensor& embeddings) const {
    int64_t flat_dim = embeddings.numel() / cfg_.embedding_dim;
    Tensor indices(Shape{flat_dim}, DType::I64);
    indices.zero_();

    const float* emb = embeddings.data<float>();
    const float* cb = codebook_.data<float>();
    int64_t* idx = indices.data<int64_t>();

    for (int64_t i = 0; i < flat_dim; ++i) {
        float min_dist = 1e30f;
        int64_t best = 0;
        for (int64_t j = 0; j < cfg_.codebook_size; ++j) {
            float dist = 0.0f;
            for (int64_t d = 0; d < cfg_.embedding_dim; ++d) {
                float diff = emb[i * cfg_.embedding_dim + d] - cb[j * cfg_.embedding_dim + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best = j;
            }
        }
        idx[i] = best;
    }
    return indices;
}

VectorQuantizer::VQResult VectorQuantizer::forward(const Tensor& embeddings) const {
    VQResult result;
    int64_t flat_dim = embeddings.numel() / cfg_.embedding_dim;

    // Find nearest codebook entries
    result.indices = find_nearest(embeddings);

    // Gather quantized vectors
    result.quantized = Tensor(embeddings.shape(), DType::F32);
    float* quant = result.quantized.data<float>();
    const float* emb = embeddings.data<float>();
    const float* cb = codebook_.data<float>();
    const int64_t* idx = result.indices.data<int64_t>();

    for (int64_t i = 0; i < flat_dim; ++i) {
        int64_t code = idx[i];
        for (int64_t d = 0; d < cfg_.embedding_dim; ++d) {
            quant[i * cfg_.embedding_dim + d] = cb[code * cfg_.embedding_dim + d];
        }
    }

    // Commitment loss: ||z_e - sg(z_q)||^2
    float commit_loss = 0.0f;
    for (int64_t i = 0; i < embeddings.numel(); ++i) {
        float diff = emb[i] - quant[i];
        commit_loss += diff * diff;
    }
    result.commitment_loss = cfg_.commitment_weight * commit_loss / embeddings.numel();

    // Codebook loss (0 for EMA, non-zero for gradient updates)
    result.codebook_loss = 0.0f;

    return result;
}

Tensor VectorQuantizer::backward(const Tensor& grad_output) {
    // STE: pass gradients through as-is (identity)
    return grad_output.clone();
}

void VectorQuantizer::update_ema(const Tensor& embeddings, const Tensor& indices) {
    int64_t flat_dim = embeddings.numel() / cfg_.embedding_dim;
    const float* emb = embeddings.data<float>();
    const int64_t* idx = indices.data<int64_t>();
    float* cb = codebook_.data<float>();
    float* counts = codebook_counts_.data<float>();
    float* sums = codebook_sum_.data<float>();

    // Accumulate
    for (int64_t i = 0; i < flat_dim; ++i) {
        int64_t code = idx[i];
        counts[code] += 1.0f;
        for (int64_t d = 0; d < cfg_.embedding_dim; ++d) {
            sums[code * cfg_.embedding_dim + d] += emb[i * cfg_.embedding_dim + d];
        }
    }

    // EMA update
    for (int64_t j = 0; j < cfg_.codebook_size; ++j) {
        if (counts[j] < 1.0f) continue;
        float decay = cfg_.ema_decay;
        float nudge = (1.0f - decay) * counts[j];
        for (int64_t d = 0; d < cfg_.embedding_dim; ++d) {
            cb[j * cfg_.embedding_dim + d] =
                decay * cb[j * cfg_.embedding_dim + d] +
                nudge * sums[j * cfg_.embedding_dim + d] / counts[j];
        }
        counts[j] *= decay;
        for (int64_t d = 0; d < cfg_.embedding_dim; ++d) {
            sums[j * cfg_.embedding_dim + d] *= decay;
        }
    }
}

int64_t VectorQuantizer::usage_rate() const {
    const float* counts = codebook_counts_.data<float>();
    int64_t used = 0;
    for (int64_t i = 0; i < cfg_.codebook_size; ++i) {
        if (counts[i] > 0.5f) used++;
    }
    return used;
}

float VectorQuantizer::perplexity() const {
    const float* counts = codebook_counts_.data<float>();
    float total = 0.0f;
    for (int64_t i = 0; i < cfg_.codebook_size; ++i) total += counts[i];
    if (total < 1.0f) return 0.0f;
    float entropy = 0.0f;
    for (int64_t i = 0; i < cfg_.codebook_size; ++i) {
        float p = counts[i] / total;
        if (p > 1e-10f) entropy -= p * std::log(p);
    }
    return std::exp(entropy);
}

// ============================================================================
// QUANT8VectorQuantizer — 256-entry FP32 codebook
// ============================================================================

QUANT8VectorQuantizer::QUANT8VectorQuantizer() {
    codebook_.resize(256);
    codebook_sum_.resize(256);
    codebook_counts_.resize(256, 0);
    // Initialize with uniform distribution
    for (int i = 0; i < 256; ++i) {
        codebook_[i] = (i - 128) / 128.0f; // Range [-1, 1]
    }
}

QUANT8VectorQuantizer::QUANT8VectorQuantizer(const std::vector<float>& initial_codebook)
    : codebook_(initial_codebook) {
    codebook_sum_.resize(256);
    codebook_counts_.resize(256, 0);
}

int QUANT8VectorQuantizer::find_nearest(float value) const {
    float min_dist = 1e30f;
    int best = 0;
    for (int i = 0; i < 256; ++i) {
        float diff = value - codebook_[i];
        float dist = diff * diff;
        if (dist < min_dist) {
            min_dist = dist;
            best = i;
        }
    }
    return best;
}

std::vector<uint8_t> QUANT8VectorQuantizer::quantize(const float* weights, int64_t n) {
    std::vector<uint8_t> indices(n);
    for (int64_t i = 0; i < n; ++i) {
        indices[i] = static_cast<uint8_t>(find_nearest(weights[i]));
    }
    return indices;
}

void QUANT8VectorQuantizer::dequantize(const uint8_t* indices, float* output, int64_t n) const {
    for (int64_t i = 0; i < n; ++i) {
        output[i] = codebook_[indices[i]];
    }
}

void QUANT8VectorQuantizer::update_ema(const float* weights, const uint8_t* indices, int64_t n) {
    std::fill(codebook_sum_.begin(), codebook_sum_.end(), 0.0f);
    std::fill(codebook_counts_.begin(), codebook_counts_.end(), 0);
    for (int64_t i = 0; i < n; ++i) {
        int idx = indices[i];
        codebook_counts_[idx] += 1;
        codebook_sum_[idx] += weights[i];
    }
    for (int i = 0; i < 256; ++i) {
        if (codebook_counts_[i] > 0.5f) {
            codebook_[i] = 0.99f * codebook_[i] + 0.01f * codebook_sum_[i] / codebook_counts_[i];
        }
    }
}

// ============================================================================
// QUANT4VectorQuantizer — 16-entry FP16 codebook
// ============================================================================

QUANT4VectorQuantizer::QUANT4VectorQuantizer() {
    codebook_.resize(16);
    codebook_sum_.resize(16);
    codebook_counts_.resize(16, 0);
    for (int i = 0; i < 16; ++i) {
        codebook_[i] = float_to_fp16((i - 8) / 8.0f);
    }
}

QUANT4VectorQuantizer::QUANT4VectorQuantizer(const std::vector<uint16_t>& initial_codebook)
    : codebook_(initial_codebook) {
    codebook_sum_.resize(16);
    codebook_counts_.resize(16, 0);
}

float QUANT4VectorQuantizer::fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        f = (sign << 31) | (frac << 13);
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (frac << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

uint16_t QUANT4VectorQuantizer::float_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(uint32_t));
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (bits >> 13) & 0x3FF;
    if (exp <= 0) return static_cast<uint16_t>((sign << 15));
    if (exp >= 31) return static_cast<uint16_t>((sign << 15) | 0x7C00);
    return static_cast<uint16_t>((sign << 15) | (exp << 10) | frac);
}

int QUANT4VectorQuantizer::find_nearest(float value) const {
    float min_dist = 1e30f;
    int best = 0;
    for (int i = 0; i < 16; ++i) {
        float cb_val = fp16_to_float(codebook_[i]);
        float diff = value - cb_val;
        float dist = diff * diff;
        if (dist < min_dist) {
            min_dist = dist;
            best = i;
        }
    }
    return best;
}

std::vector<uint8_t> QUANT4VectorQuantizer::quantize(const float* weights, int64_t n) {
    int64_t packed_size = (n + 1) / 2;
    std::vector<uint8_t> packed(packed_size, 0);
    for (int64_t i = 0; i < n; ++i) {
        int idx = find_nearest(weights[i]);
        if (i % 2 == 0) {
            packed[i / 2] = (idx & 0x0F);
        } else {
            packed[i / 2] |= (idx & 0x0F) << 4;
        }
    }
    return packed;
}

void QUANT4VectorQuantizer::dequantize(const uint8_t* packed_indices, float* output, int64_t n) const {
    for (int64_t i = 0; i < n; ++i) {
        uint8_t byte = packed_indices[i / 2];
        int nibble = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        output[i] = fp16_to_float(codebook_[nibble]);
    }
}

void QUANT4VectorQuantizer::update_ema(const float* weights, const uint8_t* packed_indices, int64_t n) {
    std::fill(codebook_sum_.begin(), codebook_sum_.end(), 0.0f);
    std::fill(codebook_counts_.begin(), codebook_counts_.end(), 0);
    for (int64_t i = 0; i < n; ++i) {
        uint8_t byte = packed_indices[i / 2];
        int idx = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        codebook_counts_[idx] += 1;
        codebook_sum_[idx] += weights[i];
    }
    for (int i = 0; i < 16; ++i) {
        if (codebook_counts_[i] > 0.5f) {
            float new_val = 0.99f * fp16_to_float(codebook_[i]) +
                           0.01f * codebook_sum_[i] / codebook_counts_[i];
            codebook_[i] = float_to_fp16(new_val);
        }
    }
}

// ============================================================================
// AudioCodebook — Residual Vector Quantization
// ============================================================================

AudioCodebook::AudioCodebook(const AudioCodebookConfig& cfg) : cfg_(cfg) {
    VQConfig vq_cfg;
    vq_cfg.codebook_size = cfg.codebook_size;
    vq_cfg.embedding_dim = cfg.embedding_dim;
    vq_cfg.commitment_weight = cfg.commitment_weight;
    vq_cfg.ema_decay = cfg.ema_decay;
    for (int64_t i = 0; i < cfg.num_quantizers; ++i) {
        quantizers_.emplace_back(vq_cfg);
    }
}

Tensor AudioCodebook::encode(const Tensor& features) {
    int64_t batch = features.dim(0);
    int64_t seq_len = features.dim(1);
    int64_t emb_dim = features.dim(2);

    Tensor indices(Shape{cfg_.num_quantizers, batch, seq_len}, DType::I64);
    indices.zero_();

    Tensor residual = features.clone();
    for (int64_t q = 0; q < cfg_.num_quantizers; ++q) {
        auto result = quantizers_[q].forward(residual);
        // Store indices
        int64_t* idx_data = indices.data<int64_t>() + q * batch * seq_len;
        const int64_t* res_idx = result.indices.data<int64_t>();
        std::memcpy(idx_data, res_idx, batch * seq_len * sizeof(int64_t));
        // Compute residual
        for (int64_t i = 0; i < residual.numel(); ++i) {
            residual.data<float>()[i] -= result.quantized.data<float>()[i];
        }
    }
    return indices;
}

Tensor AudioCodebook::decode(const Tensor& indices) {
    int64_t batch = indices.dim(1);
    int64_t seq_len = indices.dim(2);
    Tensor output(Shape{batch, seq_len, cfg_.embedding_dim}, DType::F32);
    output.zero_();

    for (int64_t q = 0; q < cfg_.num_quantizers; ++q) {
        const float* cb = quantizers_[q].codebook().data<float>();
        const int64_t* idx = indices.data<int64_t>() + q * batch * seq_len;
        float* out = output.data<float>();
        for (int64_t i = 0; i < batch * seq_len; ++i) {
            int64_t code = idx[i];
            for (int64_t d = 0; d < cfg_.embedding_dim; ++d) {
                out[i * cfg_.embedding_dim + d] += cb[code * cfg_.embedding_dim + d];
            }
        }
    }
    return output;
}

float AudioCodebook::total_commitment_loss() const {
    float total = 0.0f;
    for (auto& q : quantizers_) total += q.forward(Tensor(Shape{1, 1, cfg_.embedding_dim})).commitment_loss;
    return total;
}

float AudioCodebook::total_codebook_loss() const {
    return 0.0f; // EMA mode
}

} // namespace quant
