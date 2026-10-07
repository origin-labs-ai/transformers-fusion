#pragma once
// ============================================================================
// PILLAR 3: VectorQuantizer — EMA-based VQ with commitment loss
// ============================================================================
// WHY: Audio/Video codecs (EnCodec, DAC) need discrete codebooks. The VQ
// encoder maps continuous embeddings to the nearest codebook vector.
// EMA updates ensure stable codebook learning without exploding gradients.
//
// COMMITMENT LOSS: Encourages encoder outputs to stay close to codebook
// vectors. Without it, codebook vectors collapse to a few entries.
//
// FORMATS: Supports Q8 (256 entries, FP32) and Q4 (16 entries, FP16).
// ============================================================================

#include "quant/tensor.h"
#include "quant/codebook.h"
#include <vector>
#include <cstdint>
#include <random>

namespace quant {

// ============================================================================
// VectorQuantizer — Core VQ with EMA centroid updates
// ============================================================================

struct VQConfig {
    int64_t codebook_size = 256;     // Number of codebook entries
    int64_t embedding_dim = 64;      // Dimension of each entry
    float commitment_weight = 0.25f; // Commitment loss weight (beta)
    float ema_decay = 0.99f;         // EMA decay for centroid updates
    float codebook_init_std = 1.0f;  // Initialization std for codebook
    bool use_ema = true;             // Use EMA vs gradient updates
};

class VectorQuantizer {
public:
    explicit VectorQuantizer(const VQConfig& cfg = VQConfig{});

    // Forward: quantize embeddings to nearest codebook vectors
    // Returns: quantized embeddings, indices, commitment loss, codebook loss
    struct VQResult {
        Tensor quantized;       // {batch, seq_len, embedding_dim}
        Tensor indices;         // {batch, seq_len} — codebook indices
        float commitment_loss;  // ||z_e - sg(z_q)||^2
        float codebook_loss;    // ||sg(z_e) - z_q||^2 (EMA: 0)
    };

    VQResult forward(const Tensor& embeddings) const;

    // Backward: straight-through estimator (identity gradient)
    Tensor backward(const Tensor& grad_output);

    // EMA update (called after each batch)
    void update_ema(const Tensor& embeddings, const Tensor& indices);

    // Gradient-based codebook update (when use_ema = false)
    void update_codebook_grad(const Tensor& grad_quantized, const Tensor& indices);

    // Access codebook
    const Tensor& codebook() const { return codebook_; }
    Tensor& codebook() { return codebook_; }

    const VQConfig& config() const { return cfg_; }

    // Stats
    int64_t usage_rate() const; // Fraction of codebook entries used
    float perplexity() const;   // exp(entropy) — effective codebook usage

private:
    VQConfig cfg_;
    Tensor codebook_;           // {codebook_size, embedding_dim}
    Tensor codebook_counts_;    // {codebook_size} — usage counts for EMA
    Tensor codebook_sum_;       // {codebook_size, embedding_dim} — EMA accumulator

    void initialize_codebook();
    Tensor find_nearest(const Tensor& embeddings) const;
};

// ============================================================================
// Q8 Codebook VQ — 256-entry FP32 codebook for salient weights
// ============================================================================
// Maps quantized indices (uint8) back to FP32 values via lookup table.
// Used for the 1% most important weights in the QUANT format.
// ============================================================================

class Q8VectorQuantizer {
public:
    Q8VectorQuantizer();
    explicit Q8VectorQuantizer(const std::vector<float>& initial_codebook);

    // Quantize FP32 weights to Q8 indices
    std::vector<uint8_t> quantize(const float* weights, int64_t n);

    // Dequantize Q8 indices back to FP32
    void dequantize(const uint8_t* indices, float* output, int64_t n) const;

    // Update codebook via EMA
    void update_ema(const float* weights, const uint8_t* indices, int64_t n);

    const std::vector<float>& codebook() const { return codebook_; }
    std::vector<float>& codebook() { return codebook_; }

private:
    std::vector<float> codebook_; // 256 FP32 values
    std::vector<float> codebook_sum_;
    std::vector<int64_t> codebook_counts_;

    int find_nearest(float value) const;
};

// ============================================================================
// Q4 Codebook VQ — 16-entry FP16 codebook for moderate importance weights
// ============================================================================
// Uses packed nibble indices (2 indices per byte).
// 4x compression vs Q8, used for the next4% of weights.
// ============================================================================

class Q4VectorQuantizer {
public:
    Q4VectorQuantizer();
    explicit Q4VectorQuantizer(const std::vector<uint16_t>& initial_codebook);

    // Quantize FP32 weights to Q4 packed nibble indices
    // Returns packed: each byte holds 2 nibble indices
    std::vector<uint8_t> quantize(const float* weights, int64_t n);

    // Dequantize Q4 packed indices back to FP32
    void dequantize(const uint8_t* packed_indices, float* output, int64_t n) const;

    // Update codebook via EMA
    void update_ema(const float* weights, const uint8_t* packed_indices, int64_t n);

    const std::vector<uint16_t>& codebook() const { return codebook_; }
    std::vector<uint16_t>& codebook() { return codebook_; }

private:
    std::vector<uint16_t> codebook_; // 16 FP16 values (stored as uint16)
    std::vector<float> codebook_sum_;
    std::vector<int64_t> codebook_counts_;

    int find_nearest(float value) const;

    // FP16 <-> float conversion (software fallback)
    static float fp16_to_float(uint16_t h);
    static uint16_t float_to_fp16(float f);
};

// ============================================================================
// Audio Codebook — EnCodec/DAC style residual vector quantization
// ============================================================================
// RVQ: Multiple VQ layers in cascade. Each layer quantizes the residual
// from the previous layer. Used for audio tokenization.
//
// Typical: 8 VQ layers, codebook_size=1024 each
// ============================================================================

struct AudioCodebookConfig {
    int64_t num_quantizers = 8;     // Number of RVQ layers
    int64_t codebook_size = 1024;   // Entries per layer
    int64_t embedding_dim = 128;    // Embedding dimension
    float commitment_weight = 0.25f;
    float ema_decay = 0.99f;
};

class AudioCodebook {
public:
    explicit AudioCodebook(const AudioCodebookConfig& cfg = AudioCodebookConfig{});

    // Encode: continuous audio features -> discrete token indices
    // Returns: {num_quantizers, batch, seq_len} indices
    Tensor encode(const Tensor& features);

    // Decode: discrete token indices -> continuous audio features
    Tensor decode(const Tensor& indices);

    // Get all VQ layers
    const std::vector<VectorQuantizer>& quantizers() const { return quantizers_; }
    std::vector<VectorQuantizer>& quantizers() { return quantizers_; }

    // Total codebook loss across all layers
    float total_commitment_loss() const;
    float total_codebook_loss() const;

private:
    AudioCodebookConfig cfg_;
    std::vector<VectorQuantizer> quantizers_;
};

} // namespace quant
