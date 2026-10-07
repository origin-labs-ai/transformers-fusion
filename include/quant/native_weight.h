#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <memory>
#include "quant/types.h"

namespace quant {
namespace native {

enum class NativeFormat : uint8_t {
    Q8 = 0,  // 8-bit index, 256-entry FP32 codebook, per-weight scale
    Q1 = 1,  // 1-bit index, 3 centroids {-1,0,+1}, per-block scale (B=128 default)
    Q4 = 2,  // 4-bit index, 16-entry FP32 codebook, per-block scale (B=128 default)
};

inline const char* native_format_name(NativeFormat f) {
    switch (f) {
        case NativeFormat::Q8: return "native_quant8";
        case NativeFormat::Q1: return "native_quant1";
        case NativeFormat::Q4: return "native_quant4";
        default: return "unknown";
    }
}

// QUANT codebook values
inline float quant1_value(uint8_t idx) {
    return static_cast<float>(static_cast<int>(idx)) - 1.0f; // 0→-1, 1→0, 2→+1
}

// Q4 codebook — 16 FP32 centroids shared per block
class Q4Codebook {
public:
    static constexpr size_t K = 16;
    Q4Codebook();
    explicit Q4Codebook(const float* centroids);
    void train(const float* data, size_t n, int iterations = 10);
    uint8_t nearest(float val) const;
    float centroid(uint8_t idx) const { return centroids_[idx]; }
    const float* data() const { return centroids_; }
    void set_quantile_spacing(const float* data, size_t n);
    size_t size() const { return K; }
private:
    float centroids_[K];
};

// Dead zone threshold per format (Δ_c/2 where Δ_c = min codebook gap)
inline float dead_zone_radius(NativeFormat fmt, float scale) {
    switch (fmt) {
        case NativeFormat::Q8: return scale * 0.00390625f; // 1/256 of normalized range
        case NativeFormat::Q1: return scale * 0.5f;       // Δ_c=1 between {-1,0} or {0,+1}
        case NativeFormat::Q4: return scale * 0.0625f;    // Δ_c≈0.125 for 16-entry uniform codebook over [-1,1]
        default: return 0.0f;
    }
}

// Shared Q8 codebook: 256 FP32 entries, trained once via k-means or quantile spacing
class Q8Codebook {
public:
    static constexpr size_t K = 256;
    Q8Codebook();
    explicit Q8Codebook(const float* centroids);
    void train(const float* data, size_t n, int iterations = 20);
    uint8_t nearest(float val) const;
    float centroid(uint8_t idx) const { return centroids_[idx]; }
    const float* data() const { return centroids_; }
    void set_quantile_spacing(const float* data, size_t n);
    size_t size() const { return K; }
private:
    float centroids_[K];
};

// Per-weight representation (B=1 mode, theoretical)
struct QUANTWeight {
    uint8_t idx;
    float scale;
    NativeFormat fmt;
    float dequantize() const;
    uint8_t quantize(float val) const;
};

// Block-shared representation (B=128 default)
struct QUANTBlock {
    std::unique_ptr<uint8_t[]> indices;  // [block_size]
    float scale;
    NativeFormat fmt;
    size_t block_size;

    QUANTBlock(NativeFormat f, size_t B);
    QUANTBlock(const QUANTBlock&) = delete;
    QUANTBlock& operator=(const QUANTBlock&) = delete;
    ~QUANTBlock() = default;
    float dequantize(size_t i) const;
    uint8_t quantize(size_t i, float val) const;
};

// Native QUANT weight store — no FP32 master weights
class NativeQWeightStore {
public:
    NativeQWeightStore(size_t num_weights, size_t block_size = 128);
    ~NativeQWeightStore() = default;

    // Initialize from FP32 weights + sensitivity estimates
    // allocate: assign Q8 to top frac_quant8 by sensitivity, Q1 to middle, Q4 to bottom
    void initialize(const float* fp32_weights, const float* sensitivity,
                    float frac_quant8 = 0.01f, float frac_quant1 = 0.95f);

    // Convert FP32 weights to QUANT format using existing format assignment
    void convert_from_fp32(const float* src);

    // Dequantize all weights to FP32 buffer
    void dequantize(float* dst) const;

    // Two-timescale SGD update (Theorem 5d.3)
    // grad: gradient w.r.t. dequantized weights (∂L/∂w, size = num_weights)
    // lr_scale: learning rate for continuous scale updates
    // lr_weight: effective learning rate for index changes (η in Theorem 5d.3)
    void apply_quant_update(const float* grad, float lr_scale, float lr_weight);

    // CID allocation: reassign formats based on accumulated sensitivity
    void reallocate_by_sensitivity(const float* sensitivity,
                                    float frac_quant8 = 0.01f,
                                    float frac_quant1 = 0.95f);

    // Access
    size_t size() const { return num_weights_; }
    size_t num_blocks() const { return num_blocks_; }
    size_t block_size() const { return block_size_; }
    uint8_t get_index(size_t i) const;
    float get_scale(size_t i) const;
    NativeFormat get_format(size_t i) const;
    float get_weight(size_t i) const;
    const uint8_t* indices_data() const { return indices_.get(); }
    uint8_t* indices_data() { return indices_.get(); }
    const float* block_scales_data() const { return block_scales_.get(); }
    float* block_scales_data() { return block_scales_.get(); }
    const NativeFormat* formats_data() const { return formats_.get(); }

    // Static codebooks (shared across all instances)
    static Q8Codebook& global_codebook();
    static Q4Codebook& global_quant4_codebook();

private:
    size_t num_weights_;
    size_t block_size_;
    size_t num_blocks_;
    std::unique_ptr<NativeFormat[]> formats_;        // [num_blocks_]
    std::unique_ptr<uint8_t[]> indices_;             // [num_weights_]
    std::unique_ptr<float[]> block_scales_;          // [num_blocks_]
    std::unique_ptr<bool[]> frozen_flag_;            // [num_weights_] — index is frozen (dead zone)
};

} // namespace native
} // namespace quant
