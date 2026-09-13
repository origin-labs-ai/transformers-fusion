#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/codebook.h"
#include "quant/qat.h"

namespace quant {

// Straight-Through Estimator for QUANT-native training
class STEQuantizer {
public:
    STEQuantizer() = default;
    explicit STEQuantizer(Format target_format);
    
    // Forward: quantize weights to target format
    // Backward: identity gradient (straight-through)
    Tensor forward(const Tensor& fp32_weight);

    // Mixed-format forward: different formats per block
    // per_block_formats[i] specifies the format for the i-th block of size block_size
    // The last block may be smaller than block_size
    Tensor forward_mixed(const Tensor& weights, const std::vector<Format>& per_block_formats, int block_size = 256);
    
    // Quantize with codebook training (caller-owned exclusive: the codebook
    // must be exclusively owned by the caller for the call — trained in place
    // if untrained and mutated via batch stats; do not share across threads).
    Tensor quantize_with_codebook(const Tensor& fp32_weight, CodebookQUANT8& codebook);
    Tensor quantize_with_codebook(const Tensor& fp32_weight, CodebookQUANT4& codebook);
    
    // Quantize to QUANT/Q1 with scale
    void quantize_quant(const float* src, uint8_t* dst, float* scale, int64_t n);
    void quantize_Q1(const float* src, uint8_t* dst, float* scale, int64_t n);
    
    // Set target format (NOT thread-safe: mutates target_format_ without
    // synchronization; external locking required if shared across threads).
    void set_target_format(Format fmt);
    Format target_format() const;

    // QAT integration: fake-quant wrapper using this STEQuantizer's format
    // STE flow: gradient passes unchanged (via qat::FakeQuantizeFunction)
    Tensor fake_quantize_qat(const Tensor& fp32_weight, float scale_override = 0.0f);
    // LSQ wrapper: trainable scale
    Tensor lsq_quantize(const Tensor& fp32_weight, Tensor& scale_param);
    // Observer-calibrated scale hook
    Tensor fake_quantize_with_observer(const Tensor& fp32_weight, qat::Observer& obs, int bits = 0);
    
private:
    Format target_format_ = Format::Q1;
    
    // Find scale factor (max abs)
    float find_scale(const float* data, int64_t n);
    int bits_for_format(Format f) const;
};

} // namespace quant
