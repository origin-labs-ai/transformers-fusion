#pragma once

#include "quant/types.h"
#include <string>
#include <vector>

namespace quant {

enum class RegFormat : uint32_t {
    Q1 = 0,
    Q2,
    Q3,
    Q4,
    Q6,
    Q8,
    Q12,
    Q16,
    Q24,
    Q32,
    Q1_G,
    Q2_G,
    Q3_G,
    Q4_G,
    Q6_G,
    Q8_G,
    Q12_G,
    Q16_G,
    Q24_G,
    MXQ_3_5,
    MXQ_4_5,
    MXQ_6_5,
    MXQ_8_5,
    MXQ_12_5,
    MXQ_16_5,
    MXQ_24_5,
    MXQ_3_5_G,
    MXQ_4_5_G,
    MXQ_6_5_G,
    MXQ_8_5_G,
    MXQ_12_5_G,
    MXQ_16_5_G,
    MXQ_24_5_G
};

struct FormatDescriptor {
    std::string name;
    RegFormat id;
    float bpw;
    int num_centroids;
    bool lossless;
    bool grouped;
    int num_groups;
    float group_size;
    float est_mse;
    std::string description;
};

struct MixDescriptor {
    std::string name;
    RegFormat id;
    int num_tiers;
    RegFormat tier1_fmt;
    float tier1_ratio;
    RegFormat tier2_fmt;
    float tier2_ratio;
    RegFormat tier3_fmt;
    float tier3_ratio;
    RegFormat tier4_fmt;
    float tier4_ratio;
    float effective_bpw;
    // Adaptive mixes (Q_MIX): blocks are assigned to member formats by
    // measured reconstruction benefit per byte, under a HARD budget equal to
    // the claimed effective_bpw. Non-adaptive mixes keep the registry ratios.
    bool adaptive = false;
};

struct QuantResult {
    FormatDescriptor format;
    // Canonical wire payload: the exact bytes the block codec stores on disk,
    // concatenated per 256-weight wire block (block_codec.h). dequantize()
    // walks block_idx_bytes/block_cb_bytes to slice each block's payload.
    std::vector<uint8_t> indices;
    std::vector<uint8_t> codebook;         // wire codebook channel (Q1 block means)
    std::vector<uint32_t> block_idx_bytes; // per wire block: indices payload bytes
    std::vector<uint32_t> block_cb_bytes;  // per wire block: codebook payload bytes
    // In-memory metadata kept for compatibility (Q32 FP32 copy and any
    // caller-visible scales). NOT stored on disk — the wire payload in
    // indices/codebook is the complete serialized form.
    std::vector<float> codebook_fp32;
    std::vector<float> group_scales;
    std::vector<float> group_zero_points;
    float global_scale;
    int64_t num_elements;
    int block_size;
    bool success;
};

class FormatRegistry {
public:
    // Per-block layout produced by the adaptive mix allocator. Block sizes
    // may differ per block (row-aligned segmentation for narrow 2D tensors);
    // every block's byte claim plus the format table entry is fully
    // self-describing on disk (num_weights is stored per block).
    struct MixBlockPlan {
        std::vector<Format> formats;       // per block, wire formats 0..14
        std::vector<int64_t> block_starts; // flat offset of each block in data
        std::vector<int64_t> block_lens;   // weights per block
    };

    static const std::vector<FormatDescriptor>& get_all_singles();
    static const std::vector<MixDescriptor>& get_all_twi_mixes();
    static const std::vector<MixDescriptor>& get_all_four_mixes();

    static FormatDescriptor get_single_format(float target_bpw);
    static MixDescriptor get_twi_mix(float target_bpw);
    static MixDescriptor get_four_mix(float target_bpw);

    static QuantResult quantize(const float* data, int64_t n, const FormatDescriptor& fmt);
    static QuantResult quantize_block(const float* block, size_t block_size, const FormatDescriptor& fmt);
    static void dequantize(const QuantResult& qr, float* output, int64_t n);
    // Counts every byte required to reconstruct the quantized payload.
    static size_t serialized_size_bytes(const QuantResult& qr);
    static float actual_bpw(const QuantResult& qr);
    static float measure_mse(const float* original, const float* dequantized, int64_t n);

    static float evaluate_format_quality(const float* data, int64_t n, const FormatDescriptor& fmt);
    static float evaluate_format_quality_weighted(const float* data, const float* gradients,
                                                  int64_t n, const FormatDescriptor& fmt,
                                                  float fisher_weight = 0.5f);
    static FormatDescriptor select_best_format(float target_bpw, const float* data, int64_t n);
    static MixDescriptor select_best_mix(float target_bpw, const float* data, int64_t n);
    static std::vector<FormatDescriptor> apply_forced_distribution(
            float target_bpw, int num_formats, const float* data, int64_t n);

    static std::string get_format_table();
    static FormatDescriptor parse_format_name(const std::string& name);

    static float compute_average_bpw(const std::vector<FormatDescriptor>& assignment);

    // Per-block member-format assignment that treats the claimed effective
    // BPW as a HARD byte budget: the returned plan NEVER costs more than
    // ceil(effective_bpw * n / 8) bytes, and spends every byte only where it
    // measurably reduces reconstruction error (benefit per byte spent).
    // `shape` (optional): for rank-2 tensors whose column count is below the
    // block size, blocks are aligned to whole ROWS (one block per row) so
    // scales and tiers follow the matrix's row/column structure; otherwise
    // fixed block_size blocks are used, which are row-aligned whenever the
    // block size divides the column count.
    static MixBlockPlan allocate_mix_blocks(const MixDescriptor& mix,
                                            const float* data, int64_t n,
                                            int block_size,
                                            const std::vector<int64_t>* shape = nullptr);

    static QuantResult quantize_q1(const float* data, int64_t n);
    static QuantResult quantize_q2(const float* data, int64_t n);
    static QuantResult quantize_q4(const float* data, int64_t n);
    static QuantResult quantize_q8(const float* data, int64_t n);
    static QuantResult quantize_q_twi_mix_1_5(const float* data, int64_t n, int block_size);
    static QuantResult quantize_q_sparse(const float* data, int64_t n);

private:
    static FormatDescriptor find_closest_single(float target_bpw);
};

inline RegFormat format_to_regformat(Format f) {
    switch(f) {
        case Format::Q1: case Format::Q1_K_L: case Format::Q1_K_M: case Format::Q1_K_H: case Format::Q1_G: case Format::Q1_K_L_G: case Format::Q1_K_M_G: case Format::Q1_K_H_G: return RegFormat::Q1;
        case Format::Q2: case Format::Q2_K_L: case Format::Q2_K_M: case Format::Q2_K_H: case Format::Q2_G: case Format::Q2_K_L_G: case Format::Q2_K_M_G: case Format::Q2_K_H_G: return RegFormat::Q2;
        case Format::Q3: case Format::Q3_K_L: case Format::Q3_K_M: case Format::Q3_K_H: case Format::Q3_G: case Format::Q3_K_L_G: case Format::Q3_K_M_G: case Format::Q3_K_H_G: return RegFormat::Q3;
        case Format::Q4: case Format::Q4_K_L: case Format::Q4_K_M: case Format::Q4_K_H: case Format::Q4_G: case Format::Q4_K_L_G: case Format::Q4_K_M_G: case Format::Q4_K_H_G: return RegFormat::Q4;
        case Format::Q6: case Format::Q6_K_L: case Format::Q6_K_M: case Format::Q6_K_H: case Format::Q6_G: case Format::Q6_K_L_G: case Format::Q6_K_M_G: case Format::Q6_K_H_G: return RegFormat::Q6;
        case Format::Q8: case Format::Q8_K_L: case Format::Q8_K_M: case Format::Q8_K_H: case Format::Q8_G: case Format::Q8_K_L_G: case Format::Q8_K_M_G: case Format::Q8_K_H_G: return RegFormat::Q8;
        case Format::Q12: case Format::Q12_K_L: case Format::Q12_K_M: case Format::Q12_K_H: case Format::Q12_G: case Format::Q12_K_L_G: case Format::Q12_K_M_G: case Format::Q12_K_H_G: return RegFormat::Q12;
        case Format::Q16: case Format::Q16_K_L: case Format::Q16_K_M: case Format::Q16_K_H: case Format::Q16_G: case Format::Q16_K_L_G: case Format::Q16_K_M_G: case Format::Q16_K_H_G: return RegFormat::Q16;
        case Format::Q24: case Format::Q24_K_L: case Format::Q24_K_M: case Format::Q24_K_H: case Format::Q24_G: case Format::Q24_K_L_G: case Format::Q24_K_M_G: case Format::Q24_K_H_G: return RegFormat::Q24;
        case Format::Q32: return RegFormat::Q32;
        case Format::Q_G_1_5: case Format::Q_G_2_5: case Format::Q_G_3_5: case Format::Q_G_4_5: case Format::Q_G_6_5: case Format::Q_G_8_5: case Format::Q_G_12_5: case Format::Q_G_16_5: case Format::Q_G_24_5: return RegFormat::Q1;
        case Format::MXQ_3_5: case Format::MXQ_3_5_G: return RegFormat::MXQ_3_5_G;
        case Format::MXQ_4_5: case Format::MXQ_4_5_G: return RegFormat::MXQ_4_5_G;
        case Format::MXQ_6_5: case Format::MXQ_6_5_G: return RegFormat::MXQ_6_5_G;
        case Format::MXQ_8_5: case Format::MXQ_8_5_G: return RegFormat::MXQ_8_5_G;
        case Format::MXQ_12_5: case Format::MXQ_12_5_G: return RegFormat::MXQ_12_5_G;
        case Format::MXQ_16_5: case Format::MXQ_16_5_G: return RegFormat::MXQ_16_5_G;
        case Format::MXQ_24_5: case Format::MXQ_24_5_G: return RegFormat::MXQ_24_5_G;
        default: return RegFormat::Q32;
    }
}
inline Format regformat_to_format(RegFormat rf) {
    switch(rf) {
        case RegFormat::Q1: return Format::Q1;
        case RegFormat::Q2: return Format::Q2;
        case RegFormat::Q3: return Format::Q3;
        case RegFormat::Q4: return Format::Q4;
        case RegFormat::Q6: return Format::Q6;
        case RegFormat::Q8: return Format::Q8;
        case RegFormat::Q12: return Format::Q12;
        case RegFormat::Q16: return Format::Q16;
        case RegFormat::Q24: return Format::Q24;
        case RegFormat::Q32: return Format::Q32;
        case RegFormat::Q1_G: return Format::Q1_G;
        case RegFormat::Q2_G: return Format::Q2_G;
        case RegFormat::Q3_G: return Format::Q3_G;
        case RegFormat::Q4_G: return Format::Q4_G;
        case RegFormat::Q6_G: return Format::Q6_G;
        case RegFormat::Q8_G: return Format::Q8_G;
        case RegFormat::Q12_G: return Format::Q12_G;
        case RegFormat::Q16_G: return Format::Q16_G;
        case RegFormat::Q24_G: return Format::Q24_G;
        case RegFormat::MXQ_3_5: return Format::MXQ_3_5;
        case RegFormat::MXQ_4_5: return Format::MXQ_4_5;
        case RegFormat::MXQ_6_5: return Format::MXQ_6_5;
        case RegFormat::MXQ_8_5: return Format::MXQ_8_5;
        case RegFormat::MXQ_12_5: return Format::MXQ_12_5;
        case RegFormat::MXQ_16_5: return Format::MXQ_16_5;
        case RegFormat::MXQ_24_5: return Format::MXQ_24_5;
        case RegFormat::MXQ_3_5_G: return Format::MXQ_3_5_G;
        case RegFormat::MXQ_4_5_G: return Format::MXQ_4_5_G;
        case RegFormat::MXQ_6_5_G: return Format::MXQ_6_5_G;
        case RegFormat::MXQ_8_5_G: return Format::MXQ_8_5_G;
        case RegFormat::MXQ_12_5_G: return Format::MXQ_12_5_G;
        case RegFormat::MXQ_16_5_G: return Format::MXQ_16_5_G;
        case RegFormat::MXQ_24_5_G: return Format::MXQ_24_5_G;
        default: return Format::Q32;
    }
}

} // namespace quant
