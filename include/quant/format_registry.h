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
    Q1_GRP,
    Q2_GRP,
    Q3_GRP,
    Q4_GRP,
    Q6_GRP,
    Q8_GRP,
    Q12_GRP,
    Q16_GRP,
    Q24_GRP,
    MXQ_3_5_GRP,
    MXQ_4_5_GRP,
    MXQ_6_5_GRP,
    MXQ_8_5_GRP,
    MXQ_12_5_GRP,
    MXQ_16_5_GRP,
    MXQ_24_5_GRP
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
    auto v = static_cast<uint32_t>(f);
    if (v < 38) return static_cast<RegFormat>(v);
    return RegFormat::Q32;
}

inline Format regformat_to_format(RegFormat rf) {
    auto v = static_cast<uint8_t>(rf);
    if (v < 38) return static_cast<Format>(v);
    return Format::Q32;
}

} // namespace quant
