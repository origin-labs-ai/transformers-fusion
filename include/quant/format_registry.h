#pragma once

#include "quant/types.h"
#include <string>
#include <vector>

namespace quant {

enum class RegFormat : uint32_t {
    Q_1 = 0,
    Q_2,
    Q_3,
    Q_4,
    Q_6,
    Q_8,
    Q_12,
    Q_16,
    Q_24,
    Q_32,
    Q_G_1,
    Q_G_2,
    Q_G_3,
    Q_G_4,
    Q_G_6,
    Q_G_8,
    Q_G_12,
    Q_G_16,
    Q_G_24,
    Q_MX_3_5,
    Q_MX_4_5,
    Q_MX_6_5,
    Q_MX_8_5,
    Q_MX_12_5,
    Q_MX_16_5,
    Q_MX_24_5,
    Q_G_MX_3_5,
    Q_G_MX_4_5,
    Q_G_MX_6_5,
    Q_G_MX_8_5,
    Q_G_MX_12_5,
    Q_G_MX_16_5,
    Q_G_MX_24_5,

    // --- Legacy pre-Fusion aliases (deprecated, same IDs) ---
    Q1 = 0,  // legacy alias for Q_1
    Q2 = 1,  // legacy alias for Q_2
    Q3 = 2,  // legacy alias for Q_3
    Q4 = 3,  // legacy alias for Q_4
    Q6 = 4,  // legacy alias for Q_6
    Q8 = 5,  // legacy alias for Q_8
    Q12 = 6,  // legacy alias for Q_12
    Q16 = 7,  // legacy alias for Q_16
    Q24 = 8,  // legacy alias for Q_24
    Q32 = 9,  // legacy alias for Q_32
    QG1 = 10,  // legacy alias for Q_G_1
    QG2 = 11,  // legacy alias for Q_G_2
    QG3 = 12,  // legacy alias for Q_G_3
    QG4 = 13,  // legacy alias for Q_G_4
    QG6 = 14,  // legacy alias for Q_G_6
    QG8 = 15,  // legacy alias for Q_G_8
    QG12 = 16,  // legacy alias for Q_G_12
    QG16 = 17,  // legacy alias for Q_G_16
    QG24 = 18,  // legacy alias for Q_G_24
    QG_MX_3_5 = 26,  // legacy alias for Q_G_MX_3_5
    QG_MX_4_5 = 27,  // legacy alias for Q_G_MX_4_5
    QG_MX_6_5 = 28,  // legacy alias for Q_G_MX_6_5
    QG_MX_8_5 = 29,  // legacy alias for Q_G_MX_8_5
    QG_MX_12_5 = 30,  // legacy alias for Q_G_MX_12_5
    QG_MX_16_5 = 31,  // legacy alias for Q_G_MX_16_5
    QG_MX_24_5 = 32,  // legacy alias for Q_G_MX_24_5
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
    // Phase 13 grouped MXQ ladder (mixgap fix #1). Tier convention:
    //   Q_MX_*  (plain control, adaptive=false): tiers are plain Q* singles
    //             (e.g. Q_MX_3.5 = Q1/Q3/Q8/Q32); effective_bpw = wire BPW.
    //   QG_MX_* (grouped, adaptive=true): tiers are the grouped QG*
    //             counterparts (e.g. QG_MX_3.5 = QG1/QG3/QG8/Q32, Q32 top
    //             tier shared); each tier maps via the existing
    //             regformat_to_format() to its grouped wire Format, so the
    //             allocator encodes with the grouped codec. effective_bpw is
    //             the WIRE BPW from types.h format_bpw (QG_MX_3.5 = 3.78125
    //             at canonical n=256), never the nominal label.
    // Adaptive mixes (QG_MX): blocks are assigned to member formats by
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
        case Format::Q_1: case Format::Q_1_K_L: case Format::Q_1_K_M: case Format::Q_1_K_H: case Format::Q_G_1: case Format::Q_G_1_K_L: case Format::Q_G_1_K_M: case Format::Q_G_1_K_H: return RegFormat::Q_1;
        case Format::Q_2: case Format::Q_2_K_L: case Format::Q_2_K_M: case Format::Q_2_K_H: case Format::Q_G_2: case Format::Q_G_2_K_L: case Format::Q_G_2_K_M: case Format::Q_G_2_K_H: return RegFormat::Q_2;
        case Format::Q_3: case Format::Q_3_K_L: case Format::Q_3_K_M: case Format::Q_3_K_H: case Format::Q_G_3: case Format::Q_G_3_K_L: case Format::Q_G_3_K_M: case Format::Q_G_3_K_H: return RegFormat::Q_3;
        case Format::Q_4: case Format::Q_4_K_L: case Format::Q_4_K_M: case Format::Q_4_K_H: case Format::Q_G_4: case Format::Q_G_4_K_L: case Format::Q_G_4_K_M: case Format::Q_G_4_K_H: return RegFormat::Q_4;
        case Format::Q_6: case Format::Q_6_K_L: case Format::Q_6_K_M: case Format::Q_6_K_H: case Format::Q_G_6: case Format::Q_G_6_K_L: case Format::Q_G_6_K_M: case Format::Q_G_6_K_H: return RegFormat::Q_6;
        case Format::Q_8: case Format::Q_8_K_L: case Format::Q_8_K_M: case Format::Q_8_K_H: case Format::Q_G_8: case Format::Q_G_8_K_L: case Format::Q_G_8_K_M: case Format::Q_G_8_K_H: return RegFormat::Q_8;
        case Format::Q_12: case Format::Q_12_K_L: case Format::Q_12_K_M: case Format::Q_12_K_H: case Format::Q_G_12: case Format::Q_G_12_K_L: case Format::Q_G_12_K_M: case Format::Q_G_12_K_H: return RegFormat::Q_12;
        case Format::Q_16: case Format::Q_16_K_L: case Format::Q_16_K_M: case Format::Q_16_K_H: case Format::Q_G_16: case Format::Q_G_16_K_L: case Format::Q_G_16_K_M: case Format::Q_G_16_K_H: return RegFormat::Q_16;
        case Format::Q_24: case Format::Q_24_K_L: case Format::Q_24_K_M: case Format::Q_24_K_H: case Format::Q_G_24: case Format::Q_G_24_K_L: case Format::Q_G_24_K_M: case Format::Q_G_24_K_H: return RegFormat::Q_24;
        case Format::Q_32: return RegFormat::Q_32;
        case Format::Q_G_1_5: case Format::Q_G_2_5: case Format::Q_G_3_5: case Format::Q_G_4_5: case Format::Q_G_6_5: case Format::Q_G_8_5: case Format::Q_G_12_5: case Format::Q_G_16_5: case Format::Q_G_24_5: return RegFormat::Q_1;
        case Format::Q_MX_3_5: case Format::Q_G_MX_3_5: return RegFormat::Q_G_MX_3_5;
        case Format::Q_MX_4_5: case Format::Q_G_MX_4_5: return RegFormat::Q_G_MX_4_5;
        case Format::Q_MX_6_5: case Format::Q_G_MX_6_5: return RegFormat::Q_G_MX_6_5;
        case Format::Q_MX_8_5: case Format::Q_G_MX_8_5: return RegFormat::Q_G_MX_8_5;
        case Format::Q_MX_12_5: case Format::Q_G_MX_12_5: return RegFormat::Q_G_MX_12_5;
        case Format::Q_MX_16_5: case Format::Q_G_MX_16_5: return RegFormat::Q_G_MX_16_5;
        case Format::Q_MX_24_5: case Format::Q_G_MX_24_5: return RegFormat::Q_G_MX_24_5;
        default: return RegFormat::Q_32;
    }
}
inline Format regformat_to_format(RegFormat rf) {
    switch(rf) {
        case RegFormat::Q_1: return Format::Q_1;
        case RegFormat::Q_2: return Format::Q_2;
        case RegFormat::Q_3: return Format::Q_3;
        case RegFormat::Q_4: return Format::Q_4;
        case RegFormat::Q_6: return Format::Q_6;
        case RegFormat::Q_8: return Format::Q_8;
        case RegFormat::Q_12: return Format::Q_12;
        case RegFormat::Q_16: return Format::Q_16;
        case RegFormat::Q_24: return Format::Q_24;
        case RegFormat::Q_32: return Format::Q_32;
        case RegFormat::Q_G_1: return Format::Q_G_1;
        case RegFormat::Q_G_2: return Format::Q_G_2;
        case RegFormat::Q_G_3: return Format::Q_G_3;
        case RegFormat::Q_G_4: return Format::Q_G_4;
        case RegFormat::Q_G_6: return Format::Q_G_6;
        case RegFormat::Q_G_8: return Format::Q_G_8;
        case RegFormat::Q_G_12: return Format::Q_G_12;
        case RegFormat::Q_G_16: return Format::Q_G_16;
        case RegFormat::Q_G_24: return Format::Q_G_24;
        case RegFormat::Q_MX_3_5: return Format::Q_MX_3_5;
        case RegFormat::Q_MX_4_5: return Format::Q_MX_4_5;
        case RegFormat::Q_MX_6_5: return Format::Q_MX_6_5;
        case RegFormat::Q_MX_8_5: return Format::Q_MX_8_5;
        case RegFormat::Q_MX_12_5: return Format::Q_MX_12_5;
        case RegFormat::Q_MX_16_5: return Format::Q_MX_16_5;
        case RegFormat::Q_MX_24_5: return Format::Q_MX_24_5;
        case RegFormat::Q_G_MX_3_5: return Format::Q_G_MX_3_5;
        case RegFormat::Q_G_MX_4_5: return Format::Q_G_MX_4_5;
        case RegFormat::Q_G_MX_6_5: return Format::Q_G_MX_6_5;
        case RegFormat::Q_G_MX_8_5: return Format::Q_G_MX_8_5;
        case RegFormat::Q_G_MX_12_5: return Format::Q_G_MX_12_5;
        case RegFormat::Q_G_MX_16_5: return Format::Q_G_MX_16_5;
        case RegFormat::Q_G_MX_24_5: return Format::Q_G_MX_24_5;
        default: return Format::Q_32;
    }
}

// Rule: names are nominal family labels, format_bpw() is wire truth.
// RegFormat twins delegate to the Format accessors via regformat_to_format.
inline float regformat_nominal_bpw(RegFormat rf) {
    return format_nominal_bpw(regformat_to_format(rf));
}
inline float regformat_wire_bpw(RegFormat rf) {
    return format_wire_bpw(regformat_to_format(rf));
}

} // namespace quant
