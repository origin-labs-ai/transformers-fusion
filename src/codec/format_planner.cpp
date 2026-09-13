#include "quant/format_planner.h"
#include "quant/format_registry.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>

namespace quant {

FormatPlanner::FormatPlanner(float t) : target_bpw_(t) {}

void FormatPlanner::score_importance(const Tensor& weights,
                                      const Tensor& calibration_activations,
                                      int block_size, ImportanceMetric metric) {
    if (metric == ImportanceMetric::FISHER_DIAG) {
        score_importance_fisher(weights, calibration_activations, block_size);
        return;
    }
    const float* wd = (const float*)weights.data();
    const float* ad = (const float*)calibration_activations.data();
    int64_t n = weights.numel();
    int64_t num_blocks = n / block_size;
    if (n % block_size != 0) num_blocks++;
    importance_scores_.resize((size_t)num_blocks);

    for (int64_t b = 0; b < num_blocks; b++) {
        double score = 0;
        int64_t start = b * block_size;
        int64_t end = (std::min)(start + block_size, n);
        for (int64_t i = start; i < end; i++) {
            float w = wd[i];
            float a = ad ? std::fabs(ad[i]) : 1.0f;
            score += (double)std::fabs(w) * (double)a;
        }
        importance_scores_[b] = (float)(score / (double)(end - start));
    }
}

void FormatPlanner::score_importance_fisher(const Tensor& weights,
                                             const Tensor& gradients,
                                             int block_size) {
    const float* wd = (const float*)weights.data();
    const float* gd = (const float*)gradients.data();
    int64_t n = weights.numel();
    int64_t num_blocks = n / block_size;
    if (n % block_size != 0) num_blocks++;
    importance_scores_.resize((size_t)num_blocks);

    for (int64_t b = 0; b < num_blocks; b++) {
        double score = 0;
        int64_t start = b * block_size;
        int64_t end = (std::min)(start + block_size, n);
        for (int64_t i = start; i < end; i++) {
            float g = gd[i];
            score += (double)(g * g);
        }
        score = std::sqrt(score / (double)(end - start));
        score *= (double)std::fabs(wd[start]);
        importance_scores_[b] = (float)score;
    }
}

// Extended format mix allocation supporting all 10 base Q formats.
// Allocates blocks across Q1(1.0), Q2(2.0), Q3(3.0), Q4(4.0), Q6(6.0),
// Q8(8.0), Q12(12.0), Q16(16.0), Q24(24.0), Q32(32.0) using linear
// interpolation between adjacent BPW levels for any target BPW.
struct FormatMixResult {
    int counts[10] = {}; // Q1..Q32 in order of the base enum
};

static FormatMixResult compute_format_mix_extended(int num_blocks, float target_bpw) {
    // BPW ladder: Q1=1.0, Q2=2.0, Q3=3.0, Q4=4.0, Q6=6.0,
    //             Q8=8.0, Q12=12.0, Q16=16.0, Q24=24.0, Q32=32.0
    constexpr float bpw[10] = {1.0f,2.0f,3.0f,4.0f,6.0f,8.0f,12.0f,16.0f,24.0f,32.0f};
    float fractions[10] = {};

    // Clamp to valid range
    if (target_bpw >= 32.0f) {
        fractions[9] = 1.0f;
    } else if (target_bpw <= 1.0f) {
        fractions[0] = 1.0f;
    } else {
        // Find the two adjacent formats that bracket the target BPW
        int upper = 1;
        while (upper < 10 && bpw[upper] < target_bpw) upper++;
        int lower = upper - 1;
        float range = bpw[upper] - bpw[lower];
        float ratio_upper = (target_bpw - bpw[lower]) / range;
        float ratio_lower = 1.0f - ratio_upper;
        fractions[lower] = ratio_lower;
        fractions[upper] = ratio_upper;
    }

    FormatMixResult result;
    int used = 0;
    for (int i = 0; i < 9; i++) {
        result.counts[i] = (int)std::round(fractions[i] * (double)num_blocks);
        used += result.counts[i];
    }
    result.counts[9] = num_blocks - used;
    if (result.counts[9] < 0) result.counts[9] = 0;

    return result;
}

void FormatPlanner::compute_format_mix(int num_blocks, float target_bpw,
                                        int& quant32, int& quant16, int& quant8,
                                        int& quant4, int& quant2, int& quant, int& quant1) {
    // Legacy 7-format interface — delegates to extended and maps back
    auto mix = compute_format_mix_extended(num_blocks, target_bpw);
    quant1  = mix.counts[0]; // Q1
    quant2  = mix.counts[1]; // Q2
    // mix.counts[2] = Q3, mix.counts[3] = Q4 — fold Q3 into quant2 for legacy
    quant2 += mix.counts[2];
    quant4  = mix.counts[3]; // Q4
    // mix.counts[4] = Q6 — fold into quant4 for legacy
    quant4 += mix.counts[4];
    quant8  = mix.counts[5]; // Q8
    // mix.counts[6] = Q12 — fold into quant8 for legacy
    quant8 += mix.counts[6];
    quant16 = mix.counts[7]; // Q16
    quant32 = mix.counts[8] + mix.counts[9]; // Q24+Q32
    quant   = 0;
}

static RegFormat bpw_to_reg_format(float bpw) {
    if (bpw >= 31.99f) return RegFormat::Q32;
    if (bpw >= 23.99f) return RegFormat::Q32; // Q24 → maps to Q32 in reg
    if (bpw >= 15.99f) return RegFormat::Q16;
    if (bpw >= 11.99f) return RegFormat::Q16; // Q12 → maps to Q16 in reg
    if (bpw >= 7.99f)  return RegFormat::Q8;
    if (bpw >= 5.99f)  return RegFormat::Q8;  // Q6 → maps to Q8 in reg
    if (bpw >= 3.99f)  return RegFormat::Q4;
    if (bpw >= 2.99f)  return RegFormat::Q4;  // Q3 → maps to Q4 in reg
    if (bpw >= 1.99f)  return RegFormat::Q2;
    return RegFormat::Q1;
}

FormatPlan FormatPlanner::allocate(int num_weight_blocks, int weights_per_block) {
    FormatPlan plan;
    plan.target_bpw = target_bpw_;
    plan.blocks.resize(num_weight_blocks);
    plan.num_quant32_blocks = 0;
    plan.num_quant16_blocks = 0;
    plan.num_quant8_blocks = 0;
    plan.num_quant4_blocks = 0;
    plan.num_quant2_blocks = 0;
    plan.num_quant_blocks = 0;
    plan.num_quant1_blocks = 0;
    plan.uses_mix = false;

    plan.selected_single = FormatRegistry::get_single_format(target_bpw_);

    const auto& twi_mixes = FormatRegistry::get_all_twi_mixes();
    float best_two_diff = 1e9f;
    for (const auto& m : twi_mixes) {
        float diff = std::fabs(m.effective_bpw - target_bpw_);
        if (diff < best_two_diff) {
            best_two_diff = diff;
            plan.selected_mix = m;
        }
    }

    const auto& four_mixes = FormatRegistry::get_all_four_mixes();
    float best_four_diff = 1e9f;
    MixDescriptor best_four;
    for (const auto& m : four_mixes) {
        float diff = std::fabs(m.effective_bpw - target_bpw_);
        if (diff < best_four_diff) {
            best_four_diff = diff;
            best_four = m;
        }
    }

    float single_diff = std::fabs(plan.selected_single.bpw - target_bpw_);

    // Generic mix allocator — handles BOTH the two-tier (twi) and four-tier
    // (four) blends. Each mix tier's ratio spreads across the (importance-
    // ordered) blocks, and the tier's RegFormat is mapped to its real Format
    // through the registry helper regformat_to_format(). This covers ARBITRARY
    // tier sets, including GRP formats (QG1..QG8, Q0_G) and
    // QUANT adaptive mixes, so the plan's achieved BPW is honest (it no longer
    // collapses a 4/2-tier mix like QUANT_QUAD_MIX_Q0/SARK_MIX_Q0 to Q1).
    const bool use_twi = (best_two_diff < single_diff &&
                          best_two_diff < best_four_diff);
    const bool use_four = (best_four_diff < single_diff &&
                           best_four_diff < best_two_diff);

    const auto emit_mix_plan = [&](const MixDescriptor& mix) {
        const int nt = std::max(1, std::min(mix.num_tiers, 4));
        const RegFormat tiers[4] = { mix.tier1_fmt, mix.tier2_fmt,
                                     mix.tier3_fmt, mix.tier4_fmt };
        const float ratios[4] = { mix.tier1_ratio, mix.tier2_ratio,
                                  mix.tier3_ratio, mix.tier4_ratio };

        // Per-tier block counts driven by the mix ratios. The LAST tier
        // absorbs the remainder so the blend always covers every block
        // (identical semantics for 2- and 4-tier descriptors).
        std::vector<int> counts(static_cast<size_t>(nt), 0);
        int used = 0;
        for (int t = 0; t < nt - 1; t++) {
            counts[static_cast<size_t>(t)] =
                (int)std::round(ratios[t] * (double)num_weight_blocks);
            used += counts[static_cast<size_t>(t)];
        }
        counts[static_cast<size_t>(nt) - 1] =
            (num_weight_blocks - used > 0) ? num_weight_blocks - used : 0;

        std::vector<int> indices(num_weight_blocks);
        for (int i = 0; i < num_weight_blocks; i++) indices[i] = i;
        if (!importance_scores_.empty()) {
            std::sort(indices.begin(), indices.end(), [this](int a, int b) {
                float sa = a < (int)importance_scores_.size() ? importance_scores_[a] : 0;
                float sb = b < (int)importance_scores_.size() ? importance_scores_[b] : 0;
                return sa > sb;
            });
        }

        for (int i = 0; i < num_weight_blocks; i++) {
            const int idx = indices[i];
            plan.blocks[idx].id = (uint32_t)idx;
            plan.blocks[idx].weight_index = (uint32_t)(idx * weights_per_block);
            plan.blocks[idx].num_weights = (uint32_t)weights_per_block;

            // Find which tier owns this importance-sorted slot.
            int tt = 0, cum = 0;
            for (int t = 0; t < nt; t++) {
                if (i < cum + counts[static_cast<size_t>(t)]) { tt = t; break; }
                cum += counts[static_cast<size_t>(t)];
            }

            plan.blocks[idx].registry_format = tiers[tt];
            plan.blocks[idx].assigned_format = regformat_to_format(tiers[tt]);
            plan.blocks[idx].importance_score =
                (idx < (int)importance_scores_.size()) ? importance_scores_[idx] : 0.0f;
        }
    };

    const auto count_plan = [&]() {
        plan.num_quant32_blocks = plan.num_quant16_blocks = 0;
        plan.num_quant8_blocks = plan.num_quant4_blocks = 0;
        plan.num_quant2_blocks = plan.num_quant_blocks = plan.num_quant1_blocks = 0;
        for (const auto& b : plan.blocks) {
            float bpw = format_bpw(b.assigned_format);
            if (bpw >= 24.0f) plan.num_quant32_blocks++;
            else if (bpw >= 12.0f) plan.num_quant16_blocks++;
            else if (bpw >= 6.0f) plan.num_quant8_blocks++;
            else if (bpw >= 3.0f) plan.num_quant4_blocks++;
            else if (bpw >= 2.0f) plan.num_quant2_blocks++;
            else if (bpw >= 1.5f) plan.num_quant_blocks++;
            else plan.num_quant1_blocks++;
        }
    };

    if (use_twi || use_four) {
        plan.uses_mix = true;
        if (use_four) plan.selected_mix = best_four;
        emit_mix_plan(plan.selected_mix);
        count_plan();
    } else {
        std::vector<int> indices(num_weight_blocks);
        for (int i = 0; i < num_weight_blocks; i++) indices[i] = i;

        if (!importance_scores_.empty()) {
            std::sort(indices.begin(), indices.end(), [this](int a, int b) {
                float sa = a < (int)importance_scores_.size() ? importance_scores_[a] : 0;
                float sb = b < (int)importance_scores_.size() ? importance_scores_[b] : 0;
                return sa > sb;
            });
        }

        int o32 = 0, o16 = 0, o8 = 0, o4 = 0, o2 = 0, sp = 0, o1 = 0;
        compute_format_mix(num_weight_blocks, target_bpw_,
                           o32, o16, o8, o4, o2, sp, o1);

        plan.num_quant32_blocks = o32;
        plan.num_quant16_blocks = o16;
        plan.num_quant8_blocks = o8;
        plan.num_quant4_blocks = o4;
        plan.num_quant2_blocks = o2;
        plan.num_quant_blocks = sp;
        plan.num_quant1_blocks = o1;

        // Extended allocation using all 10 base formats
        auto mix = compute_format_mix_extended(num_weight_blocks, target_bpw_);
        // Format mapping array: Q1, Q2, Q3, Q4, Q6, Q8, Q12, Q16, Q24, Q32
        constexpr Format fmt_map[10] = {
            Format::Q1, Format::Q2, Format::Q3, Format::Q4, Format::Q6,
            Format::Q8, Format::Q12, Format::Q16, Format::Q24, Format::Q32
        };

        int cursor = 0;
        for (int tier = 9; tier >= 0; tier--) { // Most salient → Q32 first
            for (int j = 0; j < mix.counts[tier] && cursor < num_weight_blocks; j++, cursor++) {
                int idx = indices[cursor];
                plan.blocks[idx].assigned_format = fmt_map[tier];
                plan.blocks[idx].registry_format = bpw_to_reg_format(format_bpw(fmt_map[tier]));
                plan.blocks[idx].importance_score =
                    (idx < (int)importance_scores_.size()) ? importance_scores_[idx] : 0.0f;
            }
        }
        // Fill any remaining (shouldn't happen, but safety)
        while (cursor < num_weight_blocks) {
            int idx = indices[cursor];
            plan.blocks[idx].assigned_format = Format::Q1;
            plan.blocks[idx].registry_format = RegFormat::Q1;
            plan.blocks[idx].importance_score =
                (idx < (int)importance_scores_.size()) ? importance_scores_[idx] : 0.0f;
            cursor++;
        }
    }

    plan.achieved_bpw = estimate_bpw(plan);
    return plan;
}

FormatPlan FormatPlanner::plan_for_model(int64_t num_weights) {
    return allocate((int)(num_weights / 256), 256);
}

const std::vector<float>& FormatPlanner::importance_scores() const {
    return importance_scores_;
}

float FormatPlanner::estimate_bpw(const FormatPlan& plan) {
    if (plan.blocks.empty()) return 0;
    float total = 0;
    for (const auto& b : plan.blocks) {
        total += format_bpw(b.assigned_format);
    }
    return total / (float)plan.blocks.size();
}

FormatPlan FormatPlanner::plan_for_target(float target_bpw, int num_blocks,
                                           int weights_per_block) {
    FormatPlanner planner(target_bpw);
    return planner.allocate(num_blocks, weights_per_block);
}

} // namespace quant
