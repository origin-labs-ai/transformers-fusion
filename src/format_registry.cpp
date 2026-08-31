#include "quant/format_registry.h"
#include "quant/codebook.h"
#include "quant/math.h"
#include "quant/block_codec.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <limits>
#include <array>
#include <random>
#include <queue>
#include <utility>

namespace quant {

static constexpr int WIRE_BLOCK = 256;  // canonical wire block size (block_codec.h)

static std::vector<FormatDescriptor> build_singles() {
    std::vector<FormatDescriptor> v;
    // Base formats
    v.push_back({"Q1",      RegFormat::Q1,      1.0f,  1,    false, false, 1,    1.0f,  0.0f,  "FP32 block mean + 1-bit quantization"});
    v.push_back({"Q2",      RegFormat::Q2,      2.0f,  4,    false, false, 1,    4.0f,  0.0f,  "2-bit lattice + 4 FP32 centroids"});
    v.push_back({"Q3",      RegFormat::Q3,      3.0f,  8,    false, false, 1,    8.0f,  0.0f,  "3-bit lattice + 8 FP32 centroids"});
    v.push_back({"Q4",      RegFormat::Q4,      4.0f,  16,   false, false, 1,    16.0f, 0.0f,  "4-bit lattice + 16 FP32 centroids"});
    v.push_back({"Q6",      RegFormat::Q6,      6.0f,  64,   false, false, 1,    64.0f, 0.0f,  "6-bit lattice + 64 FP32 centroids"});
    v.push_back({"Q8",      RegFormat::Q8,      8.0f,  256,  false, false, 1,    256.0f,0.0f,  "8-bit lattice + 256 FP32 centroids"});
    v.push_back({"Q12",     RegFormat::Q12,     12.0f, 4096, false, false, 1,    4096.0f,0.0f, "12-bit lattice + 4096 FP16 centroids"});
    v.push_back({"Q16",     RegFormat::Q16,     16.0f, 0,    false, false, 1,    0.0f,  0.0f,  "Per-block adaptive 16-bit"});
    v.push_back({"Q24",     RegFormat::Q24,     24.0f, 0,    false, false, 1,    0.0f,  0.0f,  "Per-block FP24 (16b mantissa + 8b exp)"});
    v.push_back({"Q32",     RegFormat::Q32,     32.0f, 0,    true,  false, 1,    0.0f,  0.0f,  "FP32 identity (lossless)"});

    // G variants — TRUE wire BPW (matches types.h format_bpw, audit option-c)
    v.push_back({"Q1_G",  RegFormat::Q1_G,  1.0f,   1,    false, true,  256,  0.0f,  0.0f,  "1-bit + block FP16 scale"});
    v.push_back({"Q2_G",  RegFormat::Q2_G,  2.625f, 4,    false, true,  16,   16.0f, 0.0f,  "2-bit affine per-16 (4b scale+min)"});
    v.push_back({"Q3_G",  RegFormat::Q3_G,  3.5f,   8,    false, true,  32,   16.0f, 0.0f,  "3-bit affine per-32 (6b scale+min)"});
    v.push_back({"Q4_G",  RegFormat::Q4_G,  4.5f,   16,   false, true,  32,   16.0f, 0.0f,  "4-bit affine per-32 (6b scale+min)"});
    v.push_back({"Q6_G",  RegFormat::Q6_G,  6.5625f,64,   false, true,  16,   16.0f, 0.0f,  "6-bit Q6_K scheme, per-16 int8 scales"});
    v.push_back({"Q8_G",  RegFormat::Q8_G,  8.5f,   256,  false, true,  32,   16.0f, 0.0f,  "8-bit compound: fp16 scale per-32"});
    v.push_back({"Q12_G", RegFormat::Q12_G, 12.5f,  4096, false, true,  32,   16.0f, 0.0f,  "12-bit compound: fp16 scale per-32"});
    v.push_back({"Q16_G", RegFormat::Q16_G, 16.5f,  0,    false, true,  32,   16.0f, 0.0f,  "16-bit adaptive + mean-residual tail"});
    v.push_back({"Q24_G", RegFormat::Q24_G, 24.5f,  0,    false, true,  32,   8.0f,  0.0f,  "FP24 + mean-residual tail"});

    const int64_t n_mse = 16384;
    std::vector<float> g_data((size_t)n_mse), u_data((size_t)n_mse), l_data((size_t)n_mse);
    {
        std::mt19937 rng(20260803);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::uniform_real_distribution<float> u(-2.0f, 2.0f);
        std::exponential_distribution<float> e(1.0f);
        for (int64_t i = 0; i < n_mse; ++i) {
            g_data[(size_t)i] = g(rng);
            u_data[(size_t)i] = u(rng);
            l_data[(size_t)i] = e(rng) - e(rng);
        }
    }
    const float* dists[3] = { g_data.data(), u_data.data(), l_data.data() };
    for (auto& s : v) {
        if (s.lossless) { s.est_mse = 0.0f; continue; }
        float worst = 0.0f;
        for (const float* d : dists) {
            QuantResult qr = FormatRegistry::quantize(d, n_mse, s);
            if (!qr.success) { worst = 1.0f; break; }
            std::vector<float> dq((size_t)n_mse);
            FormatRegistry::dequantize(qr, dq.data(), n_mse);
            worst = std::max(worst, FormatRegistry::measure_mse(d, dq.data(), n_mse));
        }
        s.est_mse = worst;
    }
    return v;
}

static std::vector<MixDescriptor> build_two_mixes() {
    std::vector<MixDescriptor> v;
    return v;
}

static std::vector<MixDescriptor> build_four_mixes() {
    std::vector<MixDescriptor> v;
    v.push_back({"MXQ_3.5",          RegFormat::MXQ_3_5,          4, RegFormat::Q1,      0.70f, RegFormat::Q3,      0.20f, RegFormat::Q8,      0.08f, RegFormat::Q32,     0.02f, 3.50f,  false});
    v.push_back({"MXQ_4.5",          RegFormat::MXQ_4_5,          4, RegFormat::Q2,      0.60f, RegFormat::Q4,      0.25f, RegFormat::Q12,     0.12f, RegFormat::Q32,     0.03f, 4.50f,  false});
    v.push_back({"MXQ_6.5",          RegFormat::MXQ_6_5,          4, RegFormat::Q3,      0.50f, RegFormat::Q6,      0.30f, RegFormat::Q16,     0.15f, RegFormat::Q32,     0.05f, 6.50f,  false});
    v.push_back({"MXQ_8.5",          RegFormat::MXQ_8_5,          4, RegFormat::Q4,      0.45f, RegFormat::Q8,      0.35f, RegFormat::Q16,     0.15f, RegFormat::Q32,     0.05f, 8.50f,  false});
    v.push_back({"MXQ_12.5",         RegFormat::MXQ_12_5,         4, RegFormat::Q6,      0.40f, RegFormat::Q12,     0.35f, RegFormat::Q24,     0.20f, RegFormat::Q32,     0.05f, 12.50f, false});
    v.push_back({"MXQ_16.5",         RegFormat::MXQ_16_5,         4, RegFormat::Q8,      0.35f, RegFormat::Q16,     0.40f, RegFormat::Q24,     0.20f, RegFormat::Q32,     0.05f, 16.50f, false});
    v.push_back({"MXQ_24.5",         RegFormat::MXQ_24_5,         4, RegFormat::Q12,     0.25f, RegFormat::Q16,     0.30f, RegFormat::Q24,     0.35f, RegFormat::Q32,     0.10f, 24.50f, false});
    v.push_back({"MXQ_3.5_G",      RegFormat::MXQ_3_5_G,      4, RegFormat::Q1,      0.70f, RegFormat::Q3,      0.20f, RegFormat::Q8,      0.08f, RegFormat::Q32,     0.02f, 3.50f,  true});
    v.push_back({"MXQ_4.5_G",      RegFormat::MXQ_4_5_G,      4, RegFormat::Q2,      0.60f, RegFormat::Q4,      0.25f, RegFormat::Q12,     0.12f, RegFormat::Q32,     0.03f, 4.50f,  true});
    v.push_back({"MXQ_6.5_G",      RegFormat::MXQ_6_5_G,      4, RegFormat::Q3,      0.50f, RegFormat::Q6,      0.30f, RegFormat::Q16,     0.15f, RegFormat::Q32,     0.05f, 6.50f,  true});
    v.push_back({"MXQ_8.5_G",      RegFormat::MXQ_8_5_G,      4, RegFormat::Q4,      0.45f, RegFormat::Q8,      0.35f, RegFormat::Q16,     0.15f, RegFormat::Q32,     0.05f, 8.50f,  true});
    v.push_back({"MXQ_12.5_G",     RegFormat::MXQ_12_5_G,     4, RegFormat::Q6,      0.40f, RegFormat::Q12,     0.35f, RegFormat::Q24,     0.20f, RegFormat::Q32,     0.05f, 12.50f, true});
    v.push_back({"MXQ_16.5_G",     RegFormat::MXQ_16_5_G,     4, RegFormat::Q8,      0.35f, RegFormat::Q16,     0.40f, RegFormat::Q24,     0.20f, RegFormat::Q32,     0.05f, 16.50f, true});
    v.push_back({"MXQ_24.5_G",     RegFormat::MXQ_24_5_G,     4, RegFormat::Q12,     0.25f, RegFormat::Q16,     0.30f, RegFormat::Q24,     0.35f, RegFormat::Q32,     0.10f, 24.50f, true});
    return v;
}

const std::vector<FormatDescriptor>& FormatRegistry::get_all_singles() {
    static const std::vector<FormatDescriptor> s = build_singles();
    return s;
}

const std::vector<MixDescriptor>& FormatRegistry::get_all_twi_mixes() {
    static const std::vector<MixDescriptor> s = build_two_mixes();
    return s;
}

const std::vector<MixDescriptor>& FormatRegistry::get_all_four_mixes() {
    static const std::vector<MixDescriptor> s = build_four_mixes();
    return s;
}

FormatDescriptor FormatRegistry::get_single_format(float target_bpw) {
    return find_closest_single(target_bpw);
}

MixDescriptor FormatRegistry::get_twi_mix(float target_bpw) {
    const auto& mixes = get_all_twi_mixes();
    if (mixes.empty()) return {};
    MixDescriptor best = mixes[0];
    float best_dist = std::fabs(mixes[0].effective_bpw - target_bpw);
    for (size_t i = 1; i < mixes.size(); i++) {
        float d = std::fabs(mixes[i].effective_bpw - target_bpw);
        if (d < best_dist) { best_dist = d; best = mixes[i]; }
    }
    return best;
}

MixDescriptor FormatRegistry::get_four_mix(float target_bpw) {
    const auto& mixes = get_all_four_mixes();
    if (mixes.empty()) return {};
    MixDescriptor best = mixes[0];
    float best_dist = std::fabs(mixes[0].effective_bpw - target_bpw);
    for (size_t i = 1; i < mixes.size(); i++) {
        float d = std::fabs(mixes[i].effective_bpw - target_bpw);
        if (d < best_dist) { best_dist = d; best = mixes[i]; }
    }
    return best;
}

FormatDescriptor FormatRegistry::find_closest_single(float target_bpw) {
    const auto& singles = get_all_singles();
    if (singles.empty()) return {};
    FormatDescriptor best = singles[0];
    float best_dist = std::fabs(singles[0].bpw - target_bpw);
    for (size_t i = 1; i < singles.size(); i++) {
        float d = std::fabs(singles[i].bpw - target_bpw);
        if (d < best_dist) { best_dist = d; best = singles[i]; }
    }
    return best;
}

QuantResult FormatRegistry::quantize(const float* data, int64_t n,
                                     const FormatDescriptor& fmt) {
    QuantResult qr;
    qr.format = fmt;
    qr.num_elements = n;
    qr.block_size = WIRE_BLOCK;
    qr.success = false;
    qr.global_scale = 0.0f;

    if (!data || n <= 0) return qr;

    // Canonical wire codec — the single source of truth for every format ID.
    // Quantizing here produces EXACTLY the bytes the .quant writer stores, so
    // eval/selection quality is identical to on-disk quality. The honest
    // serialized size is therefore indices.size() + codebook.size().
    const Format wire = regformat_to_format(fmt.id);
    const int64_t nb = (n + WIRE_BLOCK - 1) / WIRE_BLOCK;

    // Q32: keep the legacy in-memory FP32 copy for callers that read it.
    if (fmt.id == RegFormat::Q32) {
        qr.codebook_fp32.resize(static_cast<size_t>(n));
        std::memcpy(qr.codebook_fp32.data(), data, static_cast<size_t>(n) * sizeof(float));
    }

    qr.block_idx_bytes.reserve(static_cast<size_t>(nb));
    qr.block_cb_bytes.reserve(static_cast<size_t>(nb));
    for (int64_t b = 0; b < nb; b++) {
        const int64_t start = b * WIRE_BLOCK;
        const int64_t blen = (start + WIRE_BLOCK <= n) ? WIRE_BLOCK : (n - start);
        std::vector<uint8_t> idx, cb;
        if (!quantize_block_all(wire, data + start, static_cast<int>(blen), idx, cb))
            return qr;  // unhandled format -> success stays false
        qr.block_idx_bytes.push_back(static_cast<uint32_t>(idx.size()));
        qr.block_cb_bytes.push_back(static_cast<uint32_t>(cb.size()));
        qr.indices.insert(qr.indices.end(), idx.begin(), idx.end());
        qr.codebook.insert(qr.codebook.end(), cb.begin(), cb.end());
    }
    qr.success = true;
    return qr;
}

QuantResult FormatRegistry::quantize_block(const float* block, size_t block_size,
                                           const FormatDescriptor& fmt) {
    return quantize(block, static_cast<int64_t>(block_size), fmt);
}

// All format-specific quantize helpers delegate to the canonical quantize(),
// guaranteeing byte-identical results to the .quant writer — a single encoding
// per format ID (registry path == on-disk path).
QuantResult FormatRegistry::quantize_q1(const float* data, int64_t n) {
    return quantize(data, n, parse_format_name("Q1"));
}

QuantResult FormatRegistry::quantize_q2(const float* data, int64_t n) {
    return quantize(data, n, parse_format_name("Q2"));
}

QuantResult FormatRegistry::quantize_q4(const float* data, int64_t n) {
    return quantize(data, n, parse_format_name("Q4"));
}

QuantResult FormatRegistry::quantize_q8(const float* data, int64_t n) {
    return quantize(data, n, parse_format_name("Q8"));
}

QuantResult FormatRegistry::quantize_q_twi_mix_1_5(const float* data, int64_t n, int) {
    return quantize(data, n, parse_format_name("Q_G_1_5"));
}

QuantResult FormatRegistry::quantize_q_sparse(const float* data, int64_t n) {
    return quantize(data, n, parse_format_name("Q2")); // Map legacy Q1 to Q2
}

void FormatRegistry::dequantize(const QuantResult& qr, float* output, int64_t n) {
    if (!output || n <= 0 || !qr.success) return;

    // Legacy Q32 in-memory copy (saved by older STE adapter stores) takes
    // precedence; otherwise the wire payload (raw FP32 bytes in indices) is
    // decoded, matching the canonical .quant layout.
    if (qr.format.id == RegFormat::Q32) {
        if (qr.codebook_fp32.size() >= static_cast<size_t>(n)) {
            std::memcpy(output, qr.codebook_fp32.data(), static_cast<size_t>(n) * sizeof(float));
            return;
        }
        if (qr.indices.size() >= static_cast<size_t>(n) * 4) {
            std::memcpy(output, qr.indices.data(), static_cast<size_t>(n) * 4);
            return;
        }
        std::fill(output, output + n, 0.0f);
        return;
    }

    const Format wire = regformat_to_format(qr.format.id);

    // Multi-block wire path produced by quantize(): slice each block's
    // payload using the recorded per-block byte lengths.
    const size_t nblocks = qr.block_idx_bytes.size();
    if (nblocks > 0 && nblocks == qr.block_cb_bytes.size()) {
        size_t idx_off = 0, cb_off = 0;
        int64_t done = 0;
        for (size_t b = 0; b < nblocks && done < n; b++) {
            const int64_t blen = std::min<int64_t>(WIRE_BLOCK, n - done);
            dequantize_block_all(wire,
                                 qr.indices.data() + idx_off, qr.block_idx_bytes[b],
                                 qr.codebook.data() + cb_off, qr.block_cb_bytes[b],
                                 static_cast<uint32_t>(blen), output + done);
            idx_off += qr.block_idx_bytes[b];
            cb_off += qr.block_cb_bytes[b];
            done += blen;
        }
        return;
    }

    // Single concatenated payload (older STE adapter stores persist one
    // block-less buffer): decode it as a single block of n weights.
    if (qr.indices.empty() && qr.codebook.empty()) {
        std::fill(output, output + n, 0.0f);
        return;
    }
    dequantize_block_all(wire, qr.indices.data(), qr.indices.size(),
                         qr.codebook.data(), qr.codebook.size(),
                         static_cast<uint32_t>(n), output);
}

float FormatRegistry::measure_mse(const float* original, const float* dequantized, int64_t n) {
    if (!original || !dequantized || n <= 0) return 0.0f;
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double diff = static_cast<double>(original[i]) - static_cast<double>(dequantized[i]);
        sum += diff * diff;
    }
    return static_cast<float>(sum / static_cast<double>(n));
}

float FormatRegistry::evaluate_format_quality(const float* data, int64_t n,
                                              const FormatDescriptor& fmt) {
    if (!data || n <= 0) return 0.0f;
    QuantResult qr = quantize(data, n, fmt);
    if (!qr.success) return 0.0f;
    std::vector<float> dequant(static_cast<size_t>(n));
    dequantize(qr, dequant.data(), n);
    return measure_mse(data, dequant.data(), n);
}

float FormatRegistry::evaluate_format_quality_weighted(const float* data, const float* gradients,
                                                       int64_t n, const FormatDescriptor& fmt,
                                                       float fisher_weight) {
    if (!data || !gradients || n <= 0) return 0.0f;

    QuantResult qr = quantize(data, n, fmt);
    if (!qr.success) return 0.0f;
    std::vector<float> dequant(static_cast<size_t>(n));
    dequantize(qr, dequant.data(), n);

    double sum = 0.0, weight_sum = 0.0;
    float max_grad = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float g = std::fabs(gradients[i]);
        if (g > max_grad) max_grad = g;
    }
    if (max_grad == 0.0f) max_grad = 1.0f;
    for (int64_t i = 0; i < n; i++) {
        float importance = (1.0f - fisher_weight) + fisher_weight * (std::fabs(gradients[i]) / max_grad);
        double diff = static_cast<double>(data[i]) - static_cast<double>(dequant[static_cast<size_t>(i)]);
        sum += importance * diff * diff;
        weight_sum += importance;
    }
    return static_cast<float>(sum / weight_sum);
}

FormatDescriptor FormatRegistry::select_best_format(float target_bpw, const float* data, int64_t n) {
    const auto& singles = get_all_singles();
    FormatDescriptor best;
    float best_mse = 1e30f;
    for (const auto& fmt : singles) {
        // Selection contract: never exceed the target BPW. Formats that cost
        // more are not eligible regardless of their measured quality.
        if (fmt.bpw > target_bpw + 1e-6f) continue;
        float mse = evaluate_format_quality(data, n, fmt);
        if (mse < best_mse) { best_mse = mse; best = fmt; }
    }
    if (!best.name.empty()) return best;
    return find_closest_single(target_bpw);
}

MixDescriptor FormatRegistry::select_best_mix(float target_bpw, const float* data, int64_t n) {
    // Data-driven: allocate each candidate mix under its HARD claimed-BPW
    // budget (adaptive allocator + canonical block codec), measure the actual
    // reconstruction MSE on this tensor, and return the mix with the lowest
    // measured error. Falls back to nearest-claimed-BPW when no mix is within
    // range or no data is provided.
    const float best_mse = 1e30f;
    MixDescriptor best;
    float best_m = best_mse;
    if (data && n > 0) {
        const auto consider = [&](const MixDescriptor& m) {
            if (std::fabs(m.effective_bpw - target_bpw) > target_bpw * 0.2f + 0.05f) return;
            MixBlockPlan plan = allocate_mix_blocks(m, data, n, WIRE_BLOCK);
            if (plan.formats.empty()) return;
            double sum = 0.0;
            for (size_t b = 0; b < plan.formats.size(); b++) {
                const int64_t blen = plan.block_lens[(size_t)b];
                std::vector<uint8_t> idx, cb;
                if (!quantize_block_all(plan.formats[(size_t)b], data + plan.block_starts[(size_t)b],
                                        static_cast<int>(blen), idx, cb))
                    return;
                std::vector<float> dq(static_cast<size_t>(blen));
                dequantize_block_all(plan.formats[(size_t)b], idx.data(), idx.size(),
                                     cb.data(), cb.size(), static_cast<uint32_t>(blen), dq.data());
                for (int64_t i = 0; i < blen; i++) {
                    const double d = static_cast<double>(data[plan.block_starts[(size_t)b] + i]) -
                                     static_cast<double>(dq[static_cast<size_t>(i)]);
                    sum += d * d;
                }
            }
            const float mse = static_cast<float>(sum / static_cast<double>(n));
            if (mse < best_m) { best_m = mse; best = m; }
        };
        for (const auto& m : get_all_twi_mixes()) consider(m);
        for (const auto& m : get_all_four_mixes()) consider(m);
        if (!best.name.empty()) return best;
    }

    MixDescriptor best_two = get_twi_mix(target_bpw);
    MixDescriptor best_four = get_four_mix(target_bpw);
    float d2 = std::fabs(best_two.effective_bpw - target_bpw);
    float d4 = std::fabs(best_four.effective_bpw - target_bpw);
    if (d2 <= d4) return best_two;
    return best_four;
}

std::vector<FormatDescriptor> FormatRegistry::apply_forced_distribution(
        float target_bpw, int num_formats, const float* data, int64_t n) {
    std::vector<FormatDescriptor> result;
    if (num_formats <= 0) return result;
    const auto& singles = get_all_singles();

    // Quality-aware forced distribution: only formats at or under the target
    // BPW qualify, and they are ordered by MEASURED reconstruction error on
    // the given tensor (data is actually used). Falls back to BPW-closest
    // ordering without data.
    std::vector<std::pair<float, FormatDescriptor>> ranked;
    for (const auto& s : singles) {
        if (s.bpw > target_bpw + 1e-6f) continue;
        float mse = 1e30f;
        if (data && n > 0) mse = evaluate_format_quality(data, n, s);
        ranked.push_back({mse, s});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    const int count = std::min(num_formats, static_cast<int>(ranked.size()));
    for (int i = 0; i < count; i++) result.push_back(ranked[static_cast<size_t>(i)].second);
    if (!result.empty()) return result;

    std::vector<std::pair<float, int>> candidates;
    for (int i = 0; i < static_cast<int>(singles.size()); i++) {
        float d = std::fabs(singles[static_cast<size_t>(i)].bpw - target_bpw);
        candidates.push_back({d, i});
    }
    std::sort(candidates.begin(), candidates.end());
    const int count2 = std::min(num_formats, static_cast<int>(candidates.size()));
    for (int i = 0; i < count2; i++) {
        result.push_back(singles[static_cast<size_t>(candidates[static_cast<size_t>(i)].second)]);
    }
    return result;
}

float FormatRegistry::compute_average_bpw(const std::vector<FormatDescriptor>& assignment) {
    if (assignment.empty()) return 0.0f;
    double sum = 0.0;
    for (const auto& f : assignment) sum += f.bpw;
    return static_cast<float>(sum / static_cast<double>(assignment.size()));
}

std::string FormatRegistry::get_format_table() {
    const auto& singles = get_all_singles();
    std::ostringstream os;
    os << "Format          BPW   Centroids  Lossless  Grouped  GrpSize  MSE        Description\n";
    os << "--------------  ----  ---------  --------  -------  -------  ---------- ----------------\n";
    for (const auto& s : singles) {
        os << s.name;
        int pad = 16 - static_cast<int>(s.name.size());
        if (pad > 0) os << std::string(static_cast<size_t>(pad), ' ');
        os.width(4); os << std::right << s.bpw << "   ";
        os.width(9); os << std::right << s.num_centroids << "  ";
        os << (s.lossless ? "Yes" : "No ") << "      ";
        os << (s.grouped ? "Yes" : "No ") << "     ";
        os.width(7); os << std::right << s.group_size << "  ";
        os.width(10); os << std::right << s.est_mse << "  ";
        os << s.description << "\n";
    }
    return os.str();
}

FormatDescriptor FormatRegistry::parse_format_name(const std::string& name) {
    const auto& singles = get_all_singles();
    for (const auto& s : singles) {
        if (s.name == name) return s;
    }
    return {};
}

// Canonical wire accounting for a quantized payload: every byte needed to
// reconstruct the payload lives in the indices + codebook channels produced
// by the canonical block codec (group scales/zero-points are in-memory
// metadata and are never serialized to disk by the canonical path). The
// reported size is therefore the ACTUAL stored byte count — never a
// round-up of the nominal BPW claim.
size_t FormatRegistry::serialized_size_bytes(const QuantResult& qr) {
    if (!qr.success || qr.num_elements <= 0) return 0;
    return qr.indices.size() + qr.codebook.size();
}

float FormatRegistry::actual_bpw(const QuantResult& qr) {
    if (qr.num_elements <= 0) return 0.0f;
    return 8.0f * static_cast<float>(serialized_size_bytes(qr)) /
           static_cast<float>(qr.num_elements);
}

FormatRegistry::MixBlockPlan FormatRegistry::allocate_mix_blocks(
        const MixDescriptor& mix, const float* data, int64_t n,
        int block_size, const std::vector<int64_t>* shape) {
    MixBlockPlan plan;
    if (!data || n <= 0 || mix.num_tiers <= 0) return plan;
    const int bs = block_size > 0 ? block_size : 256;

    // Member ladder in ascending claimed-BPW order: every block starts on the
    // cheapest member and the greedy walker upgrades it step by step (this is
    // the priority-wise order: the highest-precision member is the most
    // expensive, so it is only reached where benefit per byte justifies it).
    std::vector<Format> ladder;
    const RegFormat tiers[4] = { mix.tier1_fmt, mix.tier2_fmt, mix.tier3_fmt, mix.tier4_fmt };
    for (int i = 0; i < mix.num_tiers && i < 4; i++)
        ladder.push_back(regformat_to_format(tiers[i]));
    std::sort(ladder.begin(), ladder.end(),
              [](Format a, Format b) { return format_bpw(a) < format_bpw(b); });
    ladder.erase(std::unique(ladder.begin(), ladder.end()), ladder.end());
    if (ladder.empty()) return plan;
    const int tiers_n = (int)ladder.size();

    // Block segmentation. Rank-2 tensors whose column count is below the
    // block size get one block per ROW (per-row scales, row/column-aligned);
    // everything else is flat fixed-size blocks, which are row-aligned
    // whenever the block size divides the column count.
    struct Seg { int64_t start; int64_t len; };
    std::vector<Seg> segs;
    if (shape && shape->size() >= 2 && (*shape)[1] > 0 && (*shape)[1] < bs) {
        const int64_t cols = (*shape)[1];
        const int64_t rows = n / cols;
        segs.reserve((size_t)rows);
        for (int64_t r = 0; r < rows; r++) segs.push_back({ r * cols, cols });
        if (rows * cols < n) segs.push_back({ rows * cols, n - rows * cols });
    } else {
        const int64_t nb = (n + bs - 1) / bs;
        segs.reserve((size_t)nb);
        for (int64_t b = 0; b < nb; b++)
            segs.push_back({ b * bs, std::min<int64_t>(bs, n - b * bs) });
    }
    const int nb = (int)segs.size();

    // HARD byte budget: the claimed effective BPW is a cap that is never
    // exceeded; bytes are spent only where they buy measurable quality.
    const int64_t budget =
        (int64_t)std::ceil((double)mix.effective_bpw * (double)n / 8.0);

    // Measure each member format on every block with the CANONICAL block
    // codec (quantize_block_all/dequantize_block_all — the exact codec the
    // .quant writer uses), so the benefit estimate matches the stored quality.
    // bytes[b][t] = ACTUAL canonical stored bytes (indices + codebook, never
    // the nominal claimed-bpw round-up); mse[b][t] = measured MSE.
    std::vector<std::vector<int64_t>> bytes((size_t)nb);
    std::vector<std::vector<float>> mse((size_t)nb);
    for (int b = 0; b < nb; b++) {
        bytes[(size_t)b].resize((size_t)tiers_n);
        mse[(size_t)b].resize((size_t)tiers_n, 1e30f);
        for (int t = 0; t < tiers_n; t++) {
            const Format f = ladder[(size_t)t];
            const int64_t blen = segs[(size_t)b].len;
            std::vector<uint8_t> idx, cb;
            if (!quantize_block_all(f, data + segs[(size_t)b].start, (int)blen, idx, cb))
                continue;
            bytes[(size_t)b][(size_t)t] = (int64_t)(idx.size() + cb.size());
            std::vector<float> dq((size_t)blen);
            dequantize_block_all(f, idx.data(), idx.size(), cb.data(), cb.size(),
                                 (uint32_t)blen, dq.data());
            mse[(size_t)b][(size_t)t] =
                measure_mse(data + segs[(size_t)b].start, dq.data(), blen);
        }
    }

    // Greedy benefit-per-byte upgrades under the hard budget. A candidate is
    // (block, ANY member format more precise than the current one) and is
    // accepted only if it measurably reduces MSE and the added bytes still
    // fit, so the final plan never exceeds the budget and spends every byte
    // where it buys the most quality. Considering every member format (not
    // only the next ladder step) keeps blocks from getting stuck on a
    // format the canonical codec happens to do poorly at.
    std::vector<int> state((size_t)nb, 0);
    int64_t used = 0;
    for (int b = 0; b < nb; b++) used += bytes[(size_t)b][0];

    struct Cand { float ratio; int b; int t; };
    struct Cmp {
        bool operator()(const Cand& a, const Cand& c) const {
            if (a.ratio != c.ratio) return a.ratio < c.ratio;
            return a.t < c.t;  // priority-wise: prefer the more precise tier on ties
        }
    };
    std::priority_queue<Cand, std::vector<Cand>, Cmp> pq;
    auto push_cand = [&](int b, int t) {
        if (t <= state[(size_t)b] || t >= tiers_n) return;
        const int64_t added = bytes[(size_t)b][(size_t)t] - bytes[(size_t)b][(size_t)state[(size_t)b]];
        if (added <= 0 || used + added > budget) return;  // unaffordable (only gets worse)
        const float benefit = mse[(size_t)b][(size_t)state[(size_t)b]] - mse[(size_t)b][(size_t)t];
        if (!(benefit > 0.0f)) return;  // no measurable improvement
        pq.push({ benefit / (float)added, b, t });
    };
    for (int b = 0; b < nb; b++)
        for (int t = state[(size_t)b] + 1; t < tiers_n; t++) push_cand(b, t);
    while (!pq.empty()) {
        const Cand c = pq.top();
        pq.pop();
        if (c.t <= state[(size_t)c.b] || c.t >= tiers_n) continue;
        const int64_t added = bytes[(size_t)c.b][(size_t)c.t] - bytes[(size_t)c.b][(size_t)state[(size_t)c.b]];
        if (added <= 0 || used + added > budget) continue;
        const float benefit = mse[(size_t)c.b][(size_t)state[(size_t)c.b]] - mse[(size_t)c.b][(size_t)c.t];
        if (!(benefit > 0.0f)) continue;
        used += added;
        state[(size_t)c.b] = c.t;
        for (int t = c.t + 1; t < tiers_n; t++) push_cand(c.b, t);
    }

    plan.formats.resize((size_t)nb);
    plan.block_starts.resize((size_t)nb);
    plan.block_lens.resize((size_t)nb);
    for (int b = 0; b < nb; b++) {
        plan.formats[(size_t)b] = ladder[(size_t)state[(size_t)b]];
        plan.block_starts[(size_t)b] = segs[(size_t)b].start;
        plan.block_lens[(size_t)b] = segs[(size_t)b].len;
    }
    return plan;
}

} // namespace quant