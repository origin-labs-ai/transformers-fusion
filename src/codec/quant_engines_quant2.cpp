#include "quant/quant_engines.h"
#include "quant/math.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

namespace quant {
namespace engines {

// ===========================================================================
// QUANT2 Engine: 4-entry FP32 codebook, 2-bit indices (4 per byte)
// ===========================================================================

QUANT2Engine::QUANT2Engine() {
    codebook_.resize(4);
    for (int i = 0; i < 4; ++i)
        codebook_[(size_t)i] = (float)(i - 1) * 0.5f;
}

void QUANT2Engine::train_codebook(const float* data, int64_t n) {
    if (n == 0) return;
    float dmin = data[0], dmax = data[0];
    for (int64_t i = 1; i < n; ++i) {
        dmin = std::min(dmin, data[i]);
        dmax = std::max(dmax, data[i]);
    }
    if (dmax - dmin < 1e-10f) dmax = dmin + 1.0f;
    for (int j = 0; j < 4; ++j)
        codebook_[(size_t)j] = dmin + (dmax - dmin) * (float)j / 3.0f;
    for (int iter = 0; iter < 20; ++iter) {
        std::vector<int> counts(4, 0);
        std::vector<double> sums(4, 0.0);
        for (int64_t i = 0; i < n; ++i) {
            int best = 0;
            float best_dist = 1e10f;
            for (int j = 0; j < 4; ++j) {
                float dist = std::abs(data[i] - codebook_[(size_t)j]);
                if (dist < best_dist) { best_dist = dist; best = j; }
            }
            counts[best]++;
            sums[best] += data[i];
        }
        for (int j = 0; j < 4; ++j) {
            if (counts[j] > 0)
                codebook_[(size_t)j] = (float)(sums[j] / counts[j]);
        }
    }
}

void QUANT2Engine::train_codebook_per_block(const float* data, int64_t n,
                                           int64_t block_size, int lloyd_iters) {
    int64_t num_blocks = n / block_size;
    if (n % block_size != 0) num_blocks++;

    std::vector<float> normalized(n);
    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t start = b * block_size;
        int64_t end = (std::min)(start + block_size, n);
        float bmin = data[start], bmax = data[start];
        for (int64_t i = start + 1; i < end; i++) {
            bmin = (std::min)(bmin, data[i]);
            bmax = (std::max)(bmax, data[i]);
        }
        float half_range = (bmax - bmin) * 0.5f;
        float center = (bmax + bmin) * 0.5f;
        if (half_range < 1e-10f) half_range = 1.0f;
        for (int64_t i = start; i < end; i++)
            normalized[(size_t)i] = (data[i] - center) / half_range;
    }

    for (int i = 0; i < 4; ++i)
        codebook_[(size_t)i] = -1.0f + 2.0f * (float)i / 3.0f;

    for (int iter = 0; iter < lloyd_iters; ++iter) {
        std::vector<int> counts(4, 0);
        std::vector<double> sums(4, 0.0);
        for (int64_t i = 0; i < n; ++i) {
            int best = 0;
            float best_dist = 1e10f;
            for (int j = 0; j < 4; ++j) {
                float dist = std::abs(normalized[(size_t)i] - codebook_[(size_t)j]);
                if (dist < best_dist) { best_dist = dist; best = j; }
            }
            counts[best]++;
            sums[best] += normalized[(size_t)i];
        }
        bool changed = false;
        for (int j = 0; j < 4; ++j) {
            if (counts[j] > 0) {
                float new_val = (float)(sums[j] / (double)counts[j]);
                if (std::abs(new_val - codebook_[(size_t)j]) > 1e-8f) changed = true;
                codebook_[(size_t)j] = new_val;
            }
        }
        if (!changed) break;
    }
}

void QUANT2Engine::quantize_per_block(const float* data, int64_t n, int64_t block_size,
                                     uint8_t* indices_out, float* scales_out) const {
    int64_t num_blocks = n / block_size;
    if (n % block_size != 0) num_blocks++;

    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t start = b * block_size;
        int64_t end = (std::min)(start + block_size, n);
        int64_t len = end - start;

        float bmin = data[start], bmax = data[start];
        for (int64_t i = start + 1; i < end; i++) {
            bmin = (std::min)(bmin, data[i]);
            bmax = (std::max)(bmax, data[i]);
        }
        float half_range = (bmax - bmin) * 0.5f;
        float center = (bmax + bmin) * 0.5f;
        if (half_range < 1e-10f) half_range = 1.0f;
        scales_out[(size_t)b * 2 + 0] = center;
        scales_out[(size_t)b * 2 + 1] = half_range;

        for (int64_t i = 0; i < len; i++) {
            float normalized = (data[start + i] - center) / half_range;
            int best = 0;
            float best_dist = 1e10f;
            for (int j = 0; j < 4; ++j) {
                float dist = std::abs(normalized - codebook_[(size_t)j]);
                if (dist < best_dist) { best_dist = dist; best = j; }
            }
            indices_out[start + i] = (uint8_t)best;
        }
    }
}

void QUANT2Engine::dequantize_per_block(const uint8_t* indices, const float* scales,
                                       int64_t n, int64_t block_size, float* out) const {
    int64_t num_blocks = n / block_size;
    if (n % block_size != 0) num_blocks++;

    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t start = b * block_size;
        int64_t end = (std::min)(start + block_size, n);
        float center = scales[(size_t)b * 2 + 0];
        float half_range = scales[(size_t)b * 2 + 1];
        for (int64_t i = start; i < end; i++) {
            out[i] = codebook_[(size_t)indices[i]] * half_range + center;
        }
    }
}

uint8_t QUANT2Engine::quantize(float val) const {
    if (use_stochastic_) {
        // Two-nearest-centroid stochastic rounding: unbiased and minimum
        // variance. BUG FIXED 2026-09-12 — this was a softmax over all 4
        // centroids with weight exp(-d/T), which depends on the raw distance
        // rather than the local scale, is far too flat, and sampled almost
        // uniformly across the codebook (same defect as QUANT8, where it made
        // MSE ~100x worse than argmin).
        int i1 = 0, i2 = 0;
        float d1 = 1e10f, d2 = 1e10f;
        for (int i = 0; i < 4; ++i) {
            float d = std::abs(val - codebook_[(size_t)i]);
            if (d < d1) { d2 = d1; i2 = i1; d1 = d; i1 = i; }
            else if (d < d2) { d2 = d; i2 = i; }
        }
        if (!(d1 + d2 > 1e-30f)) return (uint8_t)i1;
        static thread_local std::mt19937 rng(42);
        const float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
        return (uint8_t)(r < d1 / (d1 + d2) ? i2 : i1);
    }
    int best = 0;
    float best_dist = 1e10f;
    for (int i = 0; i < 4; ++i) {
        float dist = std::abs(val - codebook_[(size_t)i]);
        if (dist < best_dist) { best_dist = dist; best = i; }
    }
    return (uint8_t)best;
}

float QUANT2Engine::dequantize(uint8_t idx) const {
    if (idx >= 4) idx = 0;
    return codebook_[idx];
}

Tensor QUANT2Engine::dequant_tensor(const uint8_t* indices, int64_t n) const {
    Tensor out({n});
    float* od = out.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        uint8_t packed = indices[i / 4];
        uint8_t code = (uint8_t)((packed >> ((i % 4) * 2)) & 0x3);
        od[i] = dequantize(code);
    }
    return out;
}

Tensor QUANT2Engine::quantize_tensor(const float* data, int64_t n) const {
    int64_t packed_size = (n + 3) / 4;
    Tensor out({packed_size}, quant::DType::U8);
    uint8_t* od = out.data<uint8_t>();
    for (int64_t i = 0; i < n; i += 4) {
        uint8_t byte = 0;
        for (int j = 0; j < 4; ++j) {
            uint8_t code = (i + j < n) ? quantize(data[i + j]) : 0;
            byte |= (code << (j * 2));
        }
        od[i / 4] = byte;
    }
    return out;
}

Tensor QUANT2Engine::quant_gemm(const Tensor& a, const uint8_t* b_idx,
                               int64_t M, int64_t N, int64_t K) const {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    float* cd = C.data<float>();
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            for (int64_t n = 0; n < N; ++n) {
                int64_t flat = k * N + n;
                uint8_t packed = b_idx[flat / 4];
                uint8_t code = (uint8_t)((packed >> ((flat % 4) * 2)) & 0x3);
                cd[m * N + n] += a_val * dequantize(code);
            }
        }
    }
    return C;
}

void QUANT2Engine::quantize_per_channel(const Tensor& t, int channel_dim,
                                       Tensor& q, Tensor& scales) const {
    QUANT_CHECK(t.rank() == 2, "QUANT2 per-channel expects 2D tensor");
    int64_t d0 = t.dim(0), d1 = t.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    int64_t packed_size = (t.numel() + 3) / 4;
    q = Tensor({packed_size}, quant::DType::U8);
    scales = Tensor({channels});
    const float* td = t.data<float>();
    uint8_t* qd = q.data<uint8_t>();
    float* sd = scales.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float max_abs = 0;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            max_abs = std::max(max_abs, std::abs(td[idx]));
        }
        sd[c] = max_abs;
        if (max_abs < 1e-10f) max_abs = 1.0f;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            float normalized = td[idx] / max_abs;
            uint8_t qv = quantize(normalized);
            int64_t packed_idx = idx / 4;
            int bit_off = (int)(idx % 4) * 2;
            qd[packed_idx] = (uint8_t)((qd[packed_idx] & ~(0x3 << bit_off)) | (qv << bit_off));
        }
    }
}

void QUANT2Engine::dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                         int channel_dim, Tensor& out) const {
    int64_t total = scales.numel();
    int64_t d0 = total, d1 = 1;
    if (channel_dim == 0) { d0 = total; d1 = q.numel() * 4 / total; }
    else { d1 = total; d0 = q.numel() * 4 / total; }
    out = Tensor({d0, d1});
    const uint8_t* qd = q.data<uint8_t>();
    const float* sd = scales.data<float>();
    float* od = out.data<float>();
    for (int64_t c = 0; c < total; ++c) {
        float scale = sd[c];
        int64_t other = out.numel() / total;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * total + c;
            uint8_t packed = qd[idx / 4];
            uint8_t code = (uint8_t)((packed >> ((idx % 4) * 2)) & 0x3);
            od[idx] = dequantize(code) * scale;
        }
    }
}

float QUANT2Engine::quant_error(const Tensor& original, const Tensor& reconstructed) const {
    return compute_quant_mse(original, reconstructed);
}

float QUANT2Engine::quant_snr(const Tensor& original, const Tensor& reconstructed) const {
    return compute_quant_snr(original, reconstructed);
}

} // namespace engines
} // namespace quant