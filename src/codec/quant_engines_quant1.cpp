#include "quant/quant_engines.h"
#include "quant/math.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace quant {
namespace engines {

// ===========================================================================
// Quant1 Engine: Block mean (1 FP32 centroid per 32 elements)
// Matches FormatRegistry::quantize_quant1() behavior
// ===========================================================================

Quant1Engine::Quant1Engine() {}

Tensor Quant1Engine::quantize(const Tensor& weight) {
    int64_t n = weight.numel();
    int64_t block_size = 32;
    int64_t num_blocks = (n + block_size - 1) / block_size;
    Tensor out({num_blocks});
    const float* wd = weight.data<float>();
    float* od = out.data<float>();
    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t start = b * block_size;
        int64_t end = std::min(start + block_size, n);
        double sum = 0;
        for (int64_t i = start; i < end; i++) sum += wd[i];
        od[b] = static_cast<float>(sum / static_cast<double>(end - start));
    }
    return out;
}

Tensor Quant1Engine::dequantize(const Tensor& packed, float, int64_t n) {
    int64_t block_size = 32;
    int64_t num_blocks = (n + block_size - 1) / block_size;
    const float* pd = packed.data<float>();
    Tensor out({n});
    float* od = out.data<float>();
    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t start = b * block_size;
        int64_t end = std::min(start + block_size, n);
        float val = pd[b];
        for (int64_t i = start; i < end; i++) od[i] = val;
    }
    return out;
}

// ===========================================================================
// Quant1 Engine: Extensions
// ===========================================================================

Tensor Quant1Engine::quantize_batch(const Tensor& t) {
    return quantize(t);
}

Tensor Quant1Engine::dequantize_batch(const Tensor& q) {
    return dequantize(q, 0.0f, 0);
}

Tensor Quant1Engine::quant_gemm(const Tensor& a, const Tensor& b_packed,
                                 float, int64_t M, int64_t N, int64_t K) {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    float* cd = C.data<float>();
    const float* bd = b_packed.data<float>();
    int64_t block_size = 32;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            for (int64_t n = 0; n < N; ++n) {
                int64_t flat_idx = k * N + n;
                int64_t block_idx = flat_idx / block_size;
                float b_val = bd[block_idx];
                cd[m * N + n] += a_val * b_val;
            }
        }
    }
    return C;
}

void Quant1Engine::quantize_per_channel(const Tensor& t, int channel_dim,
                                          Tensor& q, Tensor& scales) {
    QUANT_CHECK(t.rank() == 2, "Quant1 per-channel expects 2D tensor");
    int64_t d0 = t.dim(0), d1 = t.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    int64_t block_size = 32;
    int64_t num_blocks = (t.numel() + block_size - 1) / block_size;
    q = Tensor({num_blocks});
    scales = Tensor({channels});
    const float* td = t.data<float>();
    float* qd = q.data<float>();
    float* sd = scales.data<float>();
    for (int64_t c = 0; c < channels; ++c) {
        float max_abs = 0;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            max_abs = std::max(max_abs, std::abs(td[idx]));
        }
        sd[c] = max_abs;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * channels + c;
            int64_t block_idx = idx / block_size;
            double sum = 0; int cnt = 0;
            for (int64_t j = (block_idx * block_size); j < std::min((block_idx + 1) * block_size, t.numel()); j++) {
                if (channel_dim == 0) { int64_t r = j / other; if (r == c) { sum += td[j]; cnt++; } }
                else { int64_t col = j % channels; if (col == c) { sum += td[j]; cnt++; } }
            }
            qd[block_idx] = cnt > 0 ? static_cast<float>(sum / cnt) : 0.0f;
        }
    }
}

void Quant1Engine::dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                            int channel_dim, Tensor& out) {
    (void)scales; (void)channel_dim;
    float* od = out.data<float>();
    const float* qd = q.data<float>();
    int64_t block_size = 32;
    int64_t n = out.numel();
    int64_t num_blocks = (n + block_size - 1) / block_size;
    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t start = b * block_size;
        int64_t end = std::min(start + block_size, n);
        float val = qd[b];
        for (int64_t i = start; i < end; i++) od[i] = val;
    }
}

float Quant1Engine::quant_error(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_mse(original, reconstructed);
}

float Quant1Engine::quant_snr(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_snr(original, reconstructed);
}

} // namespace engines
} // namespace quant
