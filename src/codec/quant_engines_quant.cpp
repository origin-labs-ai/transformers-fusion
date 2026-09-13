#include "quant/quant_engines.h"
#include "quant/math.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace quant {
namespace engines {

// ===========================================================================
// QUANT Engine: {-1, 0, +1} with per-block scale (BitNet b1.58)
// ===========================================================================

QuantEngine::QuantEngine(int64_t block_size) : block_size_(block_size) {}

Tensor QuantEngine::quantize(const Tensor& weight) {
    int64_t n = weight.numel();
    int64_t num_blocks = (n + block_size_ - 1) / block_size_;
    int64_t packed_size = (n + 3) / 4;
    int64_t packed_floats = (packed_size + 3) / 4;
    int64_t hdr = 2;
    // Layout: [num_blocks, packed_size, scales[num_blocks], packed_bytes...]
    Tensor out({hdr + num_blocks + packed_floats});
    out.zero_();
    float* rd = out.data<float>();
    rd[0] = (float)num_blocks;
    rd[1] = (float)packed_size;
    float* sd = rd + hdr;
    uint8_t* pd = reinterpret_cast<uint8_t*>(sd + num_blocks);
    const float* wd = weight.data<float>();
    for (int64_t b = 0; b < num_blocks; ++b) {
        int64_t start = b * block_size_;
        int64_t end = std::min(start + block_size_, n);
        float max_abs = 0;
        for (int64_t i = start; i < end; ++i)
            max_abs = std::max(max_abs, std::abs(wd[i]));
        sd[b] = max_abs;
        if (max_abs < 1e-10f) max_abs = 1.0f;
        for (int64_t i = start; i < end; ++i) {
            float normalized = wd[i] / max_abs;
            int8_t quant_val;
            if (normalized > 0.33f) quant_val = 1;
            else if (normalized < -0.33f) quant_val = -1;
            else quant_val = 0;
            uint8_t code = (uint8_t)((quant_val + 1) & 0x3);
            int64_t byte_idx = i / 4;
            int bit_off = (int)(i % 4) * 2;
            if (bit_off == 0)
                pd[byte_idx] = code;
            else
                pd[byte_idx] |= (code << bit_off);
        }
    }
    return out;
}

Tensor QuantEngine::dequantize(const Tensor& packed, const Tensor& scales, int64_t n) {
    const float* rd = packed.data<float>();
    int64_t num_blocks = (int64_t)rd[0];
    int64_t packed_size = (int64_t)rd[1];
    // BUGFIX (bug census): packed_size (embedded payload length) was read
    // then (void)-discarded — truncated buffers decoded garbage instead of
    // failing. Validate it covers the payload before touching pd[].
    int64_t hdr = 2;
    if (num_blocks < 0 || packed_size < 0) throw Error("QuantEngine::dequantize: corrupt header");
    if (hdr + num_blocks + (packed_size + 3) / 4 > packed.numel())
        throw Error("QuantEngine::dequantize: truncated buffer");
    if (n < 0 || n > num_blocks * block_size_ + block_size_)
        throw Error("QuantEngine::dequantize: bad output length");
    const float* sd = rd + hdr;
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(sd + num_blocks);
    // Use embedded scales if passed scales are insufficient (dummy)
    const float* scale_ptr = sd;
    int64_t scale_count = num_blocks;
    if (scales.numel() >= num_blocks) {
        scale_ptr = scales.data<float>();
        scale_count = scales.numel();
    }
    Tensor out({n});
    float* od = out.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        int64_t block_idx = i / block_size_;
        float scale = (block_idx < scale_count) ? scale_ptr[block_idx] : 0.0f;
        int64_t byte_idx = i / 4;
        int bit_off = (int)(i % 4) * 2;
        uint8_t code = (pd[byte_idx] >> bit_off) & 0x3;
        float quant_val = (code == 0) ? -1.0f : ((code == 2) ? 1.0f : 0.0f);
        od[i] = quant_val * scale;
    }
    return out;
}

// ===========================================================================
// QUANT Engine: Extensions
// ===========================================================================

Tensor QuantEngine::quantize_batch(const Tensor& t) {
    int64_t n = t.numel();
    int64_t num_blocks = (n + block_size_ - 1) / block_size_;
    int64_t packed_size = (n + 3) / 4;
    int64_t hdr = 3;
    int64_t packed_floats = (packed_size + 3) / 4;
    Tensor out({hdr + packed_floats + num_blocks});
    out.zero_();
    float* rd = out.data<float>();
    rd[0] = (float)n;
    rd[1] = (float)block_size_;
    rd[2] = (float)num_blocks;
    uint8_t* pd = reinterpret_cast<uint8_t*>(rd + hdr);
    float* sd = rd + hdr + packed_floats;
    const float* wd = t.data<float>();
    for (int64_t b = 0; b < num_blocks; ++b) {
        int64_t start = b * block_size_;
        int64_t end = std::min(start + block_size_, n);
        float max_abs = 0;
        for (int64_t i = start; i < end; ++i)
            max_abs = std::max(max_abs, std::abs(wd[i]));
        sd[b] = max_abs;
        if (max_abs < 1e-10f) max_abs = 1.0f;
        for (int64_t i = start; i < end; ++i) {
            float normalized = wd[i] / max_abs;
            int8_t quant_val;
            if (normalized > 0.33f) quant_val = 1;
            else if (normalized < -0.33f) quant_val = -1;
            else quant_val = 0;
            uint8_t code = (uint8_t)((quant_val + 1) & 0x3);
            int64_t byte_idx = i / 4;
            int bit_off = (int)(i % 4) * 2;
            if (bit_off == 0)
                pd[byte_idx] = code;
            else
                pd[byte_idx] |= (code << bit_off);
        }
    }
    return out;
}

Tensor QuantEngine::dequantize_batch(const Tensor& q) {
    const float* rd = q.data<float>();
    int64_t n = (int64_t)rd[0];
    int64_t block_size = (int64_t)rd[1];
    int64_t packed_size = (n + 3) / 4;
    int64_t hdr = 3;
    int64_t packed_floats = (packed_size + 3) / 4;
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(rd + hdr);
    const float* sd = rd + hdr + packed_floats;
    Tensor out({n});
    float* od = out.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        int64_t block_idx = i / block_size;
        float scale = sd[block_idx];
        int64_t byte_idx = i / 4;
        int bit_off = (int)(i % 4) * 2;
        uint8_t code = (pd[byte_idx] >> bit_off) & 0x3;
        float quant_val = (code == 0) ? -1.0f : ((code == 2) ? 1.0f : 0.0f);
        od[i] = quant_val * scale;
    }
    return out;
}

Tensor QuantEngine::quant_gemm(const Tensor& a, const Tensor& b_packed,
                                   const Tensor& b_scales, int64_t M, int64_t N, int64_t K) {
    Tensor C({M, N});
    C.zero_();
    const float* ad = a.data<float>();
    float* cd = C.data<float>();
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(b_packed.data<float>());
    const float* sd = b_scales.data<float>();
#ifdef QUANT_HAS_AVX2
    quant_gemm_quant_avx2(ad, cd, pd, sd, M, N, K, block_size_);
#else
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            int64_t n = 0;
            while (n < N) {
                int64_t flat = k * N + n;
                int64_t block_idx = flat / block_size_;
                float a_scl = a_val * sd[block_idx];
                int64_t block_end = std::min(N,
                    (block_idx + 1) * block_size_ - k * N);
                for (; n < block_end; ++n) {
                    int64_t flat2 = k * N + n;
                    int64_t byte_idx = flat2 / 4;
                    int bit_off = (int)(flat2 % 4) * 2;
                    uint8_t code = (pd[byte_idx] >> bit_off) & 0x3;
                    if (code == 2) cd[m * N + n] += a_scl;
                    else if (code == 0) cd[m * N + n] -= a_scl;
                }
            }
        }
    }
#endif
    return C;
}

void QuantEngine::quantize_per_channel(const Tensor& t, int channel_dim,
                                          Tensor& q, Tensor& scales) {
    QUANT_CHECK(t.rank() == 2, "QUANT per-channel expects 2D tensor");
    int64_t d0 = t.dim(0), d1 = t.dim(1);
    int64_t channels = (channel_dim == 0) ? d0 : d1;
    int64_t other = (channel_dim == 0) ? d1 : d0;
    int64_t n = t.numel();
    int64_t packed_size = (n + 3) / 4;
    q = Tensor({packed_size});
    q.zero_();
    scales = Tensor({channels});
    const float* td = t.data<float>();
    uint8_t* pd = reinterpret_cast<uint8_t*>(q.data<float>());
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
            int8_t quant_val;
            if (normalized > 0.33f) quant_val = 1;
            else if (normalized < -0.33f) quant_val = -1;
            else quant_val = 0;
            uint8_t code = (uint8_t)((quant_val + 1) & 0x3);
            int64_t byte_idx = idx / 4;
            int bit_off = (int)(idx % 4) * 2;
            if (bit_off == 0)
                pd[byte_idx] = code;
            else
                pd[byte_idx] |= (code << bit_off);
        }
    }
}

void QuantEngine::dequantize_per_channel(const Tensor& q, const Tensor& scales,
                                            int channel_dim, Tensor& out) {
    int64_t n = scales.numel();
    int64_t d0 = n, d1 = 1;
    if (channel_dim == 0) { d0 = n; d1 = q.numel() * 4 / n; }
    else { d1 = n; d0 = q.numel() * 4 / n; }
    out = Tensor({d0, d1});
    float* od = out.data<float>();
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(q.data<float>());
    const float* sd = scales.data<float>();
    for (int64_t c = 0; c < n; ++c) {
        float scale = sd[c];
        int64_t other = out.numel() / n;
        for (int64_t i = 0; i < other; ++i) {
            int64_t idx = (channel_dim == 0) ? c * other + i : i * n + c;
            int64_t byte_idx = idx / 4;
            int bit_off = (int)(idx % 4) * 2;
            uint8_t code = (pd[byte_idx] >> bit_off) & 0x3;
            float quant_val = (code == 0) ? -1.0f : ((code == 2) ? 1.0f : 0.0f);
            od[idx] = quant_val * scale;
        }
    }
}

float QuantEngine::quant_error(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_mse(original, reconstructed);
}

float QuantEngine::quant_snr(const Tensor& original, const Tensor& reconstructed) {
    return compute_quant_snr(original, reconstructed);
}

} // namespace engines
} // namespace quant
