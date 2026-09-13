// kernel_q24.cpp — SIMD GEMM kernel for Q24 (24-bit direct, FP24)
// Q24 stores weights as 24-bit floats (16-bit mantissa + 8-bit exponent)
// 3 bytes per weight — no codebook needed
// This provides higher precision than FP16 (10-bit mantissa) with less
// storage than FP32 (23-bit mantissa), at a 25% savings vs FP32.
#include "quant/kernel.h"
#include "quant/tensor.h"
#include <cstring>
#include <cmath>
#include <cstdint>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace quant {
namespace kernel {

// Q24 format (Transcender custom FP24):
// Byte 0: exponent (8 bits, same as FP32 exponent, biased by 127)
// Byte 1-2: mantissa (16 bits, top 16 of 23 FP32 mantissa bits)
// Sign is stored in the MSB of the exponent byte (bit 7)
//
// Conversion: fp24 → fp32:
//   sign = byte0[7]
//   exponent = byte0[6:0] (7 bits) + bias adjustment
//   mantissa = (byte1 << 8 | byte2) << 7  (pad bottom 7 bits with zeros)
//
// Actually simpler: We store the top 24 bits of FP32 (sign + 8-bit exp + 15-bit mantissa)
// This means: just take the FP32 uint32 and drop the bottom 8 bits.
static inline float fp24_to_float(const uint8_t* src) {
    // Reconstruct FP32 from top 24 bits
    uint32_t f32 = ((uint32_t)src[0] << 24) | ((uint32_t)src[1] << 16) | ((uint32_t)src[2] << 8);
    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

static inline void float_to_fp24(float val, uint8_t* dst) {
    uint32_t f32;
    memcpy(&f32, &val, sizeof(f32));
    // Store top 24 bits, truncating bottom 8 mantissa bits
    dst[0] = (uint8_t)(f32 >> 24);
    dst[1] = (uint8_t)(f32 >> 16);
    dst[2] = (uint8_t)(f32 >> 8);
}

[[maybe_unused]] static void q24_gemm_scalar(const uint8_t* packed_weights,
                              const float* activations, float* output,
                              int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            const uint8_t* w_row = packed_weights + ((int64_t)m * N + n) * K * 3;
            const float* a_row = activations + (int64_t)n * K;

#if defined(_MSC_VER)
            _mm_prefetch(reinterpret_cast<const char*>(w_row + K * 3), _MM_HINT_T0);
#else
            __builtin_prefetch(w_row + K * 3, 0, 3);
#endif

            for (int k = 0; k < K; k++) {
                float w = fp24_to_float(w_row + k * 3);
                sum += w * a_row[k];
            }
            output[m * N + n] = sum;
        }
    }
}

#if defined(__AVX2__)
static void q24_gemm_avx2(const uint8_t* packed_weights,
                           const float* activations, float* output,
                           int M, int N, int K) {
    // Phase 7-G3 Q24 (D4): AVX2 3B->32b shuffle + slli 8-wide, bit-exact.
    // Wire BE: triple [b0,b1,b2]=F32[31:24,23:16,15:8]; F32 LE=[00,b2,b1,b0].
    // Pre-shift gather is byte-reversed [2,1,0] per triple; slli drops pad.
    const __m256i kQ24ShufBE = _mm256_setr_epi8(
        2,1,0,(char)0x80, 5,4,3,(char)0x80, 8,7,6,(char)0x80, 11,10,9,(char)0x80,
        2,1,0,(char)0x80, 5,4,3,(char)0x80, 8,7,6,(char)0x80, 11,10,9,(char)0x80);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            __m256 sum8 = _mm256_setzero_ps();
            const uint8_t* w_row = packed_weights + ((int64_t)m * N + n) * K * 3;
            const float* a_row = activations + (int64_t)n * K;

            int k = 0;
            for (; k + 8 <= K; k += 8) {
                // Unpack 8 FP24 weights to FP32 (overlapping 16B loads keep
                // each triple lane-local for _mm256_shuffle_epi8).
                const uint8_t* wp = w_row + (size_t)k * 3;
                __m256i v;
                if (k + 10 <= K) {
                    __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wp));
                    __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wp + 12));
                    v = _mm256_set_m128i(hi, lo);
                    v = _mm256_shuffle_epi8(v, kQ24ShufBE);
                } else {
                    // Final window of row: memcpy 24B to avoid 4B overread.
                    alignas(32) uint8_t tmp[32] = {0};
                    std::memcpy(tmp, wp, 24);
                    __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tmp));
                    __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tmp + 12));
                    v = _mm256_set_m128i(hi, lo);
                    v = _mm256_shuffle_epi8(v, kQ24ShufBE);
                }
                v = _mm256_slli_epi32(v, 8);
                __m256 w = _mm256_castsi256_ps(v);

#if defined(_MSC_VER)
                _mm_prefetch(reinterpret_cast<const char*>(w_row + (k + 8) * 3), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(a_row + k + 8), _MM_HINT_T0);
#else
                __builtin_prefetch(w_row + (k + 8) * 3, 0, 3);
                __builtin_prefetch(a_row + k + 8, 0, 3);
#endif

                __m256 a = _mm256_loadu_ps(a_row + k);
                sum8 = _mm256_fmadd_ps(w, a, sum8);
            }

            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(sum8, 1);
            __m128 lo = _mm256_castps256_ps128(sum8);
            __m128 sum4 = _mm_add_ps(lo, hi);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum4 = _mm_hadd_ps(sum4, sum4);
            float result = _mm_cvtss_f32(sum4);

            // Handle tail
            for (; k < K; k++) {
                float w = fp24_to_float(w_row + k * 3);
                result += w * a_row[k];
            }
            output[m * N + n] = result;
        }
    }
}
#endif

void q24_gemm(const uint8_t* packed_weights,
              const float* activations, float* output,
              int M, int N, int K) {
#if defined(__AVX2__)
    q24_gemm_avx2(packed_weights, activations, output, M, N, K);
#else
    q24_gemm_scalar(packed_weights, activations, output, M, N, K);
#endif
}

// Utility: encode FP32 weights to Q24 packed format
void q24_encode(const float* weights, uint8_t* packed, int count) {
    for (int i = 0; i < count; i++) {
        float_to_fp24(weights[i], packed + i * 3);
    }
}

// Utility: decode Q24 packed format to FP32 weights
void q24_decode(const uint8_t* packed, float* weights, int count) {
#if defined(__AVX2__)
    // Phase 7-G3 Q24 (D4): same BE shuffle + slli 8-wide as GEMM path.
    const __m256i kQ24ShufBE = _mm256_setr_epi8(
        2,1,0,(char)0x80, 5,4,3,(char)0x80, 8,7,6,(char)0x80, 11,10,9,(char)0x80,
        2,1,0,(char)0x80, 5,4,3,(char)0x80, 8,7,6,(char)0x80, 11,10,9,(char)0x80);
    int i = 0;
    for (; i + 8 <= count; i += 8) {
        const uint8_t* wp = packed + (size_t)i * 3;
        __m256i v;
        if (i + 10 <= count) {
            __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wp));
            __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wp + 12));
            v = _mm256_set_m128i(hi, lo);
            v = _mm256_shuffle_epi8(v, kQ24ShufBE);
        } else {
            alignas(32) uint8_t tmp[32] = {0};
            std::memcpy(tmp, wp, 24);
            __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tmp));
            __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tmp + 12));
            v = _mm256_set_m128i(hi, lo);
            v = _mm256_shuffle_epi8(v, kQ24ShufBE);
        }
        v = _mm256_slli_epi32(v, 8);
        _mm256_storeu_ps(weights + i, _mm256_castsi256_ps(v));
    }
    for (; i < count; i++) {
        weights[i] = fp24_to_float(packed + (size_t)i * 3);
    }
#else
    for (int i = 0; i < count; i++) {
        weights[i] = fp24_to_float(packed + i * 3);
    }
#endif
}

} // namespace kernel
} // namespace quant
