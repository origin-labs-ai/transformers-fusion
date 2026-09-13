#include "quant/quant_engines.h"
#include "quant/math.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace quant {
namespace engines {

// ===========================================================================
// AVX2 SIMD-accelerated quantize/dequantize batch operations
// ===========================================================================
#ifdef QUANT_HAS_AVX2

void dequant_tensor_quant8_avx2(const uint8_t* indices, float* out,
                                      int64_t n, const float* cb) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i idx8 = _mm_loadl_epi64((const __m128i*)(indices + i));
        __m256i idx32 = _mm256_cvtepu8_epi32(idx8);
        __m256 val = _mm256_i32gather_ps(cb, idx32, 4);
        _mm256_storeu_ps(out + i, val);
    }
    for (; i < n; ++i) out[i] = cb[indices[i]];
}

void dequant_tensor_quant4_avx2(const uint8_t* packed, float* out,
                                      int64_t n, const float* cb) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint32_t p4;
        memcpy(&p4, packed + i / 2, 4);
        uint8_t idxs[8];
        idxs[0] = p4 & 0xF; idxs[1] = (p4 >> 4) & 0xF;
        idxs[2] = (p4 >> 8) & 0xF; idxs[3] = (p4 >> 12) & 0xF;
        idxs[4] = (p4 >> 16) & 0xF; idxs[5] = (p4 >> 20) & 0xF;
        idxs[6] = (p4 >> 24) & 0xF; idxs[7] = (p4 >> 28) & 0xF;
        __m128i idx8 = _mm_loadl_epi64((const __m128i*)idxs);
        __m256i idx32 = _mm256_cvtepu8_epi32(idx8);
        __m256 val = _mm256_i32gather_ps(cb, idx32, 4);
        _mm256_storeu_ps(out + i, val);
    }
    for (; i < n; ++i) {
        uint8_t code = (i % 2 == 0) ? (packed[i / 2] & 0xF)
                                     : (packed[i / 2] >> 4);
        out[i] = cb[code];
    }
}

// AVX2 quant_gemm: QUANT8 (256-entry codebook LUT)
void quant_gemm_quant8_avx2(const float* ad, float* cd,
                                  const uint8_t* b_idx,
                                  int64_t M, int64_t N, int64_t K,
                                  const float* cb) {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            __m256 a_v = _mm256_set1_ps(a_val);
            int64_t n = 0;
            for (; n + 8 <= N; n += 8) {
                __m128i idx8 = _mm_loadl_epi64(
                    (const __m128i*)(b_idx + k * N + n));
                __m256i idx32 = _mm256_cvtepu8_epi32(idx8);
                __m256 b_v = _mm256_i32gather_ps(cb, idx32, 4);
                __m256 c_v = _mm256_loadu_ps(cd + m * N + n);
                _mm256_storeu_ps(cd + m * N + n,
                                 _mm256_fmadd_ps(a_v, b_v, c_v));
            }
            for (; n < N; ++n)
                cd[m * N + n] += a_val * cb[b_idx[k * N + n]];
        }
    }
}

// AVX2 quant_gemm: QUANT4 (packed 4-bit indices, 16-entry codebook)
void quant_gemm_quant4_avx2(const float* ad, float* cd,
                                  const uint8_t* packed,
                                  int64_t M, int64_t N, int64_t K,
                                  const float* cb) {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            __m256 a_v = _mm256_set1_ps(a_val);
            int64_t n = 0;
            for (; n + 8 <= N; n += 8) {
                uint32_t p4;
                memcpy(&p4, packed + (k * N + n) / 2, 4);
                uint8_t idxs[8];
                idxs[0] = p4 & 0xF; idxs[1] = (p4 >> 4) & 0xF;
                idxs[2] = (p4 >> 8) & 0xF; idxs[3] = (p4 >> 12) & 0xF;
                idxs[4] = (p4 >> 16) & 0xF; idxs[5] = (p4 >> 20) & 0xF;
                idxs[6] = (p4 >> 24) & 0xF; idxs[7] = (p4 >> 28) & 0xF;
                __m128i idx8 = _mm_loadl_epi64((const __m128i*)idxs);
                __m256i idx32 = _mm256_cvtepu8_epi32(idx8);
                __m256 b_v = _mm256_i32gather_ps(cb, idx32, 4);
                __m256 c_v = _mm256_loadu_ps(cd + m * N + n);
                _mm256_storeu_ps(cd + m * N + n,
                                 _mm256_fmadd_ps(a_v, b_v, c_v));
            }
            for (; n < N; ++n) {
                uint8_t code = (n % 2 == 0) ? (packed[(k * N + n) / 2] & 0xF)
                                            : (packed[(k * N + n) / 2] >> 4);
                cd[m * N + n] += a_val * cb[code];
            }
        }
    }
}

// AVX2 quant_gemm: QUANT add/sub only (no multiply in inner loop)
void quant_gemm_quant_avx2(const float* ad, float* cd,
                                     const uint8_t* pd, const float* sd,
                                     int64_t M, int64_t N, int64_t K,
                                     int64_t block_size) {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = ad[m * K + k];
            if (a_val == 0.0f) continue;
            int64_t n = 0;
            while (n < N) {
                int64_t flat = k * N + n;
                int64_t block_idx = flat / block_size;
                float a_scl = a_val * sd[block_idx];
                __m256 a_scl_v = _mm256_set1_ps(a_scl);
                int64_t block_end = std::min(N, (block_idx + 1) * block_size - k * N);
                for (; n + 8 <= block_end; n += 8) {
                    uint32_t p32;
                    int64_t byte_base = (k * N + n) / 4;
                    memcpy(&p32, pd + byte_base, 4);
                    int shift = (int)((k * N + n) % 4) * 2;
                    uint32_t codes4 = p32 >> shift;
                    __m256i codes = _mm256_setr_epi32(
                        (int)(codes4 >> 0) & 3, (int)(codes4 >> 2) & 3,
                        (int)(codes4 >> 4) & 3, (int)(codes4 >> 6) & 3,
                        (int)(codes4 >> 8) & 3, (int)(codes4 >> 10) & 3,
                        (int)(codes4 >> 12) & 3, (int)(codes4 >> 14) & 3);
                    __m256i mask_pos = _mm256_cmpeq_epi32(codes, _mm256_set1_epi32(2));
                    __m256i mask_neg = _mm256_cmpeq_epi32(codes, _mm256_setzero_si256());
                    __m256 pos_sel = _mm256_blendv_ps(
                        _mm256_setzero_ps(), a_scl_v, _mm256_castsi256_ps(mask_pos));
                    __m256 neg_sel = _mm256_blendv_ps(
                        _mm256_setzero_ps(), _mm256_sub_ps(_mm256_setzero_ps(), a_scl_v),
                        _mm256_castsi256_ps(mask_neg));
                    __m256 delta = _mm256_add_ps(pos_sel, neg_sel);
                    __m256 c_v = _mm256_loadu_ps(cd + m * N + n);
                    _mm256_storeu_ps(cd + m * N + n, _mm256_add_ps(c_v, delta));
                }
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
}

#endif // QUANT_HAS_AVX2

} // namespace engines
} // namespace quant
