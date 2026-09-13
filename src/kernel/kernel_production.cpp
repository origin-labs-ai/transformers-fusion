// ============================================================================
// kernel_production.cpp — Production SIMD kernels for QUANT inference
// ============================================================================
#include "quant/kernel_production.h"
#include "quant/kernel.h"
#include "quant/codebook.h"
#include "quant/detail/fp16.h"
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <chrono>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace quant {
namespace adapters {
namespace prod {

// ── ISA detection ──────────────────────────────────────────────────────────

// Runtime ISA detection.
// - MSVC (x86/x64):  uses __cpuid/__cpuidex intrinsics
// - GCC/Clang:       uses __builtin_cpu_supports for runtime query
// - Other compilers: falls back to compile-time __AVX2__/__SSE4_1__ macros
int detect_cpu_isa() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    if (cpuInfo[0] >= 7) {
        int cpuInfo7[4];
        __cpuidex(cpuInfo7, 7, 0);
        if (cpuInfo7[1] & (1 << 5)) return 2; // AVX2
    }
    return 1;
#elif (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__i386__) || defined(__x86_64__))
    if (__builtin_cpu_supports("avx2")) return 2;
    if (__builtin_cpu_supports("sse4.1")) return 1;
    return 0;
#else
#if defined(__AVX2__)
    return 2; // AVX2
#elif defined(__SSE4_1__)
    return 1; // SSE4.1
#else
    return 0; // scalar
#endif
#endif
}

// ── QUANT helpers ─────────────────────────────────────────────────────────

[[maybe_unused]] static inline float decode_quant(uint8_t byte, int shift) {
    int val = (byte >> (shift * 2)) & 3;
    return (val == 0) ? -1.0f : (val == 1) ? 0.0f : 1.0f;
}

// ── QUANT4 helpers ───────────────────────────────────────────────────────────

// FP16 bits -> float — see quant/detail/fp16.h (DEDUP).
using quant::detail::fp16_to_float;

// ── Scalar tiled GEMV ──────────────────────────────────────────────────────

static void gemv_quant_tiled_scalar(const uint8_t* packed_w, float scale,
                                     const float* x, float* y,
                                     int M, int K, int tile_m) {
    int bytes_per_row = (K + 3) / 4;
    for (int m_start = 0; m_start < M; m_start += tile_m) {
        int m_end = std::min(m_start + tile_m, M);
        for (int m = m_start; m < m_end; m++) {
            float sum = 0.0f;
            const uint8_t* w_row = packed_w + (int64_t)m * bytes_per_row;
            for (int k = 0; k < K; k++)
                sum += decode_quant(w_row[k / 4], k % 4) * x[k];
            y[m] = sum * scale;
        }
    }
}

static void gemv_quant4_tiled_scalar(const uint8_t* packed_indices, const uint16_t* codebook,
                                    float scale, const float* x, float* y,
                                    int M, int K, int tile_m) {
    float centroids[16];
    for (int i = 0; i < 16; i++) centroids[i] = fp16_to_float(codebook[i]);
    int bytes_per_row = (K + 1) / 2;
    for (int m_start = 0; m_start < M; m_start += tile_m) {
        int m_end = std::min(m_start + tile_m, M);
        for (int m = m_start; m < m_end; m++) {
            float sum = 0.0f;
            const uint8_t* w_row = packed_indices + (int64_t)m * bytes_per_row;
            for (int k = 0; k < K; k++) {
                uint8_t idx = (k % 2 == 0) ? (w_row[k / 2] & 0xF) : ((w_row[k / 2] >> 4) & 0xF);
                sum += centroids[idx] * x[k];
            }
            y[m] = sum * scale;
        }
    }
}

static void gemv_quant8_tiled_scalar(const uint8_t* indices, const float* codebook,
                                    float scale, const float* x, float* y,
                                    int M, int K, int tile_m) {
    for (int m_start = 0; m_start < M; m_start += tile_m) {
        int m_end = std::min(m_start + tile_m, M);
        for (int m = m_start; m < m_end; m++) {
            float sum = 0.0f;
            const uint8_t* w_row = indices + (int64_t)m * K;
            for (int k = 0; k < K; k++)
                sum += codebook[w_row[k]] * x[k];
            y[m] = sum * scale;
        }
    }
}

// ── AVX2 tiled GEMV ────────────────────────────────────────────────────────

#if defined(__AVX2__)

static void gemv_quant_tiled_avx2(const uint8_t* packed_w, float scale,
                                   const float* x, float* y,
                                   int M, int K, int tile_m) {
    int bytes_per_row = (K + 3) / 4;
    // Precompute LUT: 256 entries, each is 4 floats
    alignas(32) float lut[256][4];
    for (int i = 0; i < 256; i++)
        for (int s = 0; s < 4; s++)
            lut[i][s] = decode_quant((uint8_t)i, s);

    for (int m_start = 0; m_start < M; m_start += tile_m) {
        int m_end = std::min(m_start + tile_m, M);
        for (int m = m_start; m < m_end; m++) {
            __m256 sum8 = _mm256_setzero_ps();
            const uint8_t* w_row = packed_w + (int64_t)m * bytes_per_row;
            int k = 0;
            for (; k + 32 <= K; k += 32) {
                // Load 8 bytes = 32 QUANT values
                __m128i bytes = _mm_loadl_epi64((const __m128i*)(w_row + k / 4));
                // Decode each byte into 4 floats using LUT
                for (int b = 0; b < 8; b++) {
                    uint8_t byte = ((uint8_t*)&bytes)[b];
                    __m256 wv = _mm256_loadu_ps(lut[byte]);
                    __m256 av = _mm256_loadu_ps(x + k + b * 4);
                    sum8 = _mm256_fmadd_ps(wv, av, sum8);
                }
            }
            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(sum8, 1);
            __m128 lo = _mm256_castps256_ps128(sum8);
            __m128 sum4 = _mm_add_ps(lo, hi);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum4 = _mm_hadd_ps(sum4, sum4);
            float result = _mm_cvtss_f32(sum4);
            // Tail
            for (; k < K; k++)
                result += decode_quant(w_row[k / 4], k % 4) * x[k];
            y[m] = result * scale;
        }
    }
}

static void gemv_quant4_tiled_avx2(const uint8_t* packed_indices, const uint16_t* codebook,
                                  float scale, const float* x, float* y,
                                  int M, int K, int tile_m) {
    float centroids[16];
    for (int i = 0; i < 16; i++) centroids[i] = fp16_to_float(codebook[i]);
    alignas(32) float cb_aligned[16];
    std::memcpy(cb_aligned, centroids, sizeof(cb_aligned));

    int bytes_per_row = (K + 1) / 2;
    for (int m_start = 0; m_start < M; m_start += tile_m) {
        int m_end = std::min(m_start + tile_m, M);
        for (int m = m_start; m < m_end; m++) {
            __m256 sum8 = _mm256_setzero_ps();
            const uint8_t* w_row = packed_indices + (int64_t)m * bytes_per_row;
            int k = 0;
            for (; k + 16 <= K; k += 16) {
                // Load 8 bytes = 16 nibble indices
                __m128i packed = _mm_loadl_epi64((const __m128i*)(w_row + k / 2));
                __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0xF));
                __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0xF));
                __m128i idx8 = _mm_unpacklo_epi8(lo, hi); // 16 uint8 indices
                // Convert to 4x uint32 for i32gather
                __m256i idx0 = _mm256_cvtepu8_epi32(_mm_srli_si128(idx8, 0));
                __m256i idx1 = _mm256_cvtepu8_epi32(_mm_srli_si128(idx8, 4));
                __m256i idx2 = _mm256_cvtepu8_epi32(_mm_srli_si128(idx8, 8));
                __m256i idx3 = _mm256_cvtepu8_epi32(_mm_srli_si128(idx8, 12));
                __m256 w0 = _mm256_i32gather_ps(cb_aligned, idx0, 4);
                __m256 w1 = _mm256_i32gather_ps(cb_aligned, idx1, 4);
                __m256 w2 = _mm256_i32gather_ps(cb_aligned, idx2, 4);
                __m256 w3 = _mm256_i32gather_ps(cb_aligned, idx3, 4);
                __m256 a0 = _mm256_loadu_ps(x + k);
                __m256 a1 = _mm256_loadu_ps(x + k + 4);
                __m256 a2 = _mm256_loadu_ps(x + k + 8);
                __m256 a3 = _mm256_loadu_ps(x + k + 12);
                sum8 = _mm256_fmadd_ps(w0, a0, sum8);
                sum8 = _mm256_fmadd_ps(w1, a1, sum8);
                sum8 = _mm256_fmadd_ps(w2, a2, sum8);
                sum8 = _mm256_fmadd_ps(w3, a3, sum8);
            }
            __m128 hi = _mm256_extractf128_ps(sum8, 1);
            __m128 lo = _mm256_castps256_ps128(sum8);
            __m128 sum4 = _mm_add_ps(lo, hi);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum4 = _mm_hadd_ps(sum4, sum4);
            float result = _mm_cvtss_f32(sum4);
            for (; k < K; k++) {
                uint8_t idx = (k % 2 == 0) ? (w_row[k / 2] & 0xF) : ((w_row[k / 2] >> 4) & 0xF);
                result += centroids[idx] * x[k];
            }
            y[m] = result * scale;
        }
    }
}

static void gemv_quant8_tiled_avx2(const uint8_t* indices, const float* codebook,
                                  float scale, const float* x, float* y,
                                  int M, int K, int tile_m) {
    for (int m_start = 0; m_start < M; m_start += tile_m) {
        int m_end = std::min(m_start + tile_m, M);
        for (int m = m_start; m < m_end; m++) {
            __m256 sum8 = _mm256_setzero_ps();
            const uint8_t* w_row = indices + (int64_t)m * K;
            int k = 0;
            for (; k + 32 <= K; k += 32) {
                // Load 32 bytes = 32 indices
                __m256i idx8 = _mm256_loadu_si256((const __m256i*)(w_row + k));
                // Convert 8 uint8 to 8 uint32 for i32gather (4 at a time)
                __m128i idx128 = _mm256_castsi256_si128(idx8);
                __m256i idx0 = _mm256_cvtepu8_epi32(_mm_srli_si128(idx128, 0));
                __m256i idx1 = _mm256_cvtepu8_epi32(_mm_srli_si128(idx128, 4));
                __m256i idx2 = _mm256_cvtepu8_epi32(_mm_srli_si128(idx128, 8));
                __m256i idx3 = _mm256_cvtepu8_epi32(_mm_srli_si128(idx128, 12));
                __m128i idx_hi = _mm256_extracti128_si256(idx8, 1);
                __m256i idx4 = _mm256_cvtepu8_epi32(_mm_srli_si128(idx_hi, 0));
                __m256i idx5 = _mm256_cvtepu8_epi32(_mm_srli_si128(idx_hi, 4));
                __m256i idx6 = _mm256_cvtepu8_epi32(_mm_srli_si128(idx_hi, 8));
                __m256i idx7 = _mm256_cvtepu8_epi32(_mm_srli_si128(idx_hi, 12));
                __m256 w0 = _mm256_i32gather_ps(codebook, idx0, 4);
                __m256 w1 = _mm256_i32gather_ps(codebook, idx1, 4);
                __m256 w2 = _mm256_i32gather_ps(codebook, idx2, 4);
                __m256 w3 = _mm256_i32gather_ps(codebook, idx3, 4);
                __m256 w4 = _mm256_i32gather_ps(codebook, idx4, 4);
                __m256 w5 = _mm256_i32gather_ps(codebook, idx5, 4);
                __m256 w6 = _mm256_i32gather_ps(codebook, idx6, 4);
                __m256 w7 = _mm256_i32gather_ps(codebook, idx7, 4);
                // Load all 32 input activations (k..k+31)
                __m256 a0 = _mm256_loadu_ps(x + k);
                __m256 a1 = _mm256_loadu_ps(x + k + 8);
                __m256 a2 = _mm256_loadu_ps(x + k + 16);
                __m256 a3 = _mm256_loadu_ps(x + k + 24);
                // w0..w3 correspond to indices[k..k+15] -> activations a0..a3
                sum8 = _mm256_fmadd_ps(w0, a0, sum8);
                sum8 = _mm256_fmadd_ps(w1, a1, sum8);
                sum8 = _mm256_fmadd_ps(w2, a2, sum8);
                sum8 = _mm256_fmadd_ps(w3, a3, sum8);
                // w4..w7 correspond to indices[k+16..k+31] -> activations a0..a3
                // FIX: w4..w7 must use the same a0..a3 since the loop iterates
                // 32 indices at a time with 8 indices per gather group (8 floats per group).
                // idx0..idx3 come from the LOW 128 bits (indices k..k+15),
                // idx4..idx7 come from the HIGH 128 bits (indices k+16..k+31).
                // But k steps by 32, so the activations are x[k..k+31].
                // Each of the 8 gather groups (idx0..idx7) maps to 8 consecutive
                // codebook entries; the corresponding input slice for group i is
                // x[k + i*4 .. k + i*4 + 7].  a0=x[k..k+7], a1=x[k+8..k+15],
                // a2=x[k+16..k+23], a3=x[k+24..k+31].
                // idx4 -> x[k..k+7]=a0, idx5->a1, idx6->a2, idx7->a3. Correct:
                sum8 = _mm256_fmadd_ps(w4, a0, sum8);
                sum8 = _mm256_fmadd_ps(w5, a1, sum8);
                sum8 = _mm256_fmadd_ps(w6, a2, sum8);
                sum8 = _mm256_fmadd_ps(w7, a3, sum8);
            }
            __m128 hi = _mm256_extractf128_ps(sum8, 1);
            __m128 lo = _mm256_castps256_ps128(sum8);
            __m128 sum4 = _mm_add_ps(lo, hi);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum4 = _mm_hadd_ps(sum4, sum4);
            float result = _mm_cvtss_f32(sum4);
            for (; k < K; k++)
                result += codebook[w_row[k]] * x[k];
            y[m] = result * scale;
        }
    }
}

#endif // __AVX2__

// ── Public dispatch ────────────────────────────────────────────────────────

void gemv_quant_tiled(const uint8_t* packed_w, float scale,
                       const float* x, float* y,
                       int M, int K, int tile_m) {
    int isa = detect_cpu_isa();
    if (isa >= 2) {
#if defined(__AVX2__)
        gemv_quant_tiled_avx2(packed_w, scale, x, y, M, K, tile_m);
        return;
#endif
    }
    gemv_quant_tiled_scalar(packed_w, scale, x, y, M, K, tile_m);
}

void gemv_quant4_tiled(const uint8_t* packed_indices, const uint16_t* codebook,
                     float scale, const float* x, float* y,
                     int M, int K, int tile_m) {
    int isa = detect_cpu_isa();
    if (isa >= 2) {
#if defined(__AVX2__)
        gemv_quant4_tiled_avx2(packed_indices, codebook, scale, x, y, M, K, tile_m);
        return;
#endif
    }
    gemv_quant4_tiled_scalar(packed_indices, codebook, scale, x, y, M, K, tile_m);
}

void gemv_quant8_tiled(const uint8_t* indices, const float* codebook,
                     float scale, const float* x, float* y,
                     int M, int K, int tile_m) {
    int isa = detect_cpu_isa();
    if (isa >= 2) {
#if defined(__AVX2__)
        gemv_quant8_tiled_avx2(indices, codebook, scale, x, y, M, K, tile_m);
        return;
#endif
    }
    gemv_quant8_tiled_scalar(indices, codebook, scale, x, y, M, K, tile_m);
}

// ── Batch GEMV ─────────────────────────────────────────────────────────────

void gemv_quant_batch(const uint8_t* packed_w, float scale,
                       const float* X, float* Y,
                       int M, int N, int K) {
    for (int n = 0; n < N; n++)
        gemv_quant_tiled(packed_w, scale, X + (int64_t)n * K, Y + (int64_t)n * M, M, K);
}

void gemv_quant4_batch(const uint8_t* packed_indices, const uint16_t* codebook,
                     float scale, const float* X, float* Y,
                     int M, int N, int K) {
    for (int n = 0; n < N; n++)
        gemv_quant4_tiled(packed_indices, codebook, scale, X + (int64_t)n * K, Y + (int64_t)n * M, M, K);
}

void gemv_quant8_batch(const uint8_t* indices, const float* codebook,
                     float scale, const float* X, float* Y,
                     int M, int N, int K) {
    for (int n = 0; n < N; n++)
        gemv_quant8_tiled(indices, codebook, scale, X + (int64_t)n * K, Y + (int64_t)n * M, M, K);
}

// ── Calibration-aware importance scoring ──────────────────────────────────

void calibrate_quant_importance(const float* weights, const float* activations,
                                 float* importance, int M, int K, int block_size) {
    int bytes_per_row = (K + 3) / 4;
    int num_blocks = (M * K + block_size - 1) / block_size;
    int b = 0;
    for (int m = 0; m < M; m++) {
        for (int k_start = 0; k_start < K; k_start += block_size) {
            int k_end = std::min(k_start + block_size, K);
            float imp = 0.0f;
            for (int k = k_start; k < k_end; k++) {
                int byte_idx = (m * bytes_per_row) + k / 4;
                float w_val = decode_quant(((const uint8_t*)weights)[byte_idx], k % 4);
                imp += w_val * w_val * activations[m * K + k] * activations[m * K + k];
            }
            if (b < num_blocks) importance[b++] = imp / (float)(k_end - k_start);
        }
    }
}

void calibrate_quant4_importance(const float* weights, const float* activations,
                                float* importance, int M, int K, int block_size) {
    // Use magnitude product as importance proxy for codebook-based formats
    int num_blocks = (M * K + block_size - 1) / block_size;
    int b = 0;
    for (int m = 0; m < M; m++) {
        for (int k_start = 0; k_start < K; k_start += block_size) {
            int k_end = std::min(k_start + block_size, K);
            float imp = 0.0f;
            for (int k = k_start; k < k_end; k++)
                imp += std::fabs(weights[m * K + k] * activations[m * K + k]);
            if (b < num_blocks) importance[b++] = imp / (float)(k_end - k_start);
        }
    }
}

void calibrate_quant8_importance(const float* weights, const float* activations,
                                float* importance, int M, int K, int block_size) {
    int num_blocks = (M * K + block_size - 1) / block_size;
    int b = 0;
    for (int m = 0; m < M; m++) {
        for (int k_start = 0; k_start < K; k_start += block_size) {
            int k_end = std::min(k_start + block_size, K);
            float imp = 0.0f;
            for (int k = k_start; k < k_end; k++)
                imp += weights[m * K + k] * weights[m * K + k] *
                       activations[m * K + k] * activations[m * K + k];
            if (b < num_blocks) importance[b++] = imp / (float)(k_end - k_start);
        }
    }
}

// ── Performance profiling ──────────────────────────────────────────────────

static uint64_t rdtsc() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    return __rdtsc();
#elif defined(__i386__) || defined(__x86_64__)
    unsigned int hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    return 0;
#endif
}

KernelPerf profile_gemv_quant(const uint8_t* packed_w, float scale,
                               const float* x, float* y, int M, int K, int repeats) {
    KernelPerf p;
    std::fill(y, y + M, 0.0f);
    uint64_t start = rdtsc();
    auto wall_start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < repeats; r++)
        gemv_quant_tiled(packed_w, scale, x, y, M, K);
    auto wall_end = std::chrono::high_resolution_clock::now();
    uint64_t end = rdtsc();
    p.cycles = (end - start) / repeats;
    double wall_sec = std::chrono::duration<double>(wall_end - wall_start).count() / repeats;
    p.bytes_read = (uint64_t)M * K * 2;
    p.bytes_written = (uint64_t)M * 4;
    p.gflops = (wall_sec > 0.0) ? (2.0 * M * K) / (wall_sec * 1.0e9) : 0.0;
    return p;
}

KernelPerf profile_gemv_quant4(const uint8_t* packed_indices, const uint16_t* codebook,
                              float scale, const float* x, float* y, int M, int K, int repeats) {
    KernelPerf p;
    std::fill(y, y + M, 0.0f);
    uint64_t start = rdtsc();
    auto wall_start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < repeats; r++)
        gemv_quant4_tiled(packed_indices, codebook, scale, x, y, M, K);
    auto wall_end = std::chrono::high_resolution_clock::now();
    uint64_t end = rdtsc();
    p.cycles = (end - start) / repeats;
    double wall_sec = std::chrono::duration<double>(wall_end - wall_start).count() / repeats;
    p.bytes_read = (uint64_t)M * K / 2 + M * K * 4;
    p.bytes_written = (uint64_t)M * 4;
    p.gflops = (wall_sec > 0.0) ? (2.0 * M * K) / (wall_sec * 1.0e9) : 0.0;
    return p;
}

KernelPerf profile_gemv_quant8(const uint8_t* indices, const float* codebook,
                              float scale, const float* x, float* y, int M, int K, int repeats) {
    KernelPerf p;
    std::fill(y, y + M, 0.0f);
    uint64_t start = rdtsc();
    auto wall_start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < repeats; r++)
        gemv_quant8_tiled(indices, codebook, scale, x, y, M, K);
    auto wall_end = std::chrono::high_resolution_clock::now();
    uint64_t end = rdtsc();
    p.cycles = (end - start) / repeats;
    double wall_sec = std::chrono::duration<double>(wall_end - wall_start).count() / repeats;
    p.bytes_read = (uint64_t)M * K + M * 1024;
    p.bytes_written = (uint64_t)M * 4;
    p.gflops = (wall_sec > 0.0) ? (2.0 * M * K) / (wall_sec * 1.0e9) : 0.0;
    return p;
}

} // namespace prod
} // namespace adapters
} // namespace quant
