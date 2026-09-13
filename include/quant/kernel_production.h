// ============================================================================
// kernel_production.h — Production SIMD kernels for QUANT inference
// ----------------------------------------------------------------------------
// Tiled GEMV, batch GEMV, and calibration-aware kernels for:
//   Q1_5 (1.50-bit, 4 values/byte), QUANT4 (4-bit, 2/byte), QUANT8 (8-bit, 1/byte)
//
// ISA dispatch: AVX2 > SSE4.1 > Scalar (runtime CPU detection)
// ============================================================================
#pragma once
#include "quant/types.h"
#include <cstdint>
#include <cstddef>

namespace quant {
namespace adapters {
namespace prod {

// ── ISA detection ──────────────────────────────────────────────────────────
int detect_cpu_isa();  // 2=AVX2, 1=SSE4.1, 0=scalar

// ── Tiled GEMV (cache-blocked M dimension) ─────────────────────────────────
// y = scale * (W @ x), where W is in QUANT format.
// tile_m = rows processed per cache-blocking step (default: L1/line_size)

void gemv_quant_tiled(const uint8_t* packed_w, float scale,
                      const float* x, float* y,
                      int M, int K, int tile_m = 32);

void gemv_quant4_tiled(const uint8_t* packed_indices, const uint16_t* codebook,
                     float scale, const float* x, float* y,
                     int M, int K, int tile_m = 32);

void gemv_quant8_tiled(const uint8_t* indices, const float* codebook,
                     float scale, const float* x, float* y,
                     int M, int K, int tile_m = 32);

// ── Batch GEMV (multiple activation vectors) ──────────────────────────────
// Y[m, n] = scale * (W[m, :] @ X[n, :])
// M = output rows, N = batch size, K = inner dimension

void gemv_quant_batch(const uint8_t* packed_w, float scale,
                      const float* X, float* Y,
                      int M, int N, int K);

void gemv_quant4_batch(const uint8_t* packed_indices, const uint16_t* codebook,
                     float scale, const float* X, float* Y,
                     int M, int N, int K);

void gemv_quant8_batch(const uint8_t* indices, const float* codebook,
                     float scale, const float* X, float* Y,
                     int M, int N, int K);

// ── Calibration-aware importance scoring ──────────────────────────────────
// Given weights + sample activations, compute per-block importance scores.
// Higher score → more important → deserves QUANT8 or QUANT4.
// importance array must be pre-allocated with num_blocks elements.

void calibrate_quant_importance(const float* weights, const float* activations,
                                float* importance, int M, int K, int block_size);

void calibrate_quant4_importance(const float* weights, const float* activations,
                               float* importance, int M, int K, int block_size);

void calibrate_quant8_importance(const float* weights, const float* activations,
                               float* importance, int M, int K, int block_size);

// ── Performance profiling hooks ────────────────────────────────────────────
struct KernelPerf {
    uint64_t cycles = 0;
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;
    double   gflops = 0.0;
};

KernelPerf profile_gemv_quant(const uint8_t* packed_w, float scale,
                              const float* x, float* y, int M, int K, int repeats = 100);

KernelPerf profile_gemv_quant4(const uint8_t* packed_indices, const uint16_t* codebook,
                             float scale, const float* x, float* y, int M, int K, int repeats = 100);

KernelPerf profile_gemv_quant8(const uint8_t* indices, const float* codebook,
                             float scale, const float* x, float* y, int M, int K, int repeats = 100);

} // namespace prod
} // namespace adapters
} // namespace quant
