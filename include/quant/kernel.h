#pragma once
#include "quant/types.h"
#include "quant/tensor.h"
#include "quant/codebook.h"
#include <cstdint>

namespace quant {
namespace kernel {

// TL1: QUANT Lookup Table, groups of 2
void tl1_gemm(const Tensor& weights, const Tensor& activations,
              Tensor& output, int M, int N, int K);
void tl1_precompute_lut(const int8_t* activations, int8_t* lut,
                        int K, float scales);

// TL2: QUANT Lookup Table, groups of 3 (element-wise mirror consolidation)
void tl2_gemm(const Tensor& weights, const Tensor& activations,
              Tensor& output, int M, int N, int K);
void tl2_precompute_lut(const int8_t* activations, int8_t* lut,
                        int K, float scales);

// Q8: Codebook lookup GEMM
// weights = uint8_t indices, codebook = float[256], activations = float
void quant8_gemm(const uint8_t* indices, const float* codebook,
               const float* activations, float* output,
               int M, int N, int K);

// Q4: Codebook lookup GEMM
// weights = nibble packed indices, codebook = uint16_t[16] (FP16), activations = float
void quant4_gemm(const uint8_t* packed_indices, const uint16_t* codebook,
               const float* activations, float* output,
               int M, int N, int K);

// Q3: 3-bit codebook lookup GEMM (8 FP32 centroids, 3-bit packed indices)
void q3_gemm(const uint8_t* packed_indices, const float* codebook,
             const float* activations, float* output,
             int M, int N, int K);

// Q6: 6-bit codebook lookup GEMM (64 FP32 centroids, 6-bit packed indices)
void q6_gemm(const uint8_t* packed_indices, const float* codebook,
             const float* activations, float* output,
             int M, int N, int K);

// Q12: 12-bit codebook lookup GEMM (4096 FP16 centroids, 12-bit packed indices)
void q12_gemm(const uint8_t* packed_indices, const uint16_t* codebook_fp16,
              const float* activations, float* output,
              int M, int N, int K);

// Q24: 24-bit direct GEMM (FP24, 3 bytes per weight)
void q24_gemm(const uint8_t* packed_weights,
              const float* activations, float* output,
              int M, int N, int K);
void q24_encode(const float* weights, uint8_t* packed, int count);
void q24_decode(const uint8_t* packed, float* weights, int count);

// Scalar fallback matmul
void scalar_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K);

// C[M,N] = A[M,K] x B[N,K]^T — B stored row-major as {N, K} (weights {out, in})
void scalar_gemm_bt(const float* A, const float* B, float* C,
                    int M, int N, int K);

// AVX2 variant of scalar_gemm_bt
void avx2_gemm_bt(const float* A, const float* B, float* C,
                  int M, int N, int K);

// Tiled GEMM with 64x64 blocking for cache efficiency
void tiled_gemm(const float* A, const float* B, float* C,
                int M, int N, int K);

// AVX2 matmul
void avx2_gemm(const float* A, const float* B, float* C,
               int M, int N, int K);

// AVX2 tiled GEMM
void avx2_tiled_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K);

} // namespace kernel
} // namespace quant
